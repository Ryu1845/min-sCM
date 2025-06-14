import argparse
import math
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from dit import DiT_Llama
from trigflow import load_trigflow_checkpoint


# Adaptive Weighting Network
class AdaptiveWeighting(nn.Module):
    def __init__(self, time_embedding_dim=64, hidden_dim=128):
        super().__init__()
        # A simple MLP to learn w_phi(t)
        # Input: t (scalar for each batch item)
        # Output: w_phi(t) (scalar for each batch item)
        self.mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim), # Takes t reshaped to (B, 1)
            nn.SiLU(),
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs w_phi(t)
        )

    def forward(self, t):
        # t is expected to be of shape (B,)
        # Reshape t to (B, 1) for the first linear layer
        return self.mlp(t.unsqueeze(-1))

# TrigFlow sCM: f_theta(x_t, t, y) = cos(t) * x_t - sin(t) * sigma_d * F_theta(x_t/sigma_d, t, y)
class sCM:
    def __init__(self, model, ema_model, sigma_d=1.0):
        self.model = model
        self.sigma_d = sigma_d
        self.ema_model = ema_model  # EMA of student model

    def forward(self, x_t, t, y):
        # x_t: (B, C, H, W), t: (B,), y: (B,)
        cos_t = torch.cos(t).view(-1, 1, 1, 1)
        sin_t = torch.sin(t).view(-1, 1, 1, 1)
        x_scaled = x_t / self.sigma_d
        F_out = self.model(x_scaled, t, y)
        return cos_t * x_t - sin_t * self.sigma_d * F_out

    def forward_with_tangent(self, x_t, t_orig, y, dx_dt, tangent_warmup_factor_r=1.0):
        # x_t: (B, C, H, W), t_orig: (B,), y: (B,), dx_dt: (B, C, H, W)
        # tangent_warmup_factor_r: scalar for warmup (r in Algorithm 1)
        # Implementation following Algorithm 1 from the paper appendix

        # F_theta^- is the target model (EMA of the student model), used for computing the tangent.
        # The teacher model is used to compute dx_dt for sCD, which is passed in.
        target_model = self.ema_model.module

        # Ensure t is float for JVP and trig functions
        t = t_orig.float()

        cos_t = torch.cos(t).view(-1, 1, 1, 1)
        sin_t = torch.sin(t).view(-1, 1, 1, 1)

        x_scaled = x_t / self.sigma_d

        # Compute F_theta^- (stopped gradient) using target model
        with torch.no_grad():
            F_theta_stopped = target_model(x_scaled, t_orig, y).detach()

        # --------  JVP rearrangement for numerical stability (Sec 4.1 of paper)  --------
        # Instead of computing dF_θ⁻/dt directly, we use JVP rearrangement as described in the paper:
        # cos(t)sin(t) * dF_θ⁻/dt = JVP with scaled tangents, then divide by cos(t)sin(t)

        scale_cs = cos_t * sin_t  # shape (B,1,1,1)
        safe_scale = scale_cs + 1e-8  # prevent divide-by-zero

        # JVP w.r.t. x_scaled with scaled tangents
        with torch.no_grad():
            dx_scaled_dt = dx_dt / self.sigma_d
            scaled_x_tangent = scale_cs * dx_scaled_dt
            _, dF_dx_scaled = target_model.forward_with_jvp(
                x_scaled.detach(),
                scaled_x_tangent.detach(),
                t_orig,
                y,
            )
        dF_dx_part = dF_dx_scaled.detach() / safe_scale  # unscale

        # JVP w.r.t. t with scaled tangents
        with torch.no_grad():
            scaled_t_tangent = scale_cs.squeeze() * self.sigma_d # (B,)
            _, dF_dt_scaled = target_model.forward_with_jvp_time(
                x_scaled.detach(),
                t_orig,
                scaled_t_tangent.detach(),
                y,
            )
        dF_dt_part = dF_dt_scaled.detach() / safe_scale

        # Total derivative dF_theta^-/dt
        dF_theta_stopped_dt = dF_dx_part + dF_dt_part

        # Algorithm 1: Compute tangent g
        # g ← -cos²(t)(σ_d F_θ⁻ - dx_t/dt) - r·cos(t)sin(t)(x_t + σ_d dF_θ⁻/dt)
        term1 = -cos_t * cos_t * (self.sigma_d * F_theta_stopped - dx_dt)  # -cos²(t) term
        term2 = -tangent_warmup_factor_r * cos_t * sin_t * (x_t + self.sigma_d * dF_theta_stopped_dt)  # -r·cos(t)sin(t) term

        # Clamp the terms to prevent extreme values
        term1 = torch.clamp(term1, -1e6, 1e6)
        term2 = torch.clamp(term2, -1e6, 1e6)

        g_tangent = term1 + term2

        # Algorithm 1: Tangent normalization
        # g ← g / (||g|| + c)
        b = g_tangent.size(0)
        tangent_norm = g_tangent.view(b, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)  # Shape (B, 1)
        c_tangent_norm = 0.1  # constant c from algorithm
        g_normalized = g_tangent / (tangent_norm.view(-1, 1, 1, 1) + c_tangent_norm)

        # Consistency model output f_theta (current model, not stopped)
        F_theta_current = self.model(x_scaled, t_orig, y)
        f = cos_t * x_t - sin_t * self.sigma_d * F_theta_current

        return f, g_normalized, F_theta_current, F_theta_stopped

    def set_ema_model(self, ema_model):
        """Set the EMA model for the student model."""
        self.ema_model = ema_model

    def update_ema(self):
        """Update the EMA model parameters."""
        self.ema_model.update_parameters(self.model)

    def get_sampling_model(self):
        """Get the model to use for sampling (always EMA model)."""
        return self.ema_model.module

    @torch.no_grad()
    def sample(self, z, y, steps=1, use_ema=True):
        # Single-step or two-step consistency sampling
        device = z.device

        # Temporarily use sampling model (EMA if available and requested)
        if use_ema:
            original_model = self.model
            self.model = self.get_sampling_model()

        try:
            # According to the paper, sampling starts from a high t.
            # t_max = arctan(sigma_max / sigma_d), with sigma_max=80.
            t_max_val = math.atan(80.0 / self.sigma_d)
            t_max = torch.full((z.shape[0],), t_max_val, device=device)

            if steps == 1:
                # One-step sampling: f(x_{t_max}, t_max)
                result = self.forward(z, t_max, y)
            elif steps == 2:
                # Two-step sampling following the consistency model sampling process.
                # The paper specifies t=1.1 as the intermediate step.
                t_mid = torch.full((z.shape[0],), 1.1, device=device)

                # 1. Get an estimate of x0 from noise z (which is x_tmax)
                x0_est = self.forward(z, t_max, y)

                # 2. Rennoise the estimate to t_mid
                # x_t = cos(t)*x_0 + sin(t)*noise
                noise = torch.randn_like(z)
                x_mid_renoisy = torch.cos(t_mid).view(-1, 1, 1, 1) * x0_est + torch.sin(t_mid).view(-1, 1, 1, 1) * noise

                # 3. Denoise again from t_mid to get the final estimate
                result = self.forward(x_mid_renoisy, t_mid, y)
            else:
                raise NotImplementedError("Only 1 or 2 step sampling supported.")
        finally:
            # Restore original model
            if use_ema:
                self.model = original_model

        return result


def main():
    parser = argparse.ArgumentParser(description="Train sCD (simplified Consistency Distillation) on MNIST or CIFAR-10")
    parser.add_argument("--cifar", action="store_true", help="Use CIFAR-10 instead of MNIST")
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to pretrained TrigFlow teacher model")
    parser.add_argument("--teacher_ema_path", type=str, help="Path to pretrained TrigFlow teacher EMA model (optional)")
    args = parser.parse_args()

    CIFAR = args.cifar

    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        channels = 3
        # Student model (consistency model)
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, class_dropout_prob=0.0
        ).cuda()
        # Teacher model (TrigFlow diffusion model)
        teacher_model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, class_dropout_prob=0.0
        ).cuda()
    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        channels = 1
        # Student model (consistency model)
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, class_dropout_prob=0.0001
        ).cuda()
        # Teacher model (TrigFlow diffusion model)
        teacher_model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, class_dropout_prob=0.0001
        ).cuda()

    # Load pretrained teacher model using the checkpoint loader
    teacher_model, teacher_trigflow, teacher_checkpoint = load_trigflow_checkpoint(args.teacher_path, device='cuda')
    teacher_model.eval()

    # Get the teacher model configuration from the checkpoint
    teacher_config = teacher_checkpoint['trigflow_config']['model_config']

    # Recreate student model with the same configuration as teacher to avoid shape mismatches
    # Use same class_dropout_prob as teacher to ensure embedding table size matches
    model = DiT_Llama(
        channels, 32,
        dim=teacher_config['dim'],
        n_layers=teacher_config['n_layers'],
        n_heads=teacher_config['n_heads'],
        num_classes=teacher_config['num_classes'],
        class_dropout_prob=teacher_config['class_dropout_prob']  # Match teacher's class_dropout_prob
    ).cuda()

    # Initialize student model from teacher for better convergence (sCD)
    model.load_state_dict(teacher_model.state_dict())

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6:.2f}M")
    batch_size = 64

    # EMA model for the student consistency model
    half_life_imgs = 0.5 * 1e6
    ema_decay = math.exp(math.log(0.5) / (half_life_imgs / batch_size))
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    # Create sCM (student consistency model)
    scm = sCM(model, ema_model)

    # Adaptive weighting network
    adaptive_weighting_net = AdaptiveWeighting().cuda()
    all_params = list(model.parameters()) + list(adaptive_weighting_net.parameters())
    optimizer = torch.optim.RAdam(all_params, lr=0.0001, betas=(0.9, 0.99), eps=1e-8)

    dataset = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    wandb.init(project=f"scd_{dataset_name}", config={
        "dataset": dataset_name,
        "batch_size": batch_size,
        "teacher_path": args.teacher_path,
        "method": "sCD"
    })

    # Dimensionality D of x0 (C*H*W)
    D = float(channels * 32 * 32)

    # Create contents directory if it doesn't exist
    os.makedirs("contents", exist_ok=True)

    for epoch in range(100):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}

        for i, (x0, y) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}"):
            # Tangent warmup factor: linearly increase from 0 to 1 over first portion of training
            global_iter = epoch * len(dataloader) + i
            warmup_iters = 10000  # Warmup over first 10k iterations (H=10000 from paper)
            tangent_warmup_factor = min(1.0, global_iter / warmup_iters)

            x0, y = x0.cuda(), y.cuda()
            b = x0.size(0)

            # Time sampling following paper settings
            # CIFAR-10: P_mean=-1.0, P_std=1.4 (from paper appendix)
            # MNIST: Similar log-normal sampling
            P_mean = -1.0
            P_std = 1.4
            sigma_d_val = scm.sigma_d

            # Sample time points using log-normal proposal (Algorithm 1)
            # τ ~ N(P_mean, P_std²), t = arctan(e^τ / σ_d)
            tau = torch.randn(b, device=x0.device) * P_std + P_mean
            t_sampled = torch.atan(torch.exp(tau) / sigma_d_val)
            t_sampled = torch.clamp(t_sampled, 0.01, math.pi/2 - 0.01)

            # Forward process using TrigFlow
            x_t, z, v_t = teacher_trigflow.forward_process(x0, t_sampled)

            # sCD: Get PF-ODE dx_t/dt from teacher model
            with torch.no_grad():
                dx_dt = teacher_trigflow.velocity_prediction(x_t, t_sampled, y)

            optimizer.zero_grad()

            # Compute tangent using JVP-based method (Algorithm 1)
            f, g_normalized, F_theta_current, F_theta_stopped = scm.forward_with_tangent(
                x_t, t_sampled, y, dx_dt, tangent_warmup_factor_r=tangent_warmup_factor
            )

            # sCD training objective (Algorithm 1):
            # L(θ,φ) = (exp(w_φ(t))/D)||F_θ(x_t/σ_d,t) - F_θ⁻(x_t/σ_d,t) - g||² - w_φ(t)

            diff_term = F_theta_current - F_theta_stopped - g_normalized
            loss_sq_per_item = (diff_term ** 2).sum(dim=[1,2,3])

            # Adaptive weighting with prior weighting w(t) = 1/(σ_d * tan(t))
            w_phi_t_raw = adaptive_weighting_net(t_sampled.float())
            w_phi_t = w_phi_t_raw.squeeze(-1)
            w_phi_t = torch.clamp(w_phi_t, -10, 10)  # Clamp for numerical stability

            # Prior weighting from the paper: w(t) = 1/(σ_d * tan(t))
            prior_weight = 1.0 / (sigma_d_val * torch.tan(t_sampled) + 1e-8)
            w_phi_t = w_phi_t + torch.log(prior_weight + 1e-8)  # Add in log space

            exp_w_phi_t = torch.exp(w_phi_t)

            # Final sCD loss with adaptive weighting (Equation 8 from paper)
            loss_terms_per_item = (exp_w_phi_t / D) * loss_sq_per_item - w_phi_t
            loss = loss_terms_per_item.mean()

            # Consistency regularization (optional)
            loss_consistency = ((f - x0) ** 2).mean()

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.1)

            optimizer.step()

            # Update EMA model
            scm.update_ema()

            # Logging
            wandb.log({
                "consistency_loss": loss_consistency.item(),
                "main_loss_term": loss_sq_per_item.mean().item(),
                "adaptive_loss": loss_terms_per_item.mean().item(),
                "total_loss": loss.item(),
                "w_phi_t_mean": w_phi_t.detach().mean().item(),
                "exp_w_phi_t_mean": exp_w_phi_t.detach().mean().item(),
                "g_tangent_norm_mean": g_normalized.view(b, -1).norm(dim=1).detach().mean().item(),
                "tangent_warmup_factor": tangent_warmup_factor,
                "prior_weight_mean": prior_weight.detach().mean().item()
            })

            # Loss binning by time for analysis
            with torch.no_grad():
                for tval, lval in zip(t_sampled, (f.detach() - x0).view(b, -1).pow(2).mean(1).cpu().tolist()):
                    idx = int(tval / (math.pi/2) * 10)
                    if 0 <= idx < 10:
                        lossbin[idx] += lval
                        losscnt[idx] += 1

        # Log per-t bin analysis
        for i_bin in range(10):
            diag_loss = lossbin[i_bin] / losscnt[i_bin]
            print(f"Epoch: {epoch}, t-bin {i_bin} (consistency_err_diagnostic): {diag_loss}")
            wandb.log({f"consistency_err_bin_{i_bin}": diag_loss})

        # Sampling demo using EMA model
        scm.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            z = torch.randn(16, channels, 32, 32).cuda()
            # Use EMA model for sampling
            x_gen = scm.sample(z, cond, steps=2, use_ema=True)
            # Unnormalize
            x_gen = x_gen * 0.5 + 0.5
            x_gen = x_gen.clamp(0, 1)
            grid = make_grid(x_gen.float(), nrow=4)
            img = grid.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")
            Image.fromarray(img).save(f"contents/scd_sample_{epoch}.png")

            # Log to wandb
            wandb.log({"samples": wandb.Image(f"contents/scd_sample_{epoch}.png")})

        scm.model.train()

        # Save checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'adaptive_weighting_state_dict': adaptive_weighting_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"contents/scd_checkpoint_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
