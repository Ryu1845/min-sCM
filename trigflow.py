# TrigFlow: A Simple Framework Unifying EDM, Flow Matching and Velocity Prediction
# Based on the paper "TrigFlow: A Simple Framework Unifying EDM, Flow Matching and Velocity Prediction"
# Implementation following the mathematical formulations from the appendix and stabilizing techniques

import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from dit import DiT_Llama


class TrigFlow:
    """
    TrigFlow: Unified framework for diffusion models using trigonometric interpolation.
    
    Key properties:
    - Diffusion process: x_t = cos(t)*x_0 + sin(t)*z for t ∈ [0, π/2]
    - PF-ODE: dx_t/dt = σ_d * F_θ(x_t/σ_d, c_noise(t))
    - Parameterization: D_θ(x_t,t) = cos(t)*x_t - sin(t)*σ_d*F_θ(x_t/σ_d, c_noise(t))
    - Training target: v_t = cos(t)*z - sin(t)*x_0
    """

    def __init__(self, model, sigma_d=1.0, c_noise_fn=None):
        self.model = model
        self.sigma_d = sigma_d
        # Default to identity time transformation for stability (c_noise(t) = t)
        self.c_noise_fn = c_noise_fn if c_noise_fn is not None else lambda t: t

    def forward_process(self, x0, t):
        """
        TrigFlow forward process: x_t = cos(t)*x_0 + sin(t)*z
        Returns: x_t, z, v_t (velocity target)
        """
        z = torch.randn_like(x0) * self.sigma_d
        cos_t = torch.cos(t).view(-1, *([1] * len(x0.shape[1:])))
        sin_t = torch.sin(t).view(-1, *([1] * len(x0.shape[1:])))

        # TrigFlow interpolation
        x_t = cos_t * x0 + sin_t * z

        # Velocity target: v_t = cos(t)*z - sin(t)*x_0
        v_t = cos_t * z - sin_t * x0

        return x_t, z, v_t

    def model_prediction(self, x_t, t, cond):
        """
        Get F_θ prediction from the model.
        """
        x_scaled = x_t / self.sigma_d
        c_noise_t = self.c_noise_fn(t)
        return self.model(x_scaled, c_noise_t, cond)

    def diffusion_model(self, x_t, t, cond):
        """
        TrigFlow diffusion model parameterization:
        D_θ(x_t,t) = cos(t)*x_t - sin(t)*σ_d*F_θ(x_t/σ_d, c_noise(t))
        """
        cos_t = torch.cos(t).view(-1, *([1] * len(x_t.shape[1:])))
        sin_t = torch.sin(t).view(-1, *([1] * len(x_t.shape[1:])))

        F_theta = self.model_prediction(x_t, t, cond)

        # TrigFlow parameterization
        D_theta = cos_t * x_t - sin_t * self.sigma_d * F_theta

        return D_theta

    def velocity_prediction(self, x_t, t, cond):
        """
        Get velocity prediction: σ_d * F_θ(x_t/σ_d, c_noise(t))
        This is the PF-ODE: dx_t/dt = σ_d * F_θ(x_t/σ_d, c_noise(t))
        """
        F_theta = self.model_prediction(x_t, t, cond)
        return self.sigma_d * F_theta

    def training_loss(self, x0, cond, t=None, use_log_normal=True):
        """
        TrigFlow training objective:
        L = E[||σ_d*F_θ(x_t/σ_d, c_noise(t)) - v_t||²]
        where v_t = cos(t)*z - sin(t)*x_0
        """
        b = x0.size(0)

        if t is None:
            if use_log_normal:
                # Default: Log-normal time sampling as recommended in the paper
                t = log_normal_time_sampling(b, device=x0.device)
            else:
                # Alternative: Uniform sampling in [0, π/2]
                t = torch.rand(b, device=x0.device) * (math.pi / 2)

        # Forward process
        x_t, z, v_t = self.forward_process(x0, t)

        # Model prediction
        v_pred = self.velocity_prediction(x_t, t, cond)

        # MSE loss
        loss = ((v_pred - v_t) ** 2).mean()

        return loss, {"x_t": x_t, "v_t": v_t, "v_pred": v_pred, "t": t}

    @torch.no_grad()
    def sample_ddim(self, z, cond, steps=50, return_trajectory=False):
        """
        1st-order sampler using DDIM formulation in TrigFlow.
        Starting from z at t=π/2, sample to t=0.
        """
        device = z.device
        b = z.size(0)

        # Time schedule from π/2 to 0
        time_schedule = torch.linspace(math.pi/2, 0, steps + 1, device=device)

        x = z.clone()
        trajectory = [x.clone()] if return_trajectory else None

        for i in range(steps):
            t_curr = time_schedule[i]
            t_next = time_schedule[i + 1]

            # Current time for all batch items
            t_batch = torch.full((b,), t_curr, device=device)

            # Get velocity prediction
            v_pred = self.velocity_prediction(x, t_batch, cond)

            # DDIM step: x_{t_next} = x_t - (t_curr - t_next) * v_pred
            dt = t_curr - t_next
            x = x - dt * v_pred

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return trajectory
        return x

    @torch.no_grad()
    def sample_euler(self, z, cond, steps=50, return_trajectory=False):
        """
        Euler method for solving the PF-ODE.
        dx_t/dt = σ_d * F_θ(x_t/σ_d, c_noise(t))
        """
        device = z.device
        b = z.size(0)

        dt = -math.pi / (2 * steps)  # Negative because we go from π/2 to 0

        x = z.clone()
        t = math.pi / 2
        trajectory = [x.clone()] if return_trajectory else None

        for _ in range(steps):
            t_batch = torch.full((b,), t, device=device)

            # Get velocity prediction
            v_pred = self.velocity_prediction(x, t_batch, cond)

            # Euler step
            x = x + dt * v_pred
            t = t + dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return trajectory
        return x

    @torch.no_grad()
    def sample(self, z, cond, steps=50, method="ddim", return_trajectory=False):
        """
        Sample from the TrigFlow model.
        """
        if method == "ddim":
            return self.sample_ddim(z, cond, steps, return_trajectory)
        elif method == "euler":
            return self.sample_euler(z, cond, steps, return_trajectory)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


def log_normal_time_sampling(batch_size, P_mean=-1.0, P_std=1.4, sigma_d=1.0, device="cuda"):
    """
    Log-normal time sampling as described in the paper:
    τ ~ N(P_mean, P_std²), t = arctan(e^τ/σ_d)
    """
    tau = torch.randn(batch_size, device=device) * P_std + P_mean
    t = torch.atan(torch.exp(tau) / sigma_d)

    # Conservative clamping to avoid numerical issues
    t = torch.clamp(t, 0.01, math.pi/2 - 0.01)

    return t


def load_trigflow_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a TrigFlow checkpoint and create the model and TrigFlow wrapper.
    
    Returns:
        model: The DiT_Llama model with loaded weights
        trigflow: TrigFlow wrapper around the model
        checkpoint: The full checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    config = checkpoint['trigflow_config']
    model_config = config['model_config']
    channels = config['channels']

        # Create model with the same architecture
    model = DiT_Llama(
        channels, 32,  # channels and image_size as positional args
        dim=model_config['dim'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        num_classes=model_config['num_classes'],
        class_dropout_prob=model_config['class_dropout_prob']
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create TrigFlow wrapper
    trigflow = TrigFlow(model, sigma_d=config['sigma_d'])

    print(f"Loaded TrigFlow checkpoint from {checkpoint_path}")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    return model, trigflow, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train TrigFlow on MNIST or CIFAR-10")
    parser.add_argument("--cifar", action="store_true", help="Use CIFAR-10 instead of MNIST")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--method", type=str, default="ddim", choices=["ddim", "euler"], help="Sampling method")
    parser.add_argument("--uniform_time", action="store_true", help="Use uniform time sampling instead of log-normal (default)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
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
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, class_dropout_prob=0.20
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
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, class_dropout_prob=0.20
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6:.2f}M")

        # Initialize TrigFlow
    sigma_d = 1.0
    trigflow = TrigFlow(model, sigma_d=sigma_d)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Dataset
    batch_size = 128
    dataset = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize wandb
    wandb.init(project=f"trigflow_{dataset_name}", config={
        "dataset": dataset_name,
        "sigma_d": sigma_d,
        "batch_size": batch_size,
        "sampling_steps": args.steps,
        "sampling_method": args.method,
        "uniform_time": args.uniform_time,
    })

    # Create output directory
    os.makedirs("contents", exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, 100):
        model.train()

        # Loss binning for analysis
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}

        epoch_loss = 0
        num_batches = 0

        for i, (x0, y) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}"):
            x0, y = x0.cuda(), y.cuda()

            optimizer.zero_grad()

            # Compute loss (log-normal time sampling is default)
            loss, info = trigflow.training_loss(x0, y, use_log_normal=not args.uniform_time)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1

            wandb.log({
                "loss": loss.item(),
                "t_mean": info["t"].mean().item(),
                "t_std": info["t"].std().item(),
            })

            # Loss binning by time
            with torch.no_grad():
                t_vals = info["t"]
                loss_vals = ((info["v_pred"] - info["v_t"]) ** 2).mean(dim=list(range(1, len(info["v_pred"].shape)))).cpu().tolist()

                for t_val, l_val in zip(t_vals.cpu().tolist(), loss_vals):
                    # Bin index based on t ∈ [0, π/2]
                    idx = int((t_val / (math.pi / 2)) * 10)
                    if 0 <= idx < 10:
                        lossbin[idx] += l_val
                        losscnt[idx] += 1

        # Log epoch statistics
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.6f}")

        # Log loss bins
        for i in range(10):
            bin_loss = lossbin[i] / losscnt[i]
            print(f"  t-bin {i}: {bin_loss:.6f}")
            wandb.log({f"loss_bin_{i}": bin_loss})

        wandb.log({"epoch_loss": avg_epoch_loss})

        # Sampling and visualization
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Sample images with trajectory for GIF
                cond = torch.arange(0, 16).cuda() % 10
                z = torch.randn(16, channels, 32, 32).cuda() * sigma_d

                # Get sampling trajectory
                trajectory = trigflow.sample(z, cond, steps=args.steps, method=args.method, return_trajectory=True)

                # Create GIF from trajectory
                gif_frames = []
                for step_img in trajectory:
                    # Unnormalize
                    step_unnorm = step_img * 0.5 + 0.5
                    step_unnorm = step_unnorm.clamp(0, 1)
                    grid = make_grid(step_unnorm.float(), nrow=4)
                    img_array = grid.permute(1, 2, 0).cpu().numpy()
                    img_array = (img_array * 255).astype(np.uint8)
                    gif_frames.append(Image.fromarray(img_array))

                # Save GIF
                gif_frames[0].save(
                    f"contents/trigflow_sample_{epoch}.gif",
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=100,  # 100ms per frame
                    loop=0,
                )

                # Save final image
                final_img = gif_frames[-1]
                final_img.save(f"contents/trigflow_sample_{epoch}_final.png")

                # Log to wandb
                wandb.log({
                    "samples_gif": wandb.Image(f"contents/trigflow_sample_{epoch}.gif"),
                    "samples_final": wandb.Image(f"contents/trigflow_sample_{epoch}_final.png")
                })

        # Save checkpoints
        if epoch % 10 == 0 or epoch == 99:  # Save every 10 epochs and at the end
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trigflow_config': {
                    'sigma_d': sigma_d,
                    'dataset': dataset_name,
                    'channels': channels,
                    'model_config': {
                        'dim': 256 if CIFAR else 64,
                        'n_layers': 10 if CIFAR else 6,
                        'n_heads': 8 if CIFAR else 4,
                        'num_classes': 10,
                        'class_dropout_prob': 0.20
                    }
                },
                'loss': avg_epoch_loss,
                'args': vars(args)
            }

            checkpoint_path = f"contents/trigflow_{dataset_name}_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Also save a "latest" checkpoint that's easy to find
            if epoch == 99:  # Final checkpoint
                latest_path = f"contents/trigflow_{dataset_name}_final.pt"
                torch.save(checkpoint, latest_path)
                print(f"Saved final checkpoint: {latest_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
