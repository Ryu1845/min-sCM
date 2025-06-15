# Minimal implementation of simple, stable, and scalable consistency models

This is based on [cloneofsimo/minRF](https://github.com/cloneofsimo/minRF). 

I can't make it work but maybe you can.

This repo also implements flash attention jvp which you might want to use.

## Results

### TrigFlow

![trigflow on cifar10 final sample](./contents/trigflow_sample_95_final.png)
![trigflow on cifar10 gif](./contents/trigflow_sample_95.gif)

### sCD

sCD 2-step sampling

![sCD on cifar10 final sample](./contents/scd_sample_99.png)

## Usage

```bash
uv venv
uv pip install wandb torch torchvision
uv run python trigflow.py --cifar # --cifar to use cifar10, uses mnist by default
uv run python scm.py --teacher_path contents/trigflow_mnist_final.pt --cifar
```