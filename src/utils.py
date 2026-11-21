"""Fungsi bantu: seed, pengukuran latensi, penyimpanan grid gambar, plot loss."""
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src import Config


def set_seed(seed: int = Config.SEED) -> None:
    """Kunci semua sumber randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_latency(fn, n_warmup: int = 10, n_runs: int = 50) -> float:
    """Rata-rata latensi (ms) satu pemanggilan `fn()`.

    `fn` harus berupa fungsi tanpa argumen yang menghasilkan satu batch
    (64 sampel), misalnya: lambda: model.sample(64).
    Ada cuda.synchronize() agar pengukuran GPU akurat.
    """
    for _ in range(n_warmup):
        fn()
    if Config.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if Config.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000.0


def save_image_grid(imgs: torch.Tensor, path, nrow: int = 8, title: str | None = None) -> None:
    """Simpan grid gambar. `imgs` berada di rentang [0, 1], shape (N, C, H, W)."""
    imgs = imgs.detach().cpu().clamp(0, 1)
    grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=1.0)
    plt.figure(figsize=(nrow * 1.2, max(1, len(imgs) // nrow) * 1.2 + 0.5))
    if grid.shape[0] == 1:
        plt.imshow(grid[0], cmap="gray")
    else:
        plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_losses(history: dict, path, title: str = "Training Loss") -> None:
    """`history`: {'nama_kurva': [loss_per_epoch, ...]}."""
    plt.figure(figsize=(6, 4))
    for name, values in history.items():
        plt.plot(range(1, len(values) + 1), values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()