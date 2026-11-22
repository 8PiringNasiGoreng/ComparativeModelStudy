"""Figur 1: grid rekonstruksi (Ground Truth, Vanilla AE, Sparse AE, VAE).

Prasyarat: sudah menjalankan scripts.train_ae untuk vanilla, sparse, dan vae.
Jalankan: python -m scripts.fig1_reconstruction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src import Config as config
from src.data import dataset_info, get_data
from src.models.autoencoder import build_ae
from src.utils import set_seed

N = 8


def main():
    set_seed()
    channels, size = dataset_info()
    in_dim = channels * size * size
    _, _, test_loader = get_data()
    x = next(iter(test_loader))[0][:N].to(config.DEVICE)   # test set tidak di-shuffle -> selalu sama

    rows = [("Ground Truth", x)]
    for name, label in [("vanilla", "Vanilla AE"), ("sparse", "Sparse AE"), ("vae", "VAE")]:
        model = build_ae(name, in_dim=in_dim, latent=config.LATENT_DIM,
                         img_shape=(channels, size, size)).to(config.DEVICE)
        model.load_state_dict(torch.load(config.CKPT_DIR / f"{name}.pt", map_location=config.DEVICE))
        model.eval()
        with torch.no_grad():
            x_hat = model.reconstruct(x) if name == "vae" else model(x)
        rows.append((label, x_hat))

    pad = 2
    grid = make_grid(torch.cat([r[1] for r in rows]).cpu().clamp(0, 1), nrow=N, padding=pad, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(N * 1.3, len(rows) * 1.4))
    ax.imshow(grid.permute(1, 2, 0) if grid.shape[0] == 3 else grid[0], cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([pad + i * (size + pad) + size / 2 for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Figur 1: Rekonstruksi Test Set", fontsize=12)
    plt.tight_layout()
    out = config.FIG_DIR / "fig1_reconstruction.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()