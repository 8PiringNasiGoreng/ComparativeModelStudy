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

    xs, total = [], 0
    for batch in test_loader:
        xs.append(batch[0])
        total += batch[0].shape[0]
        if total >= N:
            break
    x = torch.cat(xs)[:N].to(config.DEVICE)
    print("GT shape:", tuple(x.shape))   # harus (8, C, H, W)

    rows = [("Ground Truth", x)]
    for name, label in [("vanilla", "Vanilla AE"), ("sparse", "Sparse AE"), ("vae", "VAE")]:
        model = build_ae(name, in_dim=in_dim, latent=config.LATENT_DIM,
                         img_shape=(channels, size, size)).to(config.DEVICE)
        model.load_state_dict(torch.load(config.CKPT_DIR / f"{name}.pt", map_location=config.DEVICE))
        model.eval()
        with torch.no_grad():
            x_hat = model.reconstruct(x) if name == "vae" else model(x)
        print(f"{name} shape sebelum reshape:", tuple(x_hat.shape))
        x_hat = x_hat.reshape(x.shape)
        rows.append((label, x_hat))

    fig, axes = plt.subplots(len(rows), N, figsize=(N * 1.3, len(rows) * 1.4))
    for r, (label, imgs) in enumerate(rows):
        imgs = imgs.detach().cpu().clamp(0, 1)
        for c in range(N):
            ax = axes[r, c]
            img = imgs[c]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center")
    fig.suptitle("Figur 1: Rekonstruksi Test Set", fontsize=12)
    plt.tight_layout()

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIG_DIR / "fig1_reconstruction_v2.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Tersimpan:", out.resolve())


if __name__ == "__main__":
    main()