import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src import config
from src.loading import img_shape, load_ae, load_ddpm, load_gan
from src.utils import set_seed


def main():
    shape = img_shape()
    panels = []

    set_seed()
    panels.append(("Panel A: VAE\n$z\\sim\\mathcal{N}(0,I)$", load_ae("vae").sample(16)))
    set_seed()
    panels.append(("Panel B: Vanilla GAN\n$z\\sim\\mathcal{N}(0,I)$", load_gan("gan").sample(16)))
    set_seed()
    panels.append(("Panel C: DCGAN\n$z\\sim\\mathcal{N}(0,I)$", load_gan("dcgan").sample(16)))
    set_seed()
    unet, diffusion = load_ddpm()
    panels.append(("Panel D: DDPM\n$x_T\\sim\\mathcal{N}(0,I)$ (reverse denoising)",
                   diffusion.sample(unet, 16, shape)))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, (title, imgs) in zip(axes, panels):
        g = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=4, padding=2, pad_value=1.0)
        ax.imshow(g.permute(1, 2, 0) if g.shape[0] == 3 else g[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    out = config.FIG_DIR / "fig2_sampling.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()