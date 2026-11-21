"""Figur 5: kurva training loss keenam model + trajektori reverse diffusion DDPM.
Juga membuat fig5b_gan_diagnostics.png (D(x), D(G(z)), norma gradien G) untuk analisis Q3.

Prasyarat: JSON hasil training di outputs/results/ dan outputs/figures/ddpm_trajectory.png.
Jalankan: python -m scripts.fig5_losses
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src import config


def jload(name):
    p = config.RES_DIR / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def ae_panel(ax, d, title):
    h = d["history"]
    ep = range(1, len(h["train_loss"]) + 1)
    l1, = ax.plot(ep, h["train_loss"], color="tab:blue", label="train loss (objektif)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss", color="tab:blue")
    ax2 = ax.twinx()
    l2, = ax2.plot(ep, h["val_mse"], color="tab:orange", label="val MSE")
    ax2.set_ylabel("Val MSE", color="tab:orange")
    ax.set_title(title, fontsize=10)
    ax.legend(handles=[l1, l2], fontsize=8)
    ax.grid(alpha=0.3)


def gan_panel(ax, d, title):
    h = d["history"]
    ep = range(1, len(h["loss_D"]) + 1)
    ax.plot(ep, h["loss_D"], label="Discriminator")
    ax.plot(ep, h["loss_G"], label="Generator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

    for i, (key, title) in enumerate([("vanilla", "Vanilla AE (MSE)"),
                                      ("sparse", "Sparse AE (MSE + $\\lambda\\|h\\|_1$)"),
                                      ("vae", "VAE (BCE + KL)")]):
        ax = fig.add_subplot(gs[0, i])
        d = jload(key)
        ae_panel(ax, d, title) if d else ax.text(0.5, 0.5, f"{key}.json tidak ada", ha="center")

    for i, (key, title) in enumerate([("gan", "Vanilla GAN"), ("dcgan", "DCGAN")]):
        ax = fig.add_subplot(gs[1, i])
        d = jload(key)
        gan_panel(ax, d, title) if d else ax.text(0.5, 0.5, f"{key}.json tidak ada", ha="center")

    ax = fig.add_subplot(gs[1, 2])
    d = jload("ddpm")
    if d:
        h = d["history"]
        ep = range(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], label="train")
        ax.plot(ep, h["val_loss"], label="val (EMA)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Noise-prediction MSE")
        ax.set_title("DDPM ($\\|\\epsilon-\\epsilon_\\theta\\|^2$)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    traj = config.FIG_DIR / "ddpm_trajectory.png"
    if traj.exists():
        ax.imshow(mpimg.imread(traj))
    else:
        ax.text(0.5, 0.5, "ddpm_trajectory.png tidak ada", ha="center")
    ax.axis("off")

    fig.suptitle("Figur 5: Training loss & DDPM denoising trajectory", fontsize=14)
    plt.tight_layout()
    out = config.FIG_DIR / "fig5_losses.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Tersimpan:", out)

    # ---------- diagnostik GAN (pendukung Q3) ----------
    runs = [(k, jload(k)) for k in ("gan", "gan_minimax", "dcgan")]
    runs = [(k, d) for k, d in runs if d]
    if runs:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for k, d in runs:
            h = d["history"]
            ep = range(1, len(h["D_real"]) + 1)
            axes[0].plot(ep, h["D_real"], label=k)
            axes[1].plot(ep, h["D_fake"], label=k)
            axes[2].plot(ep, h["grad_norm_G"], label=k)
        axes[0].set_title("D(x) pada data riil")
        axes[1].set_title("D(G(z)) pada data palsu")
        axes[2].set_title("Norma gradien Generator")
        axes[2].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        out = config.FIG_DIR / "fig5b_gan_diagnostics.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print("Tersimpan:", out)


if __name__ == "__main__":
    main()