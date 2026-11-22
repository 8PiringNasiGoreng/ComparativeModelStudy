import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from src import Config as config
from src.data import get_data
from src.loading import img_shape, load_ae, load_classifier, load_gan
from src.utils import set_seed

DEV = config.DEVICE
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def pick_images(loader, ca, cb):
    found = {}
    for x, y in loader:
        for xi, yi in zip(x, y):
            yi = int(yi)
            if yi in (ca, cb) and yi not in found:
                found[yi] = xi
        if len(found) == len({ca, cb}):
            break
    return found[ca].unsqueeze(0).to(DEV), found[cb].unsqueeze(0).to(DEV)


def interp(zA, zB):
    return torch.cat([(1 - a) * zA + a * zB for a in ALPHAS])          # (6, D)


@torch.no_grad()
def find_latent(G, clf, cls, nz, n=2000):
    """Pilih z yang G(z)-nya paling yakin diklasifikasikan sebagai `cls`."""
    z = torch.randn(n, nz, device=DEV)
    imgs = ((G(z) + 1) / 2).clamp(0, 1)
    p = F.softmax(clf(imgs), dim=1)[:, cls]
    return z[p.argmax()].unsqueeze(0), p.max().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_a", type=int, default=1)
    ap.add_argument("--class_b", type=int, default=9)
    args = ap.parse_args()

    shape = img_shape()
    set_seed()
    _, _, test_loader = get_data()
    xA, xB = pick_images(test_loader, args.class_a, args.class_b)

    rows = []
    for name, label in [("vanilla", "Vanilla AE"), ("sparse", "Sparse AE"), ("vae", "VAE")]:
        m = load_ae(name)
        zA, zB = (m.encode(xA)[0], m.encode(xB)[0]) if name == "vae" else (m.encode(xA), m.encode(xB))
        rows.append((label, m.decode(interp(zA, zB)).view(-1, *shape)))

    try:
        clf = load_classifier()
    except FileNotFoundError:
        clf = None
        print("mlp_classifier.pt tidak ada -> z_A, z_B GAN dipilih acak.")
    for name, label in [("gan", "Vanilla GAN"), ("dcgan", "DCGAN")]:
        set_seed()
        G = load_gan(name)
        nz = config.GAN_NOISE_DIM
        if clf is not None:
            zA, pA = find_latent(G, clf, args.class_a, nz)
            zB, pB = find_latent(G, clf, args.class_b, nz)
            print(f"{label}: kepercayaan classifier A={pA:.2f}, B={pB:.2f}")
        else:
            zA, zB = torch.randn(1, nz, device=DEV), torch.randn(1, nz, device=DEV)
        rows.append((label, ((G(interp(zA, zB)) + 1) / 2).clamp(0, 1)))

    # statistik kontinuitas: MSE antar frame berurutan (loncatan besar = transisi tidak mulus)
    stats = {lab: ((r[1:] - r[:-1]) ** 2).flatten(1).mean(1).tolist() for lab, r in rows}
    with open(config.RES_DIR / "fig4_interp_stats.json", "w") as f:
        json.dump({"alphas": ALPHAS, "consecutive_frame_mse": stats}, f, indent=2)

    pad, size = 2, shape[1]
    grid = make_grid(torch.cat([r[1] for r in rows]).cpu(), nrow=len(ALPHAS), padding=pad, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(len(ALPHAS) * 1.5, len(rows) * 1.5 + 0.6))
    ax.imshow(grid.permute(1, 2, 0) if grid.shape[0] == 3 else grid[0], cmap="gray", vmin=0, vmax=1)
    ax.set_yticks([pad + i * (size + pad) + size / 2 for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xticks([pad + i * (size + pad) + size / 2 for i in range(len(ALPHAS))])
    ax.set_xticklabels([f"$\\alpha$={a}" for a in ALPHAS], fontsize=9)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Figur 4: Interpolasi linear (kelas {args.class_a} -> {args.class_b})", y=1.1)
    plt.tight_layout()
    out = config.FIG_DIR / "fig4_interpolation.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()