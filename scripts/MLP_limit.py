import json
import copy

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from torchvision.utils import make_grid

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT))

from src import Config
from src.data import get_data, dataset_info
from src.metrics import evaluate_reconstruction
from src.models.autoencoder import VanillaAE
from src.models.mlp import MLPClassifier, FrozenLatentDecoder
from src.utils import count_params, plot_losses, set_seed

DEV = Config.DEVICE
EPOCHS = 30
LR = 1e-3

@torch.no_grad()
def accuracy(model, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total

def train_classifier(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state = 0.0, None
    hist = {"train_ce": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, y, in train_loader:
            x, y = x.to(DEV), y.to(DEV)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        val_acc = accuracy(model, val_loader)
        hist['train_ce'].append(tot/n)
        hist['val_acc'].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[CLF] ep {ep:02d} | CE {tot / n:.4f} | val acc {val_acc:.4f}")
    model.load_state_dict(best_state)
    return hist


def fit_reconstructor(module, forward_fn, train_loader, val_loader, epochs=EPOCHS, lr=LR, tag=""):
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    hist = {"train_mse": [], "val_mse": []}
    for ep in range(1, epochs + 1):
        module.train()
        tot = n = 0
        for x, _ in train_loader:
            x = x.to(DEV)
            loss = F.mse_loss(forward_fn(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        module.eval()
        vtot = vn = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEV)
                vtot += F.mse_loss(forward_fn(x), x).item() * x.size(0)
                vn += x.size(0)
        hist["train_mse"].append(tot / n)
        hist["val_mse"].append(vtot / vn)
        print(f"[{tag}] ep {ep:02d} | train MSE {tot / n:.5f} | val MSE {vtot / vn:.5f}")
    module.eval()
    return hist


@torch.no_grad()
def one_per_class(loader):
    found = {}
    for x, y in loader:
        for xi, yi in zip(x, y):
            found.setdefault(int(yi), xi)
        if len(found) == 10:
            break
    labels = sorted(found)
    return torch.stack([found[k] for k in labels]).to(DEV), torch.tensor(labels, device=DEV)


@torch.no_grad()
def collect_latents(clf, loader):
    zs = []
    for x, _ in loader:
        zs.append(clf.features(x.to(DEV)))
    return torch.cat(zs)


def save_figure3(path, gt, from_y, from_z, frozen, ae, samples):
    def grid(t, nrow=10):
        g = make_grid(t.detach().cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=1.0)
        return g[0]

    panels = [
        ("(1) Citra asli (Ground Truth, satu per kelas)", gt),
        (r"(2) Proyeksi balik $W^T$  |  baris atas: dari $y_c$,  baris bawah: dari $z_{MLP}$",
         torch.cat([from_y, from_z])),
        (r"(3) Rekonstruksi  |  atas: Frozen Decoder ($z_{MLP}$),  bawah: Vanilla AE ($z_{AE}$)",
         torch.cat([frozen, ae])),
        (r"(4) Random prior sampling $z\sim\mathcal{N}(\mu_z,\mathrm{diag}(\sigma_z^2))$ -> Frozen Decoder",
         samples),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 2, 2, 2]})
    for ax, (title, t) in zip(axes, panels):
        ax.imshow(grid(t), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    channels, size = dataset_info()
    if channels != 1:
        raise SystemExit("Skrip ini mengasumsikan dataset 1 kanal (Fashion-MNIST/MNIST/KMNIST).")
    in_dim = size * size

    set_seed()
    train_loader, val_loader, test_loader = get_data()
    results = {}


    print("\n=== TAHAP 1: Latih MLP Classifier ===")
    clf = MLPClassifier(in_dim=in_dim, latent=Config.LATENT_DIM).to(DEV)
    clf_hist = train_classifier(clf, train_loader, val_loader)
    test_acc = accuracy(clf, test_loader)
    results["classifier_test_acc"] = test_acc

    print(f"Test accuracy MLP Classifier: {test_acc:.4f}")
    if test_acc < 0.88:
        print("PERINGATAN: akurasi < 88%. Tambah EPOCHS atau ubah learning rate.")

    torch.save(clf.state_dict(), Config.CKPT_DIR / "mlp_classifier.pt")
    clf.eval()

    for p in clf.parameters():
        p.requires_grad = False  

    gt, labels = one_per_class(test_loader)           # (10,1,H,W)
    y_onehot = F.one_hot(labels, 10).float()


    print("\n=== TAHAP 2: Proyeksi balik W^T dari vektor kelas ===")
    from_y = clf.backproject_class(y_onehot).view(-1, 1, size, size)


    print("\n=== TAHAP 3: Proyeksi balik W^T dari z_MLP ===")
    from_z = clf.backproject_latent(clf.features(gt)).view(-1, 1, size, size)
    results["transpose_from_latent"] = evaluate_reconstruction(
        lambda x: clf.backproject_latent(clf.features(x)).view_as(x), test_loader)
    print("W^T dari z_MLP (test):", results["transpose_from_latent"])


    print("\n=== TAHAP 4: Frozen Latent Decoder vs Vanilla AE ===")
    set_seed()
    decoder = FrozenLatentDecoder(latent=Config.LATENT_DIM, out_dim=in_dim).to(DEV)

    def frozen_forward(x):
        with torch.no_grad():
            z = clf.features(x)
        return decoder(z).view_as(x)

    dec_hist = fit_reconstructor(decoder, frozen_forward, train_loader, val_loader, tag="FrozenDec")
    results["frozen_decoder"] = evaluate_reconstruction(frozen_forward, test_loader)

    set_seed()
    ae = VanillaAE(in_dim=in_dim, latent=Config.LATENT_DIM).to(DEV)
    ae_hist = fit_reconstructor(ae, ae, train_loader, val_loader, tag="VanillaAE")
    results["vanilla_ae"] = evaluate_reconstruction(ae, test_loader)
    torch.save(decoder.state_dict(), Config.CKPT_DIR / "frozen_decoder.pt")
    torch.save(ae.state_dict(), Config.CKPT_DIR / "vanilla_ae_mlp_exp.pt")
    results["params"] = {"frozen_decoder": count_params(decoder), "vanilla_ae": count_params(ae)}

    print(f"\n{'Model':<22}{'MSE':>10}{'PSNR(dB)':>11}{'SSIM':>9}")
    for name, key in [("Frozen Dec (z_MLP)", "frozen_decoder"), ("Vanilla AE (z_AE)", "vanilla_ae"),
                      ("W^T dari z_MLP", "transpose_from_latent")]:
        r = results[key]
        print(f"{name:<22}{r['mse']:>10.5f}{r['psnr']:>11.2f}{r['ssim']:>9.4f}")

    plot_losses({"Frozen Decoder (val MSE)": dec_hist["val_mse"], "Vanilla AE (val MSE)": ae_hist["val_mse"]},
                Config.FIG_DIR / "mlp_decoder_vs_ae_loss.png", "Frozen z_MLP decoder vs Vanilla AE")


    print("\n=== TAHAP 5: Prior sampling pada ruang laten MLP ===")
    set_seed()
    z_test = collect_latents(clf, test_loader)
    mu_z, sigma_z = z_test.mean(0), z_test.std(0)
    z_rand = mu_z + sigma_z * torch.randn(20, z_test.size(1), device=DEV)
    with torch.no_grad():
        samples = decoder(z_rand).view(-1, 1, size, size)
        
    results["prior_sampling"] = {
        "frac_negative_dims_in_z_rand": (z_rand < 0).float().mean().item(),
        "frac_zero_dims_in_real_z": (z_test == 0).float().mean().item(),
    }
    print(results["prior_sampling"])


    with torch.no_grad():
        frozen_rec = decoder(clf.features(gt)).view(-1, 1, size, size)
        ae_rec = ae(gt)
    save_figure3(Config.FIG_DIR / "fig3_mlp_limits.png", gt, from_y, from_z, frozen_rec, ae_rec, samples)

    with open(Config.RES_DIR / "mlp_limits.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFigur 3  -> {Config.FIG_DIR / 'fig3_mlp_limits.png'}")
    print(f"Metrik   -> {Config.RES_DIR / 'mlp_limits.json'}")


if __name__ == "__main__":
    main()