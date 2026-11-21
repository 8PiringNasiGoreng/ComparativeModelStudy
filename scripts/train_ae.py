"""Latih & evaluasi Vanilla AE / Sparse AE / VAE.

Contoh (dari root repo):
    python -m scripts.train_ae --model vanilla
    python -m scripts.train_ae --model sparse --lam 1e-2
    python -m scripts.train_ae --model vae

Loss:
    vanilla : MSE
    sparse  : MSE + lambda * mean(|h|)    (h = aktivasi hidden layer 256-dim)
    vae     : BCE (sum piksel) + KL       (rata-rata per sampel)
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from src import config
from src.data import dataset_info, get_dataloaders
from src.metrics import compute_fid, evaluate_reconstruction
from src.models.autoencoder import build_ae
from src.utils import count_params, measure_latency, plot_losses, save_image_grid, set_seed

DEV = config.DEVICE


def vae_loss(x_hat, x, mu, logvar):
    b = x.size(0)
    bce = F.binary_cross_entropy(x_hat.flatten(1), x.flatten(1), reduction="sum") / b
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return bce + kl, bce, kl


def compute_loss(name, model, x, lam):
    """Return (loss, dict komponen)."""
    if name == "vanilla":
        recon = F.mse_loss(model(x), x)
        return recon, {"recon": recon.item()}
    if name == "sparse":
        x_hat, h = model.forward_with_h(x)
        recon = F.mse_loss(x_hat, x)
        l1 = h.abs().mean()
        return recon + lam * l1, {"recon": recon.item(), "l1": l1.item()}
    x_hat, mu, logvar = model(x)
    loss, bce, kl = vae_loss(x_hat, x, mu, logvar)
    return loss, {"bce": bce.item(), "kl": kl.item()}


def get_recon_fn(name, model):
    """Fungsi rekonstruksi deterministik x -> x_hat (untuk metrik)."""
    return model.reconstruct if name == "vae" else model


@torch.no_grad()
def mean_mse(recon_fn, loader):
    tot = n = 0
    for x, _ in loader:
        x = x.to(DEV)
        tot += F.mse_loss(recon_fn(x), x, reduction="sum").item()
        n += x.numel()
    return tot / n


@torch.no_grad()
def hidden_stats(model, loader):
    """Statistik aktivasi hidden layer 256-dim: fraksi unit = 0 dan rata-rata aktivasi."""
    zero = total = 0
    act_sum = 0.0
    for x, _ in loader:
        h = model.hidden(x.to(DEV))
        zero += (h == 0).sum().item()
        total += h.numel()
        act_sum += h.abs().sum().item()
    return {"frac_zero": zero / total, "mean_activation": act_sum / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vanilla", "sparse", "vae"], required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=config.LAMBDA_SPARSE, help="lambda L1 (Sparse AE)")
    ap.add_argument("--tag", type=str, default="", help="akhiran nama file (mis. _lam1e-3)")
    ap.add_argument("--skip_fid", action="store_true")
    args = ap.parse_args()

    name = args.model
    run = f"{name}{args.tag}"
    channels, size = dataset_info()
    img_shape = (channels, size, size)
    in_dim = channels * size * size

    set_seed()
    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_ae(name, in_dim=in_dim, latent=config.LATENT_DIM, img_shape=img_shape).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    recon_fn = get_recon_fn(name, model)
    print(f"Model: {run} | parameter: {count_params(model):,} | device: {DEV}")

    hist = {"train_loss": [], "val_mse": []}
    comp_hist = {}
    best_val, best_ep, epoch_times = float("inf"), 0, []
    ckpt = config.CKPT_DIR / f"{run}.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot = n = 0
        comp_sum = {}
        for x, _ in train_loader:
            x = x.to(DEV)
            loss, comps = compute_loss(name, model, x, args.lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            tot += loss.item() * bs
            n += bs
            for k, v in comps.items():
                comp_sum[k] = comp_sum.get(k, 0.0) + v * bs
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        model.eval()
        val_mse = mean_mse(recon_fn, val_loader)
        hist["train_loss"].append(tot / n)
        hist["val_mse"].append(val_mse)
        for k, v in comp_sum.items():
            comp_hist.setdefault(k, []).append(v / n)
        if val_mse < best_val:
            best_val, best_ep = val_mse, ep
            torch.save(model.state_dict(), ckpt)
        extra = " | ".join(f"{k} {v[-1]:.4f}" for k, v in comp_hist.items())
        print(f"ep {ep:03d} | loss {tot / n:.5f} | {extra} | val MSE {val_mse:.5f} | {epoch_times[-1]:.1f}s")

    # ---------------- evaluasi pada TEST set (bobot terbaik) ----------------
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    model.eval()
    results = {
        "model": run, "latent_dim": config.LATENT_DIM, "params": count_params(model),
        "epochs": args.epochs, "best_epoch": best_ep,
        "avg_epoch_time_s": sum(epoch_times) / len(epoch_times),
        "test": evaluate_reconstruction(recon_fn, test_loader),
        "hidden_layer_stats": hidden_stats(model, test_loader),
    }
    if name == "sparse":
        results["lambda"] = args.lam

    x64 = next(iter(test_loader))[0][:64].to(DEV)
    results["latency_ms_recon_batch64"] = measure_latency(lambda: recon_fn(x64))

    if name == "vae":
        results["latency_ms_sample_batch64"] = measure_latency(lambda: model.sample(64))
        set_seed()
        save_image_grid(model.sample(16), config.FIG_DIR / f"{run}_samples.png", nrow=4,
                        title="VAE samples (z ~ N(0, I))")
        if not args.skip_fid:
            try:
                results["fid"] = compute_fid(test_loader, model.sample)
            except Exception as e:  # unduhan bobot Inception bisa gagal tanpa internet
                print("FID gagal dihitung:", e)
                results["fid"] = None

    results["history"] = {**hist, "components": comp_hist}
    with open(config.RES_DIR / f"{run}.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_losses({"train loss": hist["train_loss"]}, config.FIG_DIR / f"{run}_loss.png", f"{run}: training loss")

    print("\n=== HASIL TEST SET ===")
    print({k: v for k, v in results.items() if k != "history"})


if __name__ == "__main__":
    main()