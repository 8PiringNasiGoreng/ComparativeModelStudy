"""Latih & evaluasi Vanilla GAN / DCGAN.

Contoh (dari root repo):
    python -m scripts.train_gan --model gan
    python -m scripts.train_gan --model dcgan
    # eksperimen vanishing gradient (loss minimax asli, saturating):
    python -m scripts.train_gan --model gan --loss minimax --tag _minimax

Loss diskriminator : BCE(D(x), 1) + BCE(D(G(z)), 0)
Loss generator     : nonsat  -> BCE(D(G(z)), 1)          (-log D(G(z)), default)
                     minimax -> -BCE(D(G(z)), 0)          (log(1 - D(G(z))), saturating)
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT))

from src import config
from src.data import dataset_info, get_dataloaders, to_0_1, to_minus1_1
from src.metrics import compute_fid
from src.models.gan import build_gan
from src.utils import count_params, measure_latency, plot_losses, save_image_grid, set_seed

DEV = config.DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gan", "dcgan"], required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loss", choices=["nonsat", "minimax"], default="nonsat")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--snap_every", type=int, default=10, help="simpan grid sampel tiap N epoch")
    ap.add_argument("--skip_fid", action="store_true")
    args = ap.parse_args()

    run = f"{args.model}{args.tag}"
    channels, size = dataset_info()
    img_shape = (channels, size, size)
    nz = config.GAN_NOISE_DIM

    set_seed()
    train_loader, _, test_loader = get_dataloaders()
    G, D = build_gan(args.model, nz, img_shape)
    G, D = G.to(DEV), D.to(DEV)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(16, nz, device=DEV)
    print(f"Model: {run} | G: {count_params(G):,} | D: {count_params(D):,} | loss G: {args.loss} | device: {DEV}")

    hist = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": [], "grad_norm_G": []}
    epoch_times = []

    for ep in range(1, args.epochs + 1):
        G.train()
        D.train()
        t0 = time.perf_counter()
        acc = {k: 0.0 for k in hist}
        n = 0
        for x, _ in train_loader:
            x = to_minus1_1(x.to(DEV))
            b = x.size(0)
            ones, zeros = torch.ones(b, device=DEV), torch.zeros(b, device=DEV)

            # ---- (1) update Discriminator ----
            fake = G(torch.randn(b, nz, device=DEV))
            d_real, d_fake = D(x), D(fake.detach())
            loss_d = bce(d_real, ones) + bce(d_fake, zeros)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---- (2) update Generator ----
            d_fake_g = D(fake)
            loss_g = bce(d_fake_g, ones) if args.loss == "nonsat" else -bce(d_fake_g, zeros)
            opt_g.zero_grad()
            loss_g.backward()
            # norma gradien generator (bukti vanishing gradient bila mendekati 0)
            gnorm = nn.utils.clip_grad_norm_(G.parameters(), float("inf")).item()
            opt_g.step()

            acc["loss_D"] += loss_d.item() * b
            acc["loss_G"] += loss_g.item() * b
            acc["D_real"] += torch.sigmoid(d_real).mean().item() * b
            acc["D_fake"] += torch.sigmoid(d_fake).mean().item() * b
            acc["grad_norm_G"] += gnorm * b
            n += b

        if DEV.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        for k in hist:
            hist[k].append(acc[k] / n)
        print(f"ep {ep:03d} | D {hist['loss_D'][-1]:.3f} | G {hist['loss_G'][-1]:.3f} | "
              f"D(x) {hist['D_real'][-1]:.2f} | D(G(z)) {hist['D_fake'][-1]:.2f} | "
              f"|grad G| {hist['grad_norm_G'][-1]:.3f} | {epoch_times[-1]:.1f}s")

        if ep == 1 or ep % args.snap_every == 0 or ep == args.epochs:
            G.eval()
            with torch.no_grad():
                save_image_grid(to_0_1(G(fixed_z)), config.FIG_DIR / f"{run}_epoch{ep:03d}.png",
                                nrow=4, title=f"{run} epoch {ep}")

    # ---------------- evaluasi (epoch terakhir) ----------------
    G.eval()
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, config.CKPT_DIR / f"{run}.pt")

    results = {
        "model": run, "loss_G": args.loss, "noise_dim": nz, "epochs": args.epochs,
        "params_G": count_params(G), "params_D": count_params(D),
        "avg_epoch_time_s": sum(epoch_times) / len(epoch_times),
        "latency_ms_sample_batch64": measure_latency(lambda: G.sample(64)),
    }
    set_seed()
    save_image_grid(G.sample(16), config.FIG_DIR / f"{run}_samples.png", nrow=4,
                    title=f"{run} samples (z ~ N(0, I))")
    if not args.skip_fid:
        try:
            results["fid"] = compute_fid(test_loader, G.sample)
        except Exception as e:
            print("FID gagal dihitung:", e)
            results["fid"] = None
    results["history"] = hist

    with open(config.RES_DIR / f"{run}.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_losses({"Discriminator": hist["loss_D"], "Generator": hist["loss_G"]},
                config.FIG_DIR / f"{run}_loss.png", f"{run}: GAN loss")

    print("\n=== HASIL ===")
    print({k: v for k, v in results.items() if k != "history"})


if __name__ == "__main__":
    main()