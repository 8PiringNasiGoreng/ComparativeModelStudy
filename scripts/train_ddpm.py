import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src import Config as config
from src.data import dataset_info, get_dataloaders, to_minus1_1
from src.metrics import compute_fid
from src.models.ddpm import GaussianDiffusion, UNet
from src.utils import count_params, measure_latency, plot_losses, save_image_grid, set_seed

DEV = config.DEVICE


@torch.no_grad()
def ema_update(ema_model, model, step, max_decay=0.999):
    decay = min(max_decay, (1 + step) / (10 + step))      # warm-up EMA
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


@torch.no_grad()
def val_loss(diffusion, model, loader):
    """Loss prediksi derau pada validation set dengan derau/timestep tetap (generator ber-seed)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    model.eval()
    tot = n = 0
    for x, _ in loader:
        x = to_minus1_1(x.to(DEV))
        tot += diffusion.loss(model, x, generator=g).item() * x.size(0)
        n += x.size(0)
    return tot / n


def save_trajectory(traj, path, n_rows=4):
    """Figur 5 (panel difusi): reverse diffusion x_T -> x_0 untuk n_rows sampel."""
    labels = [str(t) for t, _ in traj]
    cols = len(traj)
    tiles = [traj[c][1][r] for r in range(n_rows) for c in range(cols)]
    size = tiles[0].shape[-1]
    pad = 2
    grid = make_grid(torch.stack(tiles), nrow=cols, padding=pad, pad_value=1.0)
    fig, ax = plt.subplots(figsize=(cols * 1.1, n_rows * 1.2 + 0.6))
    ax.imshow(grid.permute(1, 2, 0) if grid.shape[0] == 3 else grid[0], cmap="gray", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks([pad + c * (size + pad) + size / 2 for c in range(cols)])
    ax.set_xticklabels([f"t={l}" if l != "x0" else "x0" for l in labels], fontsize=8)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("DDPM reverse denoising: $x_T \\to x_0$", y=1.12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--base", type=int, default=64, help="lebar kanal dasar U-Net")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fid_samples", type=int, default=0, help="0 = lewati FID")
    args = ap.parse_args()

    run = f"ddpm{args.tag}"
    channels, size = dataset_info()
    img_shape = (channels, size, size)

    set_seed()
    train_loader, val_loader, test_loader = get_dataloaders()
    model = UNet(in_ch=channels, base=args.base).to(DEV)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    diffusion = GaussianDiffusion(args.timesteps).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"DDPM | U-Net: {count_params(model):,} parameter | T={args.timesteps} | device: {DEV}")

    last_path = config.CKPT_DIR / f"{run}_last.pt"
    hist = {"train_loss": [], "val_loss": []}
    epoch_times, step, start_ep = [], 0, 1
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=DEV)
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        hist, epoch_times, step, start_ep = ck["hist"], ck["epoch_times"], ck["step"], ck["epoch"] + 1
        print(f"Resume dari epoch {ck['epoch']}")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot = n = 0
        for x, _ in train_loader:
            x = to_minus1_1(x.to(DEV))
            loss = diffusion.loss(model, x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            ema_update(ema_model, model, step)
            tot += loss.item() * x.size(0)
            n += x.size(0)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(val_loss(diffusion, ema_model, val_loader))
        print(f"ep {ep:03d} | train eps-MSE {hist['train_loss'][-1]:.5f} | "
              f"val eps-MSE (EMA) {hist['val_loss'][-1]:.5f} | {epoch_times[-1]:.1f}s")
        torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(), "opt": opt.state_dict(),
                    "hist": hist, "epoch_times": epoch_times, "step": step, "epoch": ep}, last_path)

    # ---------------- evaluasi dengan bobot EMA ----------------
    torch.save(ema_model.state_dict(), config.CKPT_DIR / f"{run}.pt")
    ema_model.eval()

    results = {
        "model": run, "timesteps": args.timesteps, "epochs": args.epochs, "params": count_params(model),
        "avg_epoch_time_s": sum(epoch_times) / len(epoch_times),
    }
    x_t = torch.randn(64, *img_shape, device=DEV)
    t_mid = torch.full((64,), args.timesteps // 2, device=DEV, dtype=torch.long)
    results["latency_ms_single_step_batch64"] = measure_latency(lambda: ema_model(x_t, t_mid))
    # latensi sampling penuh T langkah (lambat -> sedikit pengulangan)
    results["latency_ms_sample_batch64"] = measure_latency(
        lambda: diffusion.sample(ema_model, 64, img_shape), n_warmup=1, n_runs=3)

    set_seed()
    imgs, traj = diffusion.sample(ema_model, 16, img_shape, record=10)
    save_image_grid(imgs, config.FIG_DIR / f"{run}_samples.png", nrow=4, title="DDPM samples")
    save_trajectory(traj, config.FIG_DIR / f"{run}_trajectory.png")

    if args.fid_samples > 0:
        try:
            results["fid"] = compute_fid(test_loader, lambda k: diffusion.sample(ema_model, k, img_shape),
                                         n_samples=args.fid_samples)
            results["fid_samples"] = args.fid_samples
        except Exception as e:
            print("FID gagal dihitung:", e)
            results["fid"] = None
    results["history"] = hist

    with open(config.RES_DIR / f"{run}.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_losses(hist, config.FIG_DIR / f"{run}_loss.png", "DDPM: noise-prediction loss")

    print("\n=== HASIL ===")
    print({k: v for k, v in results.items() if k != "history"})


if __name__ == "__main__":
    main()