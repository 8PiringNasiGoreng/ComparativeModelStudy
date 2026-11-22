import torch
from torchmetrics.functional.image import structural_similarity_index_measure as _ssim
from torchmetrics.image.fid import FrechetInceptionDistance

from src import Config


@torch.no_grad()
def evaluate_reconstruction(recon_fn, loader) -> dict:
    mse_sum, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0
    for x, _ in loader:
        x = x.to(Config.DEVICE)
        x_hat = recon_fn(x).clamp(0, 1)

        mse_img = ((x - x_hat) ** 2).flatten(1).mean(dim=1)          # (B,)
        psnr_img = 10.0 * torch.log10(1.0 / mse_img.clamp_min(1e-10))  # (B,)
        ssim_img = _ssim(x_hat, x, data_range=1.0, reduction="none")   # (B,)

        mse_sum += mse_img.sum().item()
        psnr_sum += psnr_img.sum().item()
        ssim_sum += ssim_img.sum().item()
        n += x.size(0)
    return {"mse": mse_sum / n, "psnr": psnr_sum / n, "ssim": ssim_sum / n}


@torch.no_grad()
def compute_fid(real_loader, sample_fn, n_samples: int = 10000, batch_size: int = 250) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(Config.DEVICE)

    def _prep(t):
        t = t.to(Config.DEVICE).clamp(0, 1)
        return t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t

    seen = 0
    for x, _ in real_loader:
        x = x[: n_samples - seen]            # potong agar jumlah citra riil = n_samples
        fid.update(_prep(x), real=True)
        seen += x.size(0)
        if seen >= n_samples:
            break

    generated = 0
    while generated < n_samples:
        b = min(batch_size, n_samples - generated)
        fid.update(_prep(sample_fn(b)), real=False)
        generated += b

    return fid.compute().item()