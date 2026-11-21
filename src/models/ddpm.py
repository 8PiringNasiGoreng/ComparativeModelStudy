"""DDPM (Ho et al., 2020): U-Net kecil + proses difusi Gaussian.

Forward   : q(x_t | x_0) = N( sqrt(abar_t) x_0 , (1 - abar_t) I )
Training  : L_simple = E || eps - eps_theta( sqrt(abar_t) x_0 + sqrt(1-abar_t) eps , t ) ||^2
Reverse   : x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t/sqrt(1-abar_t) * eps_theta(x_t, t) ) + sigma_t z,
            sigma_t^2 = beta_t

Data diskalakan ke [-1, 1]. `sample` mengembalikan gambar di [0, 1].
Indeks timestep di kode berbasis 0: t = 0 .. T-1.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- U-Net
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding posisi timestep -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(tdim, out_ch)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t_proj(F.silu(temb))[:, :, None, None]   # injeksi informasi timestep
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention satu kepala pada resolusi terendah (7x7 / 8x8)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [t.flatten(2).transpose(1, 2).unsqueeze(1) for t in (q, k, v)]   # (B,1,HW,C)
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """U-Net: 28x28 -> 14 -> 7 (mid + attention) -> 14 -> 28 dengan skip connection."""

    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.base = base
        tdim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb_d1 = ResBlock(base, base, tdim)
        self.down1 = nn.Conv2d(base, base, 3, 2, 1)
        self.rb_d2 = ResBlock(base, base * 2, tdim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, 2, 1)

        self.rb_m1 = ResBlock(base * 2, base * 2, tdim)
        self.attn = AttnBlock(base * 2)
        self.rb_m2 = ResBlock(base * 2, base * 2, tdim)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(base * 2, base * 2, 3, padding=1))
        self.rb_u2 = ResBlock(base * 4, base * 2, tdim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(base * 2, base * 2, 3, padding=1))
        self.rb_u1 = ResBlock(base * 3, base, tdim)

        self.out_norm = nn.GroupNorm(8, base)
        self.conv_out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h0 = self.conv_in(x)
        h1 = self.rb_d1(h0, temb)                    # skip 1
        h3 = self.rb_d2(self.down1(h1), temb)        # skip 2
        m = self.down2(h3)
        m = self.rb_m2(self.attn(self.rb_m1(m, temb)), temb)
        u = self.rb_u2(torch.cat([self.up2(m), h3], dim=1), temb)
        u = self.rb_u1(torch.cat([self.up1(u), h1], dim=1), temb)
        return self.conv_out(F.silu(self.out_norm(u)))   # prediksi derau eps_theta


# ------------------------------------------------------------- Diffusion
class GaussianDiffusion(nn.Module):
    """Menyimpan noise schedule (linear beta) dan operasi forward/reverse."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_one_minus_abar", torch.sqrt(1.0 - abar))

    @staticmethod
    def _ex(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return a[t].view(-1, 1, 1, 1)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps."""
        return self._ex(self.sqrt_abar, t) * x0 + self._ex(self.sqrt_one_minus_abar, t) * noise

    def loss(self, model, x0, generator=None):
        """L_simple: MSE antara derau asli dan prediksi derau."""
        b, dev = x0.size(0), x0.device
        t = torch.randint(0, self.T, (b,), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, n: int, img_shape, record: int = 0):
        """Reverse denoising x_T ~ N(0, I) -> x_0 (T langkah).

        record > 0: simpan `record` snapshot (dibagi rata) untuk visualisasi trajektori.
        Return: gambar [0,1] (n,C,H,W), atau (gambar, [(label_t, tensor), ...]).
        """
        dev = self.betas.device
        x = torch.randn(n, *img_shape, device=dev)
        snap_t = set(torch.linspace(self.T - 1, 0, record).long().tolist()) if record > 0 else set()
        traj = []
        view = lambda z: ((z + 1) / 2).clamp(0, 1)
        for t in range(self.T - 1, -1, -1):
            if t in snap_t:
                traj.append((t, view(x).cpu()))
            tb = torch.full((n,), t, device=dev, dtype=torch.long)
            eps = model(x, tb)
            mean = self.sqrt_recip_alphas[t] * (x - self.betas[t] / self.sqrt_one_minus_abar[t] * eps)
            x = mean + torch.sqrt(self.betas[t]) * torch.randn_like(x) if t > 0 else mean
        out = view(x)
        if record > 0:
            traj.append(("x0", out.cpu()))
            return out, traj
        return out