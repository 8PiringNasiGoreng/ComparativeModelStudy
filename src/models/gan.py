import math

import torch
import torch.nn as nn


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0, 1)


# ====================================================== Vanilla GAN (MLP)
class VanillaGenerator(nn.Module):
    def __init__(self, noise_dim: int = 100, img_shape=(1, 28, 28), hidden=(256, 512, 1024)):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_shape = tuple(img_shape)
        layers, d = [], noise_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
            d = h
        layers += [nn.Linear(d, math.prod(self.img_shape)), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, *self.img_shape)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.noise_dim, device=next(self.parameters()).device)
        return _to_01(self(z))


class VanillaDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), hidden=(1024, 512, 256)):
        super().__init__()
        layers, d = [nn.Flatten()], math.prod(img_shape)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)          # logit, shape (B,)


# ============================================================== DCGAN
class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = 100, img_shape=(1, 28, 28)):
        super().__init__()
        c, size, _ = img_shape
        init = size // 4
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, init, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, c, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1, 1, 1))

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.noise_dim, device=next(self.parameters()).device)
        return _to_01(self(z))


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        c, size, _ = img_shape
        init = size // 4
        self.net = nn.Sequential(
            nn.Conv2d(c, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, init, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build_gan(name: str, noise_dim: int = 100, img_shape=(1, 28, 28)):
    if name == "gan":
        return VanillaGenerator(noise_dim, img_shape), VanillaDiscriminator(img_shape)
    if name == "dcgan":
        g, d = DCGANGenerator(noise_dim, img_shape), DCGANDiscriminator(img_shape)
        g.apply(_init_weights)
        d.apply(_init_weights)
        return g, d
    raise ValueError(f"Model tidak dikenal: {name}")