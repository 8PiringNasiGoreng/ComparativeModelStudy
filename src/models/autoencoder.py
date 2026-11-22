import torch
import torch.nn as nn
import torch.nn.functional as F

def _decoder(latent: int, hidden: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(latent, hidden), 
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
        nn.Sigmoid()
    )

class VanillaAE(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256, latent: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = _decoder(latent, hidden, in_dim)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Aktivasi hidden layer 256-dim (untuk perbandingan sparsity)."""
        return self.encoder[1](self.encoder[0](x.flatten(1)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)).view_as(x)

class SparseAE(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256, latent: int = 64):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hidden)
        self.enc2 = nn.Linear(hidden, latent)
        self.decoder = _decoder(latent, hidden, in_dim)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.enc1(x.flatten(1)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc2(self.hidden(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_with_h(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        x_hat = self.decode(self.enc2(h)).view_as(x)
        return x_hat, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_h(x)[0]

class VAE(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256, latent: int = 64, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent = latent
        self.img_shape = img_shape
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = _decoder(latent, hidden, in_dim)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x.flatten(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)       # z = mu + sigma * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view_as(x), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Rekonstruksi deterministik memakai mean posterior mu (untuk evaluasi)."""
        mu, _ = self.encode(x)
        return self.decode(mu).view_as(x)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Generasi dari prior z ~ N(0, I) (single-pass)."""
        z = torch.randn(n, self.latent, device=next(self.parameters()).device)
        return self.decode(z).view(n, *self.img_shape)


def build_ae(name: str, in_dim: int = 784, latent: int = 64, img_shape: tuple = (1, 28, 28)) -> nn.Module:
    if name == "vanilla":
        return VanillaAE(in_dim=in_dim, latent=latent)
    if name == "sparse":
        return SparseAE(in_dim=in_dim, latent=latent)
    if name == "vae":
        return VAE(in_dim=in_dim, latent=latent, img_shape=img_shape)
    raise ValueError(f"Model tidak dikenal: {name}")