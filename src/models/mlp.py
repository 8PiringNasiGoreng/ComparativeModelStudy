import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256, latent: int = 64, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, latent)
        self.fc3 = nn.Linear(latent, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(h1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))

    @torch.no_grad()
    def backproject_class(self, y_onehot: torch.Tensor) -> torch.Tensor:
        z_hat = F.relu(y_onehot @ self.fc3.weight)
        h1_hat = F.relu(z_hat @ self.fc2.weight)
        return torch.sigmoid(h1_hat @ self.fc1.weight)

    @torch.no_grad()
    def backproject_latent(self, z: torch.Tensor) -> torch.Tensor:
        h1_hat = F.relu(z @ self.fc2.weight)
        return torch.sigmoid(h1_hat @ self.fc1.weight)

class FrozenLatentDecoder( nn.Module):
    def __init__(self, latent: int = 64, hidden: int = 256, out_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    