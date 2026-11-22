import json

import torch

from src import Config
from src.data import dataset_info
from src.models.autoencoder import build_ae
from src.models.ddpm import GaussianDiffusion, UNet
from src.models.gan import build_gan
from src.models.mlp import MLPClassifier

DEV = Config.DEVICE


def img_shape():
    c, s = dataset_info()
    return (c, s, s)


def _ckpt(name):
    path = Config.CKPT_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}. Jalankan skrip training-nya dulu.")
    return torch.load(path, map_location=DEV)


def load_ae(name: str):
    shape = img_shape()
    m = build_ae(name, in_dim=shape[0] * shape[1] * shape[2], latent=Config.LATENT_DIM, img_shape=shape).to(DEV)
    m.load_state_dict(_ckpt(name))
    return m.eval()


def load_gan(name: str):
    g, _ = build_gan(name, Config.GAN_NOISE_DIM, img_shape())
    g = g.to(DEV)
    g.load_state_dict(_ckpt(name)["G"])
    return g.eval()


def load_ddpm(base: int = 64, timesteps: int | None = None):
    if timesteps is None:
        meta = Config.RES_DIR / "ddpm.json"
        timesteps = json.load(open(meta))["timesteps"] if meta.exists() else 1000
    unet = UNet(in_ch=img_shape()[0], base=base).to(DEV)
    unet.load_state_dict(_ckpt("ddpm"))
    return unet.eval(), GaussianDiffusion(timesteps).to(DEV)


def load_classifier():
    shape = img_shape()
    clf = MLPClassifier(in_dim=shape[0] * shape[1] * shape[2], latent=Config.LATENT_DIM).to(DEV)
    clf.load_state_dict(_ckpt("mlp_classifier"))
    return clf.eval()