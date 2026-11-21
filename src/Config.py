from pathlib import Path
import torch

SEED = 42

DATASET = "FashionMNIST"
BATCH_SIZE = 128
NUM_WORKERS = 2
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

LATENT_DIM = 64
GAN_NOISE_DIM = 100
LAMBDA_SPARSE = 1e-2  

#Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
RES_DIR = OUT_DIR / "results"
#Path validation
for _d in (DATA_DIR, FIG_DIR, CKPT_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

