import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from src import Config

DATASETS = {
    "FashionMNIST" : (datasets.FashionMNIST, 1, 28),
}

def dataset_info():
    _, channels, size = DATASETS[Config.DATASET]
    return channels, size

def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_0_1(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0, 1)

def _seed_worker(worker_id:int) -> None:
    import numpy as np, random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data(batch_size: int = Config.BATCH_SIZE):
    cls ,_ ,_ = DATASETS[Config.DATASET]
    tfm = transforms.ToTensor()
    Data = ConcatDataset([
        cls(Config.DATA_DIR, train=True, download = True, transform = tfm),
        cls(Config.DATA_DIR, train=False, download = True, transform = tfm),  
    ])


    n = len(Data)
    n_train = int(n * Config.TRAIN_RATIO)
    n_val = int(n * Config.VAL_RATIO)
    n_test = int(n * Config.TEST_RATIO)

    gen = torch.Generator().manual_seed(Config.SEED)
    D_train, D_val, D_test = random_split(Data, [n_train, n_val, n_test], generator = gen)

    loader_gen = torch.Generator().manual_seed(Config.SEED)
    kw = dict(batch_size=batch_size, 
              num_workers=Config.NUM_WORKERS,
              worker_init_fn=_seed_worker, 
              pin_memory=Config.DEVICE.type == "cuda")

    train_loader = DataLoader(D_train, shuffle=True, generator=loader_gen, **kw)
    val_loader = DataLoader(D_val, shuffle=False)
    test_loader = DataLoader(D_test, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tr, va, te = get_data()
    print(f"Dataset: {Config.DATASET}")
    print(f"Train: {len(tr.dataset)} | Val: {len(va.dataset)} | Test: {len(te.dataset)}")
    x, y = next(iter(tr))
    print("Batch:", x.shape, "range:", float(x.min()), "-", float(x.max()))
