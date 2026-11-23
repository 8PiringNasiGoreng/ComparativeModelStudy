import argparse
import json

from src import Config as config
from src.data import get_data
from src.loading import img_shape, load_ae, load_ddpm, load_gan
from src.metrics import compute_fid
from src.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="jumlah sampel (maks. ukuran test set)")
    ap.add_argument("--models", nargs="+", default=["vae", "gan", "dcgan", "ddpm"],
                    choices=["vae", "gan", "dcgan", "ddpm"])
    args = ap.parse_args()

    out_path = config.RES_DIR / "fid_all.json"
    store = {"n_samples": args.n, "fid": {}}
    if out_path.exists():
        old = json.load(open(out_path))
        if old.get("n_samples") == args.n:
            store = old
        else:
            print(f"n_samples berbeda ({old.get('n_samples')} vs {args.n}) -> mulai dari awal.")

    set_seed()
    _, _, test_loader = get_data()
    shape = img_shape()

    for m in args.models:
        set_seed()
        if m == "vae":
            sample_fn = load_ae("vae").sample
        elif m in ("gan", "dcgan"):
            sample_fn = load_gan(m).sample
        else:
            unet, diff = load_ddpm()
            sample_fn = lambda k: diff.sample(unet, k, shape)
        print(f"Menghitung FID {m} (n={args.n}) ...")
        store["fid"][m] = compute_fid(test_loader, sample_fn, n_samples=args.n)
        print(f"  FID {m}: {store['fid'][m]:.2f}")
        with open(out_path, "w") as f:       # simpan bertahap
            json.dump(store, f, indent=2)

    print("Tersimpan:", out_path)


if __name__ == "__main__":
    main()