"""Kompilasi tabel metrik komparatif (Bagian IV) dari semua JSON hasil eksperimen.

Jalankan: python -m scripts.make_table
Output: outputs/results/table_metrics.md  dan  table_metrics.csv
"""
import csv
import json

from src import Config as config

RES = config.RES_DIR


def jload(name):
    p = RES / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def fmt(v, nd=2):
    return "N/A" if v is None else f"{v:.{nd}f}"


def main():
    fid_all = (jload("fid_all") or {}).get("fid", {})
    fid_n = (jload("fid_all") or {}).get("n_samples")

    spec = [  # (nama, file json, dim laten, punya encoder, kunci latensi utama)
        ("Vanilla Autoencoder", "vanilla", f"{config.LATENT_DIM}", True, "latency_ms_recon_batch64"),
        ("Sparse Autoencoder", "sparse", f"{config.LATENT_DIM} (bottleneck) / 256 (sparse)", True,
         "latency_ms_recon_batch64"),
        ("Variational Autoencoder (VAE)", "vae", f"{config.LATENT_DIM}", True, "latency_ms_sample_batch64"),
        ("Vanilla GAN", "gan", f"{config.GAN_NOISE_DIM} (prior noise)", False, "latency_ms_sample_batch64"),
        ("DCGAN", "dcgan", f"{config.GAN_NOISE_DIM} (prior noise)", False, "latency_ms_sample_batch64"),
        ("DDPM", "ddpm", "28x28 (dimensi citra)", False, "latency_ms_sample_batch64"),
    ]

    header = ["Arsitektur", "Dim Laten", "Waktu/Epoch (s)", "Latensi (ms/batch 64)", "Test MSE",
              "Test PSNR (dB)", "Test SSIM", "FID", "Kestabilan Training"]
    rows, ext = [], []
    for name, key, dim, has_enc, lat_key in spec:
        d = jload(key)
        if d is None:
            rows.append([name, dim] + ["(belum dijalankan)"] * 7)
            continue
        t = d.get("test", {})
        fid = fid_all.get(key, d.get("fid")) if key in ("vae", "gan", "dcgan", "ddpm") else None
        rows.append([
            name, dim, fmt(d.get("avg_epoch_time_s"), 1), fmt(d.get(lat_key)),
            fmt(t.get("mse"), 5) if has_enc else "N/A (tanpa encoder)",
            fmt(t.get("psnr")) if has_enc else "N/A", fmt(t.get("ssim"), 4) if has_enc else "N/A",
            fmt(fid) if key in ("vae", "gan", "dcgan", "ddpm") else "N/A (latent voids)",
            "[isi dari pengamatan kurva loss]",
        ])
        ext.append({"model": key, "params": d.get("params", d.get("params_G")),
                    "latency_recon_ms": d.get("latency_ms_recon_batch64"),
                    "latency_sample_ms": d.get("latency_ms_sample_batch64"),
                    "latency_ddpm_single_step_ms": d.get("latency_ms_single_step_batch64"),
                    "fid": fid, "fid_n_samples": fid_n if key in fid_all else d.get("fid_samples")})

    md = ["# Tabel Komparasi Metrik (Test Set)\n", "| " + " | ".join(header) + " |",
          "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["",
           "Catatan: latensi AE/Sparse AE = rekonstruksi (encoder+decoder); VAE/GAN/DCGAN/DDPM = "
           "sampling dari prior (batch 64). "
           f"FID dihitung dengan n = {fid_n if fid_n else '(lihat fid_all.json)'} sampel untuk semua model.", ""]

    # Statistik sparsity hidden layer (Q2)
    md += ["## Sparsity hidden layer 256-dim (test set)\n", "| Model | Fraksi unit = 0 | Rata-rata aktivasi |",
           "|---|---|---|"]
    for key in ("vanilla", "sparse", "vae"):
        d = jload(key)
        if d and "hidden_layer_stats" in d:
            s = d["hidden_layer_stats"]
            lam = f" (lambda={d['lambda']})" if "lambda" in d else ""
            md.append(f"| {key}{lam} | {s['frac_zero']:.3f} | {s['mean_activation']:.4f} |")

    # Eksperimen MLP Classifier (Bagian III)
    m = jload("mlp_limits")
    if m:
        md += ["", "## Eksperimen batasan MLP Classifier\n",
               f"Test accuracy MLP Classifier: {m['classifier_test_acc']:.4f}\n",
               "| Metode | MSE | PSNR (dB) | SSIM |", "|---|---|---|---|"]
        for label, k in [("Frozen Decoder (z_MLP)", "frozen_decoder"), ("Vanilla AE (z_AE)", "vanilla_ae"),
                         ("Transpos W^T dari z_MLP", "transpose_from_latent")]:
            r = m[k]
            md.append(f"| {label} | {r['mse']:.5f} | {r['psnr']:.2f} | {r['ssim']:.4f} |")
        ps = m.get("prior_sampling", {})
        if ps:
            md.append(f"\nPrior sampling: {ps['frac_negative_dims_in_z_rand'] * 100:.1f}% dimensi z_rand bernilai "
                      f"negatif, padahal z_MLP (keluaran ReLU) selalu >= 0.")

    (RES / "table_metrics.md").write_text("\n".join(md), encoding="utf-8")
    with open(RES / "table_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("\n".join(md))
    print("\nTersimpan di", RES)


if __name__ == "__main__":
    main()