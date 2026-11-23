# Tabel Komparasi Metrik (Test Set)

| Arsitektur | Dim Laten | Waktu/Epoch (s) | Latensi (ms/batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID | Kestabilan Training |
|---|---|---|---|---|---|---|---|---|
| Vanilla Autoencoder | 64 | 33.7 | 0.09 | 0.00616 | 23.27 | 0.8737 | N/A (latent voids) | [isi dari pengamatan kurva loss] |
| Sparse Autoencoder | 64 (bottleneck) / 256 (sparse) | 32.1 | 0.11 | 0.00748 | 22.35 | 0.8492 | N/A (latent voids) | [isi dari pengamatan kurva loss] |
| Variational Autoencoder (VAE) | 64 | 30.4 | 0.17 | 0.01289 | 19.77 | 0.7712 | 82.16 | [isi dari pengamatan kurva loss] |
| Vanilla GAN | 100 (prior noise) | 9.2 | 0.35 | N/A (tanpa encoder) | N/A | N/A | 107.98 | [isi dari pengamatan kurva loss] |
| DCGAN | 100 (prior noise) | 17.3 | 8.51 | N/A (tanpa encoder) | N/A | N/A | 20.09 | [isi dari pengamatan kurva loss] |
| DDPM | 28x28 (dimensi citra) | 54.9 | 21060.86 | N/A (tanpa encoder) | N/A | N/A | N/A | [isi dari pengamatan kurva loss] |

Catatan: latensi AE/Sparse AE = rekonstruksi (encoder+decoder); VAE/GAN/DCGAN/DDPM = sampling dari prior (batch 64). FID dihitung dengan n = 5000 sampel untuk semua model.

## Sparsity hidden layer 256-dim (test set)

| Model | Fraksi unit = 0 | Rata-rata aktivasi |
|---|---|---|
| vanilla | 0.433 | 0.9961 |
| sparse (lambda=0.01) | 0.786 | 0.0629 |
| vae | 0.796 | 0.2964 |

## Eksperimen batasan MLP Classifier

Test accuracy MLP Classifier: 0.8954

| Metode | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| Frozen Decoder (z_MLP) | 0.01808 | 18.26 | 0.7118 |
| Vanilla AE (z_AE) | 0.00647 | 23.02 | 0.8683 |
| Transpos W^T dari z_MLP | 0.31726 | 5.03 | 0.0371 |

Prior sampling: 17.6% dimensi z_rand bernilai negatif, padahal z_MLP (keluaran ReLU) selalu >= 0.