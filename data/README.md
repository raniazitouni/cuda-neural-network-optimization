# Data

This folder contains the synthetic datasets used to benchmark both strategies.

## Files

| File | Samples | Size on disk | Purpose |
|---|---|---|---|
| `synthetic_convex_small.csv` | 256 | ~163 KB | Quick functional test |
| `synthetic_convex_medium.csv` | 2,560 | ~1.6 MB | Medium benchmark |
| `synthetic_convex_large.csv` | 25,600 | ~16 MB | Full performance benchmark |

## Format

Each CSV file has **no header row**. Each row contains:
```
X1, X2, ..., X32, Y
```
- 32 input features (float, sampled from Uniform[−1, 1])
- 1 target label Y = Σ(Xᵢ²) + noise (convex function with Gaussian noise σ=0.1)

## Regenerate the Data

If you need to regenerate the datasets (e.g. different sizes), use the script in `playground/`:

```bash
python3 playground/generate_synthetic_data.py
```

Modify `num_samples` in the script to control dataset size. The script always generates the **large** variant — adjust the filename output accordingly for small/medium.

## Why This Dataset?

The convex function Σ(Xᵢ²) is smooth and well-conditioned, making it a clean benchmark for regression with a shallow neural network. The focus of this project is **GPU performance**, not model accuracy — the dataset is designed to be computationally representative without introducing confounding factors.