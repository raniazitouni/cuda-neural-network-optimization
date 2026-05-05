# CUDA Neural Network — Alternative Parallelization Strategy

> **Academic project** | ESI Algiers — HPC Module | 2CS SIQ1 | 2025–2026  
> Task: propose, implement, and evaluate an alternative GPU parallelization strategy for a shallow neural network vs. a reference baseline — published as a scientific article.

---

## Overview

This repository contains:

- The **reference baseline** (`reference_code/nn_cuda.cu`) — naive CUDA matrix multiplication using global memory only
- Our **optimized alternative strategy** (`src/optimized_cuda.cu`) — three stacked GPU optimizations
- All datasets, results, scripts, and the scientific article

**Network:** `INPUT(32) → HIDDEN(256, ReLU) → OUTPUT(1)` — shallow neural network for regression

```
Reference strategy               Our alternative strategy
──────────────────────           ────────────────────────────────────────
Naive global memory matmul   →   Tiled shared memory (16×16 tiles)
Uncoalesced B matrix access  →   Pre-transposed B for coalesced reads
3 separate kernel launches   →   1 fused kernel (MatMul + ReLU + MatMul)
No bank conflict handling    →   +1 padding eliminates shared mem conflicts
```

---

## Repository Structure

```
cuda-neural-network-optimization/
│
├── src/
│   └── optimized_cuda.cu            # Our alternative strategy (3 optimizations)
│
├── reference_code/                  # Reference baseline (provided by ESI)
│   ├── nn_cuda.cu                   # Reference CUDA implementation
│   ├── nn_pthreads.c                # Reference Pthreads implementation
│   ├── nn.c                         # Reference sequential implementation
│   ├── test_cuda.cu                 # CUDA unit tests
│   ├── visualize.ipynb              # Training visualization notebook
│   ├── run.sh                       # Run sequential
│   ├── run_cuda.sh                  # Run reference CUDA
│   ├── run_pthreads.sh              # Run reference Pthreads
│   ├── test_cuda.sh                 # Run CUDA tests
│   ├── Rapport de la stratégie de référence (1).pdf
│   ├── data/                        # Reference data copies
│   ├── log/                         # Reference training logs + plots
│   ├── playground/                  # Data generation + plotting scripts
│   └── results/                     # Reference benchmark results
│       ├── cuda.csv                 # Reference CUDA timings
│       ├── gpu.txt                  # GPU specs used
│       ├── pthreads.csv
│       └── sequential.csv
│
├── data/                            # Datasets (shared by both strategies)
│   ├── README.md                    # Dataset format + regeneration instructions
│   ├── synthetic_convex_small.csv   # 256 samples
│   ├── synthetic_convex_medium.csv  # 2,560 samples
│   └── synthetic_convex_large.csv   # 25,600 samples
│
├── implementation paper/
│   └── paper.pdf                    # Scientific article (English, ≤12 pages)
│
├── playground/
│   ├── generate_synthetic_data.py   # Dataset generation script
│   └── plot_training.py             # Training curve plotting
│
├── results/
│   ├── cuda.csv                     # Reference CUDA benchmark results
│   └── gpu.txt                      # GPU used: NVIDIA GeForce GTX 1650
│
├── MakeFile                         # Build both strategies
├── run_optimized.sh                 # Linux/Mac: run our strategy on all datasets
├── run_compare.sh                   # Linux/Mac: run both + compute speedup
├── run_optimized.bat                # Windows: run our strategy
└── run_compare.bat                  # Windows: run both + compute speedup
```

---

## Our Three Optimizations

### Optimization 1 — GPU Transpose Kernel

```cuda
__global__ void transpose_kernel(float *input, float *output, int rows, int cols)
```

Transposes weight matrices W1 and W2 **on the GPU** before matrix multiplication:

- Shared memory tiling with `+1` column padding → **eliminates bank conflicts**
- Coalesced reads when loading the input tile
- Coalesced writes when storing the transposed tile
- Runs once per epoch, amortized over all batches

### Optimization 2 — Tiled Matrix Multiplication with Pre-Transposed B

```cuda
__global__ void mat_mult_tiled_kernel_transpose(float *A, float *B_T, float *C, ...)
```

Replaces the reference's naive per-element global memory reads:

- Loads 16×16 tiles of A and Bᵀ into shared memory per thread block
- Since B is pre-transposed, all global reads are **fully coalesced**
- Reduces global memory traffic by factor of `TILE_SIZE` (16×) vs naive
- `#pragma unroll` on inner dot product loop for register-level speed

### Optimization 3 — Fused Forward Pass Kernel

```cuda
__global__ void fused_forward_pass_kernel(float *X, float *W1_T, float *W2_T, ...)
```

The highest-impact optimization — merges the entire forward pass into **one kernel launch**:

```
Phase 1:  Z1     = X × W1ᵀ        tiled matmul (shared memory)
Phase 2:  H      = ReLU(Z1)        computed in shared memory — never hits global mem
Phase 3:  Y_pred = H × W2ᵀ        tiled matmul reusing H from shared memory
```

vs. reference which launches 3 separate kernels, each writing/reading global memory between steps.

---

## Network & Training Parameters

| Parameter      | Value              |
| -------------- | ------------------ |
| Input features | 32                 |
| Hidden neurons | 256                |
| Output neurons | 1                  |
| Activation     | ReLU               |
| Loss           | Mean Squared Error |
| Optimizer      | SGD                |
| Learning rate  | 0.002              |
| Batch size     | 256                |
| Epochs         | 100                |
| Tile size      | 16 × 16            |

---

## Reference Baseline Results

Timings from `results/cuda.csv` (GPU: NVIDIA GeForce GTX 1650, Compute 7.5):

| Dataset | Samples | Reference time (s) |
| ------- | ------- | ------------------ |
| small   | 256     | ~0.45–0.50         |
| medium  | 2,560   | ~2.59–2.86         |
| large   | 25,600  | ~22.22–25.75       |

_Our optimized results are generated by running `run_compare.sh` — see Getting Started below._

---

## Getting Started

### Requirements

- NVIDIA GPU (Compute Capability ≥ 5.0)
- CUDA Toolkit ≥ 11.0
- GCC + OpenMP
- `nvcc` in PATH

### No GPU? Run on Google Colab (free)

Click the **Open in Colab** badge at the top, or follow these steps:

1. Go to [colab.research.google.com](https://colab.research.google.com) → New notebook
2. **Runtime → Change runtime type → T4 GPU**
3. Run these cells:

```bash
!nvidia-smi                          # verify GPU is available
!git clone https://github.com/raniazitouni/cuda-neural-network-optimization
%cd cuda-neural-network-optimization
!make all                            # compile both strategies
!./run_compare.sh                    # benchmark both + print speedup table
```

### Local (Linux / Mac)

```bash
# Clone
git clone https://github.com/raniazitouni/cuda-neural-network-optimization
cd cuda-neural-network-optimization

# Build both binaries
make all

# Run only our optimized strategy
./run_optimized.sh

# Run both strategies and compare speedup
./run_compare.sh
```

### Local (Windows)

```bat
git clone https://github.com/raniazitouni/cuda-neural-network-optimization
cd cuda-neural-network-optimization

:: Build
nvcc -O2 -Xcompiler /openmp -o nn_optimized.exe src\optimized_cuda.cu

:: Run
run_optimized.bat
run_compare.bat
```

### Makefile targets

```bash
make all        # build both nn_optimized and nn_reference
make optimized  # build our strategy only  →  nn_optimized
make reference  # build baseline only      →  nn_reference
make clean      # remove compiled binaries
make help       # show all targets
```

---

_École Nationale Supérieure d'Informatique (ESI), Algiers — 2025/2026_  
_2CS SIQ1 — HPC Module_
