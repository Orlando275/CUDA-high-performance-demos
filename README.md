## Project Overview

**CUDA High-Performance Demos** is a collection of CUDA‐accelerated vector and matrix routines showcasing advanced optimization with these techniques:

- Shared memory tiling for coalesced global loads  
- Warp‐level optimization to minimize divergence  
- Tiling strategies to maximize data reuse and throughput  

---

## Installation

### 1. Prerequisites

- NVIDIA GPU with Compute Capability ≥ 5.0  
- CUDA Toolkit (≥ 10.0) installed and in your `PATH`  
- Docker installation

### 2. Clone this repository

```bash
git clone https://github.com/Orlando275/CUDA-high-performance-demos.git
cd CUDA-high-performance-demos
```

---

## Docker Image

You can pull and run the latest version of the CUDA-high-performance-demos from Docker Hub:

```bash
docker pull orlando2705/cuda-high-perf-demos:v1.1
docker run --rm --gpus all orlando2705/cuda-high-perf-demos:v1.1
```
---

## How to Run


### Example: matrix multiplication (Shared Memory)
```bash
nvcc SH_matrix_multiplication.cu -o matrix_multiplication 
./matrix_multiplication
```

### Example imput
```bash
200  20  120    # M x N, N x P
```

### Example: vector sum
```bash
./sum_of_vectors
10000000
./SH_total_vector_sum
10000000
```

---

## Features

- **SH_ variants** use shared memory tiling to reduce global memory traffic.  
- **Warp‐level primitives** for fast reductions and minimized divergence.

---

## Project Structure
<pre>
CUDA-high-performance-demos/
├── Vectors/
│   ├── normalize_vector.cu
│   ├── SH_normalize_vector.cu
│   ├── SH_total_vector_sum.cu
|   └── sum_of_vectors.cu
├── Matrices/
│   ├── matrix_multiplication.cu
│   └── SH_matrix_multiplication.cu
├── .gitignore
├── Dockerfile
├── README.md
</pre>

---

## GPU Kernel Performance Report

### 1. Vector Addition (naive Implementation)
- **Input size:** `33,554,432` elements  
- **Kernel time:** `54.28 ms`  
- **Description:**  
  Each thread processes a single element. This approach suffers from poor scalability and inefficient memory usage.

![Exercise of sum of vector without optimization(course)](https://github.com/user-attachments/assets/29d169f4-57fc-4b81-9411-22c7d65616aa)

---

### 2. Vector Addition (Optimized Version).

- **Input size:** `33,554,432` elements  
- **Kernel time:** `1.61 ms`  
- **Description:**    
  This drastically improves resource utilization and reduces execution time by **~34×** compared to the naive version.
![sum of vectors with optimization ](https://github.com/user-attachments/assets/08de80a8-e59a-4b0e-976a-2d7000e4694e)

---

### 3. Vector Normalization (Shared Memory and Warp-Level Optimization)

- **Input size:** `33,554,432` elements  
- **Kernel time (all stages):** `2.98 ms`  
- **Description:**  
  - Uses **binary reduction** in shared memory to accumulate partial sums.  
  - Applies **warp shuffle intrinsics** (`__shfl_down_sync`) for efficient intra-warp reduction.  
![ Normalize Vector with Shared Memory](https://github.com/user-attachments/assets/dc5d86bd-39cd-474c-b71a-e3fb7812c04b)

---

## Performance Comparison

| Implementation                                           | Input Size     | Kernel Time | Speedup vs Naive |
|----------------------------------------------------------|---------------:|------------:|-----------------:|
| Vector Addition – Naive                                  | 33,554,432     | 54.28 ms    | 1×               |
| Vector Addition – Grid-Stride Loop                       | 33,554,432     | 1.61 ms     | ~34×             |
| Vector Normalization – Shared Memory + Warp Optimization | 33,554,432     | 2.98 ms     | ~18×             |

---

## What did I do

- **Baseline Execution**: Runs kernels that read and write directly from global memory without shared memory usage.  
- **Shared Memory Tiling**: Optimized versions split data into tiles stored in shared memory, processed cooperatively by threads, then written back to global memory.  
- **Warp-Level Optimization**: Uses warp-level primitives like `__shfl_down_sync` to perform fast intra‑warp reductions without shared memory.  
- **Performance Comparison**: Each demo includes both baseline and optimized versions to measure execution time, throughput, and the impact of shared memory and warp-level operations.  

## Future Improvements
- Optimize kernels using **LLVM** and custom **PTX** tuning for low‑level performance gains.
