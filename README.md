<!-- Banner -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=100&section=header&text=CUDA%20High-Performance%20Demos&fontSize=40&animation=fadeIn" />
</p>

## 🎯 Project Overview

**CUDA High-Performance Demos** is a collection of CUDA‐accelerated vector and matrix routines showcasing advanced optimization techniques:

- Shared memory tiling for coalesced global loads  
- Warp‐level optimization to minimize divergence  
- Tiling strategies to maximize data reuse and throughput  

Each demo comes in a baseline version and a shared‐memory‐optimized (`SH_`) variant, so you can compare performance gains side by side.

---

## 📥 Installation & Setup

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

## 🐳 Docker Image

You can pull and run the latest version of the CUDA-high-performance-demos from Docker Hub:

```bash
docker pull orlando2705/cuda-high-perf-demos:v1.1
docker run --rm --gpus all orlando2705/cuda-high-perf-demos:v1.1
```
---

## 🏃 How to Run


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

## ✨ Features

- **SH_ variants** use shared memory tiling to reduce global memory traffic.  
- **Warp‐level primitives** for fast reductions and minimized divergence.  
- **Parameterizable block/grid sizes** for auto‐tuning.  
- **Side‐by‐side** baseline vs. optimized implementations for performance comparison.  

---

## 📂 Project Structure
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

## 🚀 GPU Kernel Performance Report

### 1️⃣ Vector Addition – Naive Implementation
- **Input size:** `33,554,432` elements  
- **Kernel time:** `54.28 ms`  
- **Description:**  
  Each thread processes a single element. This approach suffers from poor scalability and inefficient memory usage.

![Exercise of sum of vector without optimization(course)](https://github.com/user-attachments/assets/29d169f4-57fc-4b81-9411-22c7d65616aa)

---

### 2️⃣ Vector Addition – Optimized with Grid-Stride Loop

- **Input size:** `33,554,432` elements  
- **Kernel time:** `1.61 ms`  
- **Description:**  
  Using the **grid-stride loop** pattern, each thread handles multiple elements.  
  This drastically improves resource utilization and reduces execution time by **~34×** compared to the naive version.
![sum of vectors with optimization ](https://github.com/user-attachments/assets/08de80a8-e59a-4b0e-976a-2d7000e4694e)

---

### 3️⃣ Vector Normalization – Shared Memory + Warp-Level Optimization

- **Input size:** `33,554,432` elements  
- **Kernel time (all stages):** `2.98 ms`  
- **Description:**  
  - Uses **binary reduction** in shared memory to accumulate partial sums.  
  - Applies **warp shuffle intrinsics** (`__shfl_down_sync`) for efficient intra-warp reduction.  
  - Demonstrates how combining **shared memory** and **warp-level primitives** results in high-performance kernels for reduction-type operations.
![ Normalize Vector with Shared Memory](https://github.com/user-attachments/assets/dc5d86bd-39cd-474c-b71a-e3fb7812c04b)

---

## 📊 Performance Comparison

| Implementation                                           | Input Size     | Kernel Time | Speedup vs Naive |
|----------------------------------------------------------|---------------:|------------:|-----------------:|
| Vector Addition – Naive                                  | 33,554,432     | 54.28 ms    | 1×               |
| Vector Addition – Grid-Stride Loop                       | 33,554,432     | 1.61 ms     | ~34×             |
| Vector Normalization – Shared Memory + Warp Optimization | 33,554,432     | 2.98 ms     | ~18×             |

---

### 💡 Key Takeaways:
- **Grid-stride loops** are a simple yet powerful optimization for memory-bound kernels.
- **Shared memory + warp-level primitives** are essential for high-performance reductions.
- Even with the same input size, kernel design can yield **orders of magnitude** performance differences.

---

## 🚀 How It Works

- **Baseline Execution**: Runs kernels that read and write directly from global memory without shared memory usage.  
- **Shared Memory Tiling**: Optimized versions split data into tiles stored in shared memory, processed cooperatively by threads, then written back to global memory.  
- **Warp-Level Optimization**: Uses warp-level primitives like `__shfl_down_sync` to perform fast intra‑warp reductions without shared memory.  
- **Performance Comparison**: Each demo includes both baseline and optimized versions to measure execution time, throughput, and the impact of shared memory and warp-level operations.  

## 🛠 Technologies Used

- **CUDA C/C++** – Core language for implementing high‑performance GPU kernels.  
- **NVIDIA CUDA Toolkit** – Provides compiler (`nvcc`), runtime libraries, and development utilities.  
- **Shared Memory & Warp-Level Primitives** – GPU optimization techniques for reduced latency and higher throughput.  
- **CUDA Events** – For precise kernel execution timing and performance measurement.  
- **Docker** – Containerization for consistent, portable builds and environment setup across systems.

## 🚀 Future Improvements
- Optimize kernels using **LLVM** and custom **PTX** tuning for low‑level performance gains.  
- Implement **multi‑GPU synchronization** and collective operations via **NVIDIA NCCL** for distributed execution.  
- Add support for advanced AI‑related kernels such as **softmax** and common **loss functions** (e.g., cross‑entropy, MSE).  
- Extend profiling and benchmarking suite to measure scalability across multiple GPUs.  
- Provide **Docker** setup for reproducible, portable GPU development environments.  

---
