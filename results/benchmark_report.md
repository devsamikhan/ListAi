# ListAI Benchmark Report: Pure Python Lists vs NumPy

**Generated On:** 2026-10-30 18:53:31  
**Repository:** [https://github.com/devsamikhan/ListAi](https://github.com/devsamikhan/ListAi)

---

## 1. System & Environment Telemetry

| Parameter | Specification |
|:---|:---|
| **Operating System** | `Windows-11-10.0.26300-SP0` |
| **Processor / Architecture** | `Intel64 Family 6 Model 142 Stepping 10, GenuineIntel` |
| **Python Version** | `3.14.4` (MSC v.1944 64 bit (AMD64)) |
| **NumPy Version** | `2.5.1` |
| **Dataset** | UCI Iris (150 samples: 105 Train, 24 Validation, 21 Test) |
| **Model Architecture** | Input (4) $\rightarrow$ Hidden 1 (16, ReLU) $\rightarrow$ Hidden 2 (8, ReLU) $\rightarrow$ Output (3, Softmax) |
| **Optimization** | Mini-batch SGD (Batch Size: 16, Learning Rate: 0.05, 100 Epochs) |

---

## 2. Training Performance (5 Repeated Trials)

> Timings measured using `time.perf_counter()`; peak heap memory tracked using `tracemalloc`.

| Engine | Median Time (s) | Mean Time (s) | Min Time (s) | Max Time (s) | Peak Memory (KB) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Engine A: Pure Python Lists** | **9.5460s** | 9.6318s | 9.5007s | 9.9508s | 401.77 KB |
| **Engine B: Independent NumPy** | **0.2546s** | 0.2649s | 0.2439s | 0.3189s | 134.92 KB |
| **Relative Factor** | **37.50x Faster** | 36.36x | — | — | — |

---

## 3. Inference Latency (100 Iterations on Test Set)

| Engine | Median Latency / Sample | Mean Latency / Sample | Min Latency / Sample | Max Latency / Sample |
|:---|:---:|:---:|:---:|:---:|
| **Engine A: Pure Python Lists** | **46.05 $\mu$s** | 46.16 $\mu$s | 44.34 $\mu$s | 50.16 $\mu$s |
| **Engine B: Independent NumPy** | **2.09 $\mu$s** | 2.12 $\mu$s | 2.06 $\mu$s | 4.16 $\mu$s |
| **Inference Speedup** | **22.08x Faster** | — | — | — |

---

## 4. Model Accuracy & Numerical Consistency

| Metric | Pure Python List Engine | Independent NumPy Engine | Cross-Engine Difference |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | **100.00%** | **100.00%** | Consistent (Both $\ge 95\%$) |
| **Forward Pass Parity** | Matched | Matched | **$< 10^{-12}$** |
| **Gradient Check Error** | Matched | Matched | **$< 10^{-6}$** |
| **Training Trajectory Parity** | Matched | Matched | **$< 10^{-9}$** |

---

## 5. Raw Matrix Multiplication Scaling: Lists vs BLAS

| Matrix Dimensions ($N \times N$) | Pure Python Lists (s) | NumPy BLAS (s) | NumPy Speedup Factor |
|:---|:---:|:---:|:---:|
| **10 $\times$ 10** | 0.162 ms | 0.002 ms | **85.50x** |
| **50 $\times$ 50** | 11.811 ms | 0.011 ms | **1064.06x** |
| **100 $\times$ 100** | 80.186 ms | 0.191 ms | **419.49x** |

---

## 6. Architectural Analysis: Why NumPy Outperforms Python Lists

### 1. Contiguous Memory Allocation vs Pointer Arrays
In standard CPython, a `list` is an array of pointers to `PyObject` structs. A nested matrix `List[List[float]]` consists of:
1. An outer list holding pointers to inner list objects.
2. Inner lists holding pointers to distinct `float` heap objects (each having 24 bytes overhead for reference count and type information).

Accessing `M[r][c]` causes a double pointer dereference and random memory access across the heap, causing **CPU cache misses** (L1/L2/L3). In contrast, a NumPy array allocates a single, contiguous block of raw 64-bit IEEE-754 floating-point bytes in C memory, allowing sequential memory streaming and maximum cache line utilization.

### 2. SIMD Vectorization & C/BLAS Acceleration
Python loops are interpreted opcode by opcode in the CPython virtual machine with dynamic type checks and GIL overhead on every scalar addition or multiplication. NumPy bypasses the interpreter by calling compiled C/Fortran routines (OpenBLAS, MKL) utilizing modern CPU vector instructions (**AVX-512, AVX2, FMA**), computing 4 to 8 double-precision floating-point multiplications per clock cycle per core.

### 3. Asymptotic and Constant Factor Realities
For micro-tensors (e.g., $1 \times 4$), Python object overhead dominates, but as matrix dimensions scale to $100 \times 100$ and beyond, the $\mathcal{O}(N^3)$ computational intensity is orders of magnitude faster in NumPy due to cache-blocking algorithms and multi-threading.
