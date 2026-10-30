# ListAI: Understanding Neural Networks Through Python Lists

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.14%20%7C%203.15-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-devsamikhan%2FListAi-success.svg)](https://github.com/devsamikhan/ListAi)
[![Tests Passing](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)](#testing-and-verification)

> **Tagline:** Understanding Neural Networks Through Python Lists.  
> **Repository:** [https://github.com/devsamikhan/ListAi](https://github.com/devsamikhan/ListAi)

---

## Table of Contents
1. [Project Introduction](#project-introduction)
2. [Why Build Neural Networks with Python Lists?](#why-build-neural-networks-with-python-lists)
3. [Architecture Overview](#architecture-overview)
4. [Dataset & Preprocessing Pipeline](#dataset--preprocessing-pipeline)
5. [Mathematical Formulations](#mathematical-formulations)
6. [Installation & Setup](#installation--setup)
7. [Command-Line Interface (CLI)](#command-line-interface-cli)
8. [Testing & Verification](#testing--verification)
9. [Empirical Benchmarks: Lists vs. NumPy](#empirical-benchmarks-lists-vs-numpy)
10. [Why NumPy Outperforms Python Lists](#why-numpy-outperforms-python-lists)
11. [Limitations & Future Improvements](#limitations--future-improvements)
12. [License](#license)

---

## Project Introduction

**ListAI** is an advanced educational numerical computing and artificial intelligence project designed to demystify neural networks from first principles.

Instead of relying on black-box frameworks like PyTorch, TensorFlow, or Scikit-learn, ListAI builds a complete, multi-layer artificial neural network from scratch using **native Python Lists** and standard library primitives (`math`, `random`). To scientifically validate and benchmark the implementation, ListAI creates an identical, independent **NumPy Engine** and proves their numerical parity down to floating-point tolerances of $10^{-12}$.

---

## Why Build Neural Networks with Python Lists?

Modern deep learning abstractions often hide essential numerical and linear algebra realities:
- How does matrix multiplication map to cache lines and memory addresses?
- Why does transposing weight matrices ($W^T$) enable error projection during backpropagation?
- How does subtracting the row-maximum prevent floating-point overflow in Softmax?
- Why do compiled C buffers like NumPy outperform interpreted pointer arrays?

By coding every vector dot product, matrix transposition, ReLU activation, categorical cross-entropy loss, and gradient backpropagation step using nested Python lists (`List[List[float]]`), ListAI provides complete educational transparency.

---

## Architecture Overview

ListAI employs a strict modular dual-engine design:

```
ListAI/
├── data/                      # Canonical dataset (150 samples)
├── docs/                      # In-depth technical documentation & derivations
├── listai/
│   ├── config.py              # Centralized dataclasses (Dataset, Model, Training)
│   ├── data/                  # Zero-leakage loader, stratified split, scaler
│   ├── engines/
│   │   ├── list_engine/       # Engine A: Pure Python Lists (AST-guarded 0 external deps)
│   │   └── numpy_engine/      # Engine B: Independent NumPy Engine
│   ├── evaluation/            # Scikit-learn-free metrics & table reports
│   └── utils/                 # Deterministic RNG & high-res timing
├── results/                   # Live benchmark results (benchmark_report.md)
├── scripts/                   # Dataset preparation & benchmark runner
├── tests/                     # 43+ Comprehensive unit test suites
└── main.py                    # Unified CLI
```

### Model Architecture
- **Input Layer:** 4 features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`)
- **Hidden Layer 1:** 16 neurons, **ReLU** activation
- **Hidden Layer 2:** 8 neurons, **ReLU** activation
- **Output Layer:** 3 neurons, **Softmax** activation
- **Initialization:** He normal initialization $\mathcal{N}(0, \sqrt{2 / \text{fan\_in}})$ and zero biases
- **Optimizer:** Mini-batch Stochastic Gradient Descent (SGD)

---

## Dataset & Preprocessing Pipeline

ListAI uses the official **UCI Iris Flower Dataset** (150 samples across 3 classes: *Setosa*, *Versicolor*, *Virginica*).

### Strict Data Hygiene (Zero Test Leakage):
1. **Stratified Splitting:** 70% Training (105 samples), 15% Validation (24 samples), 15% Testing (21 samples), preserving exact class balance.
2. **Standardization ($\mu = 0, \sigma = 1$):**
   $$\mu_j = \frac{1}{N} \sum_{i=1}^N X_{i,j}, \quad \sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^N (X_{i,j} - \mu_j)^2}$$
   Parameters $\mu$ and $\sigma$ are fitted **strictly on training data** and then applied to transform validation and test sets without leakage.

---

## Mathematical Formulations

### Forward Propagation
For layer $l \in \{1, 2, 3\}$:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]}), \quad A^{[2]} = \text{ReLU}(Z^{[2]}), \quad A^{[3]} = \text{Softmax}(Z^{[3]})$$

### Numerically Stable Softmax
$$\text{Softmax}(Z_{i,j}) = \frac{\exp(Z_{i,j} - \max_k Z_{i,k})}{\sum_k \exp(Z_{i,k} - \max_k Z_{i,k})}$$

### Categorical Cross-Entropy Loss
$$L(Y, \hat{Y}) = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{i,c} \ln(\text{clip}(\hat{Y}_{i,c}, \varepsilon, 1 - \varepsilon))$$

### Analytical Backpropagation
$$\frac{\partial L}{\partial Z^{[3]}} = \frac{1}{N} (A^{[3]} - Y)$$
$$\frac{\partial L}{\partial W^{[l]}} = (A^{[l-1]})^T \frac{\partial L}{\partial Z^{[l]}}, \quad \frac{\partial L}{\partial b^{[l]}} = \sum_{i=1}^N \left(\frac{\partial L}{\partial Z^{[l]}}\right)_{i, :}$$
$$\frac{\partial L}{\partial Z^{[l-1]}} = \left(\frac{\partial L}{\partial Z^{[l]}} (W^{[l]})^T\right) \odot \text{ReLU}'(Z^{[l-1]})$$

*(For complete proofs, see [docs/MATHEMATICS.md](docs/MATHEMATICS.md)).*

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.14 and 3.15)
- Standard library (`math`, `random`, `csv`, `time`, `tracemalloc`)
- NumPy (Required only for Engine B and cross-engine benchmarks)

### Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/devsamikhan/ListAi.git
cd ListAi

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Command-Line Interface (CLI)

Run training and evaluation with flexible flags:

```bash
# Train and evaluate Pure Python List Engine
.\.venv\Scripts\python.exe main.py --engine list --epochs 200

# Train and evaluate NumPy Engine
.\.venv\Scripts\python.exe main.py --engine numpy --epochs 200

# Train both engines and show side-by-side comparison
.\.venv\Scripts\python.exe main.py --engine both --epochs 200 --batch-size 16 --lr 0.05
```

### Sample Output:
```text
==================================================================
  Side-by-Side Comparison: Pure Python Lists vs NumPy
==================================================================
Metric                    Pure Python Lists         NumPy Engine
------------------------------------------------------------------
Training Time (s)                    5.2777s               0.2243s
Test Accuracy                       100.00%               95.24%
Test Loss                            0.0141                0.0994
Macro F1-Score                       1.0000                0.9521
Speedup Factor (NumPy)                23.53x faster
==================================================================
```

---

## Testing & Verification

ListAI includes **43 unit tests** covering every layer of the architecture:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Verification Highlights:
- **Numerical Gradient Checking:** Finite-difference approximations verify analytical backpropagation gradients with relative error $< 10^{-6}$.
- **Engine Parity:** Forward pass activations, backward pass gradients, and training trajectories match across engines within $10^{-12}$.
- **AST Isolation Guard:** Automated AST parsing guarantees that `listai/engines/list_engine/` contains zero imports of NumPy, SciPy, Pandas, or Scikit-learn.

---

## Empirical Benchmarks: Lists vs. NumPy

*Measured on Windows 11, Intel Core Processor, Python 3.14.4, NumPy 2.5.1 over repeated trials:*

| Metric | Pure Python List Engine | Independent NumPy Engine | Performance Factor |
|:---|:---:|:---:|:---:|
| **Training Time (100 epochs, median)** | **9.5460 s** | **0.2546 s** | **37.50x faster** |
| **Inference Latency (per sample)** | **46.05 $\mu$s** | **2.09 $\mu$s** | **22.08x faster** |
| **Peak Heap Memory (tracemalloc)** | **401.77 KB** | **134.92 KB** | **3.0x less memory** |
| **Test Set Accuracy** | **100.00%** | **100.00%** | Both $\ge 95\%$ |
| **Matrix Mul ($50 \times 50$)** | **11.811 ms** | **0.011 ms** | **1064.06x faster** |

*(For full benchmark telemetry and charts, see [results/benchmark_report.md](results/benchmark_report.md)).*

---

## Why NumPy Outperforms Python Lists

1. **Memory Contiguity:** Python lists are arrays of pointers to heap `PyObject` structs, requiring double dereferences and causing frequent L1/L2 cache misses. NumPy arrays use contiguous C-allocated memory buffers that stream sequentially through CPU cache lines.
2. **SIMD Vectorization:** NumPy calls compiled C/BLAS libraries utilizing CPU vector instructions (AVX-512, AVX2, FMA) to perform 4 to 8 operations per clock cycle. Python interprets operations scalar by scalar in bytecode.
3. **Dynamic Boxing Overhead:** In Python lists, every arithmetic operation dynamically checks types and allocates new float objects. NumPy operates with static homogenous C data types.

---

## Limitations & Future Improvements

- **Current Limitations:** Optimized specifically for small-to-medium tabular datasets (Iris); does not support GPU acceleration or automatic differentiation graphs.
- **Future Improvements:**
  - Implement momentum, RMSprop, and Adam optimizers in pure Python lists.
  - Add L1/L2 weight decay and Dropout regularization.
  - Implement convolutional and recurrent operations using pure Python nested lists.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
