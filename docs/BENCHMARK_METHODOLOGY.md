# ListAI — Benchmark Methodology & Protocol

**Repository:** [https://github.com/devsamikhan/ListAi](https://github.com/devsamikhan/ListAi)

---

## 1. Benchmarking Principles

Benchmarking software across interpreted languages and native compiled C extensions requires strict experimental controls. The ListAI benchmarking suite (`scripts/run_benchmarks.py`) was constructed following these core scientific principles:

1. **Empirical Measurement Only:**
   No performance numbers are ever estimated or hardcoded. All metrics are measured live during execution and formatted directly into `results/benchmark_report.md`.

2. **Repetition and Outlier Resilience:**
   Single-run benchmarks are vulnerable to operating system context switches, cache warmth variations, and garbage collection pauses. ListAI performs multiple repeated trials and reports **median** values along with minimum, maximum, and mean metrics.

3. **High-Resolution Timing:**
   Timings are captured using `time.perf_counter()`, providing monotonic clock measurements with sub-microsecond precision on modern platforms.

4. **Reliable Heap Memory Tracking:**
   Memory allocation is measured using Python's standard library `tracemalloc`, isolating peak heap memory consumed during training without counting external OS page buffers.

5. **Complete System Telemetry:**
   Every report records OS distribution, CPU processor family, Python compiler version, and NumPy library version to ensure reproducibility across different machines.

---

## 2. Benchmark Suites

### Suite 1: Full Model Training (5 Repeated Trials)
- **Task:** Train the $4 \rightarrow 16 \rightarrow 8 \rightarrow 3$ architecture for 100 epochs on the 105-sample standardized Iris training split using mini-batch SGD (batch size 16, learning rate 0.05).
- **Metric Captured:** Elapsed wall-clock seconds and peak heap allocation (`tracemalloc`).
- **Trial Seeds:** Evaluated across distinct random seeds ($10, 20, 30, 40, 50$) to ensure fairness across weight initialization trajectories.

### Suite 2: Inference Latency (100 Iterations)
- **Task:** Perform forward pass and prediction on the 21-sample test set for 100 consecutive iterations.
- **Metric Captured:** Median latency per sample in microseconds ($\mu\text{s}$).

### Suite 3: Matrix Multiplication Scaling
- **Task:** Multiply square matrices of dimensions $10 \times 10$, $50 \times 50$, and $100 \times 100$ for 10 iterations each.
- **Metric Captured:** Wall-clock time comparison between pure Python `matrix_mul` and NumPy `@` (BLAS).

---

## 3. Running Benchmarks Locally

To execute the benchmark suite and update `results/benchmark_report.md`:

```bash
# Activate virtual environment
.\.venv\Scripts\python.exe scripts/run_benchmarks.py
```
