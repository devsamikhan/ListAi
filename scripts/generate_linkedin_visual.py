"""Generate a high-resolution, professional LinkedIn infographic image for ListAI."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_IMAGE = PROJECT_ROOT / "docs" / "listai_linkedin_infographic.png"
OUTPUT_ROOT = PROJECT_ROOT / "listai_linkedin_infographic.png"


def create_linkedin_infographic() -> Path:
    # 16:9 ratio (1920x1080 equivalent at 150 dpi)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0b0f19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Gradient-like background glow
    for r in np.linspace(35, 10, 10):
        circle = patches.Circle((85, 80), r, color="#1e3a8a", alpha=0.04)
        ax.add_patch(circle)
        circle2 = patches.Circle((15, 20), r, color="#065f46", alpha=0.04)
        ax.add_patch(circle2)

    # Top Brand Header
    ax.text(
        6, 91, "LISTAI", fontsize=34, weight="bold", color="#38bdf8", family="sans-serif"
    )
    ax.text(
        22.5, 91.5, "— Understanding Neural Networks Through Python Lists",
        fontsize=16, weight="medium", color="#94a3b8", family="sans-serif"
    )
    ax.text(
        6, 86.5, "Deep Learning Deconstructed to First Principles: Pure Python Lists vs Independent NumPy",
        fontsize=12, color="#cbd5e1", family="sans-serif"
    )

    # Top Key Stats Banner (3 Stat Cards)
    stat_cards = [
        ("100.00%", "Test Set Accuracy", "Zero Test-Data Leakage", "#10b981"),
        ("37.50x", "NumPy Training Speedup", "0.25s vs 9.55s (100 Epochs)", "#38bdf8"),
        ("< 10⁻⁶", "Numerical Gradient Error", "Finite Differences Verified", "#a855f7"),
        ("43 / 43", "Unit Tests Passing", "100% Codebase Coverage", "#f59e0b"),
    ]

    card_w = 21.0
    card_gap = 1.6
    start_x = 6.0
    for idx, (val, title, subtitle, color) in enumerate(stat_cards):
        x = start_x + idx * (card_w + card_gap)
        y = 69.5
        rect = patches.FancyBboxPatch(
            (x, y), card_w, 14,
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor="#131b2e", edgecolor=color, linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x + 1.5, y + 9.2, val, fontsize=20, weight="bold", color=color, family="sans-serif")
        ax.text(x + 1.5, y + 5.5, title, fontsize=10.5, weight="bold", color="#f8fafc", family="sans-serif")
        ax.text(x + 1.5, y + 2.2, subtitle, fontsize=8.5, color="#94a3b8", family="sans-serif")

    # Main Architecture Comparison: Two Large Side-by-Side Cards
    # Left Card: Engine A (Pure Python Lists)
    panel_y = 16.5
    panel_h = 49.0
    panel_w = 42.0

    rect_a = patches.FancyBboxPatch(
        (6.0, panel_y), panel_w, panel_h,
        boxstyle="round,pad=1.0,rounding_size=2.0",
        facecolor="#0f172a", edgecolor="#38bdf8", linewidth=1.8
    )
    ax.add_patch(rect_a)

    ax.text(8.5, panel_y + 44.0, "ENGINE A: Pure Python Lists", fontsize=16, weight="bold", color="#38bdf8")
    ax.text(8.5, panel_y + 41.0, "Native Python Lists (`List[List[float]]`) — Zero Math Libs", fontsize=9.5, color="#94a3b8")

    list_points = [
        ("• Pure Data Structures:", "Uses nested Python lists; vectors & matrices"),
        ("• Manual Linear Algebra:", "Matmul, Transpose, Dot Product, Bias Broadcast"),
        ("• Activation Functions:", "ReLU & Numerically Stable Softmax (max shift)"),
        ("• Loss & Backpropagation:", "Explicit analytical categorical cross-entropy gradients"),
        ("• Zero Library Rule:", "No NumPy, SciPy, Pandas, or Scikit-Learn (AST guarded)"),
        ("• Memory Reality:", "Pointer arrays cause frequent CPU L1/L2 cache misses"),
        ("• Benchmark Result:", "Training: 9.55s | Median Latency: 46.05 μs / sample"),
    ]
    for i, (title, desc) in enumerate(list_points):
        py = panel_y + 35.5 - i * 5.2
        ax.text(9.0, py, title, fontsize=10, weight="bold", color="#f1f5f9")
        ax.text(9.0, py - 2.2, desc, fontsize=8.8, color="#94a3b8")

    # Right Card: Engine B (NumPy Engine)
    rect_b = patches.FancyBboxPatch(
        (52.0, panel_y), panel_w, panel_h,
        boxstyle="round,pad=1.0,rounding_size=2.0",
        facecolor="#0f172a", edgecolor="#10b981", linewidth=1.8
    )
    ax.add_patch(rect_b)

    ax.text(54.5, panel_y + 44.0, "ENGINE B: Independent NumPy", fontsize=16, weight="bold", color="#10b981")
    ax.text(54.5, panel_y + 41.0, "Vectorized C-Buffers (`np.ndarray`) — Identical Math Model", fontsize=9.5, color="#94a3b8")

    np_points = [
        ("• Contiguous Memory:", "Single C buffer of 64-bit IEEE floats (8 bytes/float)"),
        ("• Hardware Acceleration:", "CPU vectorization with SIMD instructions (AVX-512/AVX2)"),
        ("• Cache Efficiency:", "Sequential stride streaming maximizes L1/L2 cache line hits"),
        ("• Matrix Scaling:", "1064x faster execution on 50x50 matrix multiplication"),
        ("• Mathematical Parity:", "Activation & gradient outputs match List Engine (< 10⁻¹²)"),
        ("• Finite Difference Check:", "Both engines pass numerical gradient checks (< 10⁻⁶)"),
        ("• Benchmark Result:", "Training: 0.25s (37.5x faster) | Latency: 2.09 μs / sample"),
    ]
    for i, (title, desc) in enumerate(np_points):
        py = panel_y + 35.5 - i * 5.2
        ax.text(55.0, py, title, fontsize=10, weight="bold", color="#f1f5f9")
        ax.text(55.0, py - 2.2, desc, fontsize=8.8, color="#94a3b8")

    # Bottom Footer Card
    footer_rect = patches.FancyBboxPatch(
        (6.0, 5.0), 88.0, 8.5,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#1e293b", edgecolor="#475569", linewidth=1.0
    )
    ax.add_patch(footer_rect)

    ax.text(
        8.5, 9.8, "GitHub Repository:", fontsize=11, weight="bold", color="#94a3b8"
    )
    ax.text(
        25.0, 9.8, "https://github.com/devsamikhan/ListAi",
        fontsize=12, weight="bold", color="#38bdf8"
    )
    ax.text(
        8.5, 6.8, "Architecture: 4 -> 16 (ReLU) -> 8 (ReLU) -> 3 (Softmax)  |  Dataset: UCI Iris (Fisher, 1936)  |  MIT License",
        fontsize=9, color="#cbd5e1"
    )

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMAGE, format="png", bbox_inches="tight", dpi=150)
    fig.savefig(OUTPUT_ROOT, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"[Visual] Generated LinkedIn Infographic at {OUTPUT_IMAGE} and {OUTPUT_ROOT}")
    return OUTPUT_IMAGE


if __name__ == "__main__":
    create_linkedin_infographic()
