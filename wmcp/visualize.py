"""Protocol visualization — publication-ready figures."""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_message_heatmap(tokens: np.ndarray, vocab_size: int = 3,
                         title: str = "Message Frequency",
                         output_path: Optional[str] = None):
    """Generate symbol frequency heatmap per position.

    Args:
        tokens: (N, n_positions) integer token array.
        vocab_size: K.
        title: Plot title.
        output_path: Optional PNG save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pos = tokens.shape[1]
    freq = np.zeros((n_pos, vocab_size))
    for p in range(n_pos):
        counts = np.bincount(tokens[:, p], minlength=vocab_size)
        freq[p] = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(max(4, vocab_size * 1.5), max(3, n_pos * 0.8)))
    im = ax.imshow(freq, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(vocab_size))
    ax.set_xticklabels([f"sym {s}" for s in range(vocab_size)])
    ax.set_yticks(range(n_pos))
    ax.set_yticklabels([f"pos {p}" for p in range(n_pos)])
    for i in range(n_pos):
        for j in range(vocab_size):
            ax.text(j, i, f'{freq[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Frequency")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return freq


def plot_onboarding_curve(curve: List[Dict], base_accuracy: float = 0.83,
                          title: str = "Onboarding Convergence",
                          output_path: Optional[str] = None):
    """Plot accuracy over onboarding steps.

    Args:
        curve: List of {"step": int, "accuracy": float}.
        base_accuracy: Base protocol accuracy (for 90% line).
        title: Plot title.
        output_path: Optional PNG save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [c["step"] for c in curve]
    accs = [c["accuracy"] * 100 for c in curve]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, accs, 'ro-', linewidth=2, markersize=6)
    ax.axhline(y=base_accuracy * 90, color='green', linestyle='--',
               label=f'90% of base ({base_accuracy * 90:.1f}%)')
    ax.axhline(y=base_accuracy * 100, color='blue', linestyle=':',
               alpha=0.5, label=f'Base ({base_accuracy * 100:.1f}%)')
    ax.set_xlabel("Fine-tuning Steps", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_scatter(predictions: np.ndarray, ground_truth: np.ndarray,
                            title: str = "Predicted vs Actual",
                            output_path: Optional[str] = None):
    """Scatter plot of predicted vs actual property values.

    Args:
        predictions: (N,) predicted values.
        ground_truth: (N,) actual values.
        title: Plot title.
        output_path: Optional PNG save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ground_truth, predictions, s=20, alpha=0.5, color='steelblue')
    lims = [min(ground_truth.min(), predictions.min()),
            max(ground_truth.max(), predictions.max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mi_heatmap(mi_matrix: np.ndarray,
                    position_labels: Optional[List[str]] = None,
                    attribute_labels: Optional[List[str]] = None,
                    title: str = "MI(position, attribute)",
                    output_path: Optional[str] = None):
    """Plot mutual information heatmap.

    Args:
        mi_matrix: (n_positions, n_attributes) MI values.
        position_labels: Labels for positions.
        attribute_labels: Labels for attributes.
        title: Plot title.
        output_path: Optional PNG save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mi = np.array(mi_matrix)
    n_pos, n_attr = mi.shape

    if position_labels is None:
        position_labels = [f"pos {p}" for p in range(n_pos)]
    if attribute_labels is None:
        attribute_labels = [f"attr {a}" for a in range(n_attr)]

    fig, ax = plt.subplots(figsize=(max(4, n_attr * 2), max(3, n_pos * 0.8)))
    im = ax.imshow(mi, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_attr))
    ax.set_xticklabels(attribute_labels, fontsize=10)
    ax.set_yticks(range(n_pos))
    ax.set_yticklabels(position_labels, fontsize=10)
    for i in range(n_pos):
        for j in range(n_attr):
            ax.text(j, i, f'{mi[i, j]:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="MI (nats)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
