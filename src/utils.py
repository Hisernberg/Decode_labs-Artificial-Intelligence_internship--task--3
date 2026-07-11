"""Utility helpers for reproducibility and plot styling."""
from __future__ import annotations

import os
import random
import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_plots() -> None:
    """Apply the academic-clean seaborn style chosen for this project."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns

    # Register Noto Sans SC for any CJK characters; DejaVu Sans catches symbols
    for fp in [
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
            except Exception:
                pass

    sns.set_theme(
        style="whitegrid",
        palette="muted",
        rc={
            "figure.figsize": (10, 6),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.sans-serif": ["Noto Sans SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        },
    )
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, path: str, mkdir: bool = True) -> None:
    """Save a matplotlib figure to disk."""
    if mkdir:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {path}")


def print_system_info() -> dict:
    """Print and return runtime info (Python, NumPy, torch, GPU)."""
    import platform
    info = {"python": platform.python_version()}
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "not installed"
    return info


if __name__ == "__main__":
    set_seeds(42)
    print(print_system_info())
