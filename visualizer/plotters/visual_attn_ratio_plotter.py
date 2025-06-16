from ..core.interfaces import IPlotter
import numpy as np
from matplotlib import pyplot as plt


class VisualAttentionRatioPlotter(IPlotter):

    def plot(
        self,
        visual_attn_ratio: np.ndarray,
        figsize: tuple = (5, 5),
        title: str = "Visual Attention Ratio",
        cmap: str = "blues",
        show: bool = False,
        out_path: str = None,
        **kwargs,
    ) -> None:
        assert kwargs == {}, "No additional keyword arguments are expected."
        if visual_attn_ratio.ndim != 2:
            raise ValueError("Visual attention ratio must be a 2D array of shape (num_layers, num_attention_heads).")
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(visual_attn_ratio, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Attention Layer")
        if out_path is not None:
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        plt.close(fig)
