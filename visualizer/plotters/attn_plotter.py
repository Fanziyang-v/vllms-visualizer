from typing import Optional, Tuple

from .base_plotter import BasePlotter

import numpy as np
from matplotlib import pyplot as plt


class AttentionPlotter(BasePlotter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(
        self,
        image: np.ndarray,
        attn_map: np.ndarray,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = "viridis",
        overlay_alpha: float = 0.5,
        colorbar: bool = False,
        title: Optional[str] = None,
        show: bool = False,
        out_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot an attention map over an image.

        Args:
            image (np.ndarray): An image of shape (H, W, C) where H is height, W is width, and C is the number of channels.
            attn_map (np.ndarray): Visual attention map of shape (H, W).
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 10).
            cmap (str, optional): Colormap of attention map. Defaults to "viridis".
            overlay_alpha (float, optional): Transparency. Defaults to 0.5.
            colorbar (bool, optional): Whether to show colorbar. Defaults to False.
            title (Optional[str], optional): Visualization title. Defaults to None.
            show (bool, optional): Whether to show. Defaults to False.
            out_path (Optional[str], optional): Output path for saving the visualization result. Defaults to None.

        Raises:
            ValueError: If the image and attention map do not have the same spatial dimensions.
        """
        assert kwargs == {}, "No additional keyword arguments are expected."
        if image.shape[:2] != attn_map.shape:
            raise ValueError("Image and attention map must have the same spatial dimensions.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        heatmap = ax.imshow(attn_map, cmap=cmap, alpha=overlay_alpha)

        ax.axis("off")
        if title is not None:
            ax.set_title(title)
        if colorbar:
            cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attention", rotation=270, labelpad=20)
        if out_path is not None:
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        plt.close(fig)
