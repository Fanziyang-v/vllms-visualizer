from typing import Tuple

from ..core.interfaces import IPlotter

import numpy as np
from matplotlib import pyplot as plt


class FrameWeightsPlotter(IPlotter):

    def plot(
        self,
        frame_weights: np.ndarray,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Frame Weights",
        xlabel: str = "Frame Index",
        ylabel: str = "Frame Weight",
        color: str = "b",
        marker: str = "o",
        linestyle: str = "-",
        show: bool = False,
        out_path: str = None,
        **kwargs,
    ) -> None:
        assert kwargs == {}, "No additional keyword arguments are expected."
        if frame_weights.ndim != 1:
            raise ValueError("Frame weights must be a 1D array.")
        fig, ax = plt.subplots(figsize=figsize)
        frame_indices = np.arange(len(frame_weights))
        ax.plot(frame_indices, frame_weights, marker=marker, linestyle=linestyle, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if out_path is not None:
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
        plt.close(fig)
