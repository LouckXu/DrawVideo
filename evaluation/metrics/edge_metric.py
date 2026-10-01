# metrics/edge_metric.py

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from metrics.base_metric import BaseMetric


ImageInput = Union[str, Image.Image, np.ndarray, torch.Tensor]


class EdgeOverlapMetric(BaseMetric):
    """
    Edge-based structural similarity metric.

    This metric:
    1. Converts images to grayscale
    2. Extracts edges using Canny
    3. Computes edge overlap statistics:
       - precision
       - recall
       - f1

    Main intended use:
        sketch image vs generated image
    """

    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
        resize: Optional[tuple[int, int]] = (224, 224),
        dilation_kernel_size: int = 3,
        name: str = "edge_f1",
    ) -> None:
        """
        Args:
            low_threshold: Lower threshold for Canny.
            high_threshold: Higher threshold for Canny.
            resize: Optional target size (H, W). If None, no resizing.
            dilation_kernel_size: Kernel size for edge dilation before overlap matching.
            name: Main metric name. Usually 'edge_f1'.
        """
        super().__init__(name=name, higher_is_better=True)

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.resize = resize
        self.dilation_kernel_size = dilation_kernel_size

    def _load_image_from_path(self, path: str) -> Image.Image:
        """Load an image from path."""
        return Image.open(path).convert("RGB")

    def _to_numpy_rgb(self, image: ImageInput) -> np.ndarray:
        """
        Convert input image to uint8 numpy array of shape [H, W, 3].
        """
        if isinstance(image, str):
            image = self._load_image_from_path(image)

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"), dtype=np.uint8)

        if isinstance(image, np.ndarray):
            array = image

            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)

            if array.ndim != 3:
                raise ValueError(
                    f"NumPy image must have shape [H, W, C] or [H, W], got {array.shape}."
                )

            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            elif array.shape[-1] != 3:
                raise ValueError(
                    f"NumPy image channel dimension must be 1 or 3, got {array.shape[-1]}."
                )

            if array.dtype != np.uint8:
                if array.max() <= 1.0:
                    array = (array * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    array = array.clip(0, 255).astype(np.uint8)

            return array

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu().float()

            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError(
                        f"Only batch size 1 is supported, got tensor shape {tuple(tensor.shape)}."
                    )
                tensor = tensor[0]

            if tensor.ndim != 3:
                raise ValueError(
                    f"Tensor image must have shape [C,H,W] or [H,W,C], got {tuple(tensor.shape)}."
                )

            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            elif tensor.shape[-1] not in (1, 3):
                raise ValueError(
                    f"Tensor channel dimension must be 1 or 3, got shape {tuple(tensor.shape)}."
                )

            array = tensor.numpy()

            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)

            if array.max() <= 1.0:
                array = (array * 255.0).clip(0, 255).astype(np.uint8)
            else:
                array = array.clip(0, 255).astype(np.uint8)

            return array

        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            f"Expected str, PIL.Image, np.ndarray, or torch.Tensor."
        )

    def _preprocess(self, image: ImageInput) -> np.ndarray:
        """
        Convert input to uint8 RGB numpy array and optionally resize.
        """
        array = self._to_numpy_rgb(image)

        if self.resize is not None:
            h, w = self.resize
            array = cv2.resize(array, (w, h), interpolation=cv2.INTER_LINEAR)

        return array

    def _extract_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Extract binary edge map using Canny.
        Returns:
            edge map with values in {0, 1}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        edges = (edges > 0).astype(np.uint8)
        return edges

    def _dilate_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Dilate edges slightly to allow tolerant matching.
        """
        if self.dilation_kernel_size <= 1:
            return edges

        kernel = np.ones(
            (self.dilation_kernel_size, self.dilation_kernel_size),
            dtype=np.uint8,
        )
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return (dilated > 0).astype(np.uint8)

    def _compute_overlap_scores(
        self,
        pred_edges: np.ndarray,
        gt_edges: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute edge precision, recall, and F1.
        """
        pred_dilated = self._dilate_edges(pred_edges)
        gt_dilated = self._dilate_edges(gt_edges)

        # Precision: among predicted edges, how many match GT edges
        pred_positive = pred_edges.sum()
        pred_true_positive = np.logical_and(pred_edges == 1, gt_dilated == 1).sum()

        # Recall: among GT edges, how many are recovered by predicted edges
        gt_positive = gt_edges.sum()
        gt_true_positive = np.logical_and(gt_edges == 1, pred_dilated == 1).sum()

        precision = (
            float(pred_true_positive) / float(pred_positive)
            if pred_positive > 0
            else 0.0
        )
        recall = (
            float(gt_true_positive) / float(gt_positive)
            if gt_positive > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
        }

    def compute(self, **kwargs: Any) -> Dict[str, float]:
        """
        Compute edge overlap metrics between two images.

        Required kwargs:
            pred_image or pred_path
            gt_image or gt_path

        Returns:
            {
                "edge_precision": ...,
                "edge_recall": ...,
                "edge_f1": ...
            }
        """
        pred = kwargs.get("pred_image", kwargs.get("pred_path"))
        gt = kwargs.get("gt_image", kwargs.get("gt_path"))

        if pred is None:
            raise ValueError("Missing prediction input. Use 'pred_image' or 'pred_path'.")
        if gt is None:
            raise ValueError("Missing reference input. Use 'gt_image' or 'gt_path'.")

        pred_image = self._preprocess(pred)
        gt_image = self._preprocess(gt)

        pred_edges = self._extract_edges(pred_image)
        gt_edges = self._extract_edges(gt_image)

        scores = self._compute_overlap_scores(pred_edges, gt_edges)
        return scores