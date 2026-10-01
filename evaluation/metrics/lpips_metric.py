# metrics/lpips_metric.py

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import lpips

from metrics.base_metric import BaseMetric


ImageInput = Union[str, Image.Image, np.ndarray, torch.Tensor]


class LPIPSMetric(BaseMetric):
    """
    LPIPS metric for perceptual similarity between two images.

    Supported input types:
    - image file path: str
    - PIL.Image
    - numpy.ndarray
    - torch.Tensor

    Expected behavior:
    - Convert input to torch.Tensor with shape [1, 3, H, W]
    - Normalize to [-1, 1] as required by LPIPS
    - Optionally resize both inputs to a fixed resolution
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = (224, 224),
        name: str = "lpips",
    ) -> None:
        """
        Args:
            net: Backbone used by LPIPS. Common choices: 'alex', 'vgg', 'squeeze'.
            device: 'cuda', 'cpu', or None for auto selection.
            resize: Target size (H, W). If None, no resizing is applied.
            name: Metric name used in result dict.
        """
        super().__init__(name=name, higher_is_better=False)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.resize = resize
        self.net = net

        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def _load_image_from_path(self, path: str) -> Image.Image:
        """Load an image from file path and convert to RGB."""
        return Image.open(path).convert("RGB")

    def _to_tensor(self, image: ImageInput) -> torch.Tensor:
        """
        Convert input image to tensor of shape [1, 3, H, W] in range [0, 1].
        """
        if isinstance(image, str):
            image = self._load_image_from_path(image)

        if isinstance(image, Image.Image):
            array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            return tensor

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

            if array.dtype != np.float32:
                array = array.astype(np.float32)

            # If values are likely in [0, 255], scale to [0, 1]
            if array.max() > 1.0:
                array = array / 255.0

            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            return tensor

        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone().float()

            if tensor.ndim == 3:
                # [C, H, W] or [H, W, C]
                if tensor.shape[0] in (1, 3):
                    tensor = tensor.unsqueeze(0)
                elif tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                else:
                    raise ValueError(
                        f"3D tensor must be [C,H,W] or [H,W,C], got {tuple(tensor.shape)}."
                    )

            elif tensor.ndim == 4:
                # [B, C, H, W] expected, only allow batch size 1
                if tensor.shape[0] != 1:
                    raise ValueError(
                        f"Only batch size 1 is supported, got tensor shape {tuple(tensor.shape)}."
                    )
                if tensor.shape[1] not in (1, 3):
                    raise ValueError(
                        f"Tensor channel dimension must be 1 or 3, got {tensor.shape[1]}."
                    )
            else:
                raise ValueError(
                    f"Tensor image must have 3 or 4 dims, got {tensor.ndim} dims."
                )

            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)

            if tensor.max() > 1.0:
                tensor = tensor / 255.0

            return tensor

        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            f"Expected str, PIL.Image, np.ndarray, or torch.Tensor."
        )

    def _preprocess(self, image: ImageInput) -> torch.Tensor:
        """
        Convert input to tensor [1, 3, H, W], optionally resize, and normalize to [-1, 1].
        """
        tensor = self._to_tensor(image)

        if self.resize is not None:
            tensor = F.interpolate(
                tensor,
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            )

        # LPIPS expects input in [-1, 1]
        tensor = tensor.clamp(0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.to(self.device)

        return tensor

    @torch.no_grad()
    def compute(self, **kwargs: Any) -> Dict[str, float]:
        """
        Compute LPIPS between two images.

        Required kwargs:
            pred_image or pred_path
            gt_image or gt_path

        Returns:
            {"lpips": float}
        """
        pred = kwargs.get("pred_image", kwargs.get("pred_path"))
        gt = kwargs.get("gt_image", kwargs.get("gt_path"))

        if pred is None:
            raise ValueError("Missing prediction input. Use 'pred_image' or 'pred_path'.")
        if gt is None:
            raise ValueError("Missing ground truth input. Use 'gt_image' or 'gt_path'.")

        pred_tensor = self._preprocess(pred)
        gt_tensor = self._preprocess(gt)

        score = self.model(pred_tensor, gt_tensor)
        value = float(score.item())

        return {self.name: value}