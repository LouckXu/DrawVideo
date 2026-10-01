# metrics/clip_metric.py

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import CLIPModel, CLIPProcessor

from metrics.base_metric import BaseMetric


ImageInput = Union[str, Image.Image, np.ndarray, torch.Tensor]


class CLIPImageSimilarityMetric(BaseMetric):
    """
    CLIP-based image-image similarity metric.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        name: str = "clip_image_similarity",
    ) -> None:
        super().__init__(name=name, higher_is_better=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.resize = resize

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _load_image_from_path(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _to_pil(self, image: ImageInput) -> Image.Image:
        if isinstance(image, str):
            return self._load_image_from_path(image)

        if isinstance(image, Image.Image):
            return image.convert("RGB")

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

            return Image.fromarray(array).convert("RGB")

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

            return Image.fromarray(array).convert("RGB")

        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            f"Expected str, PIL.Image, np.ndarray, or torch.Tensor."
        )

    def _preprocess_image(self, image: ImageInput) -> Image.Image:
        pil_image = self._to_pil(image)

        if self.resize is not None:
            pil_image = pil_image.resize((self.resize[1], self.resize[0]))

        return pil_image

    @torch.no_grad()
    def _encode_image(self, image: ImageInput) -> torch.Tensor:
        pil_image = self._preprocess_image(image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)

        return image_features

    @torch.no_grad()
    def compute(self, **kwargs: Any) -> Dict[str, float]:
        pred = kwargs.get("pred_image", kwargs.get("pred_path"))
        gt = kwargs.get("gt_image", kwargs.get("gt_path"))

        if pred is None:
            raise ValueError("Missing prediction input. Use 'pred_image' or 'pred_path'.")
        if gt is None:
            raise ValueError("Missing reference input. Use 'gt_image' or 'gt_path'.")

        pred_features = self._encode_image(pred)
        gt_features = self._encode_image(gt)

        similarity = F.cosine_similarity(pred_features, gt_features).item()

        return {self.name: float(similarity)}