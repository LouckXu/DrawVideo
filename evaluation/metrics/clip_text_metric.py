# metrics/clip_text_metric.py

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from metrics.base_metric import BaseMetric


ImageInput = Union[str, Image.Image, np.ndarray, torch.Tensor]


class CLIPTextImageSimilarityMetric(BaseMetric):
    """
    CLIP-based text-image similarity metric.

    Supported image input types:
    - image file path: str
    - PIL.Image
    - numpy.ndarray
    - torch.Tensor

    Text input:
    - str

    Output:
        {"clip_text_image_similarity": float}

    Notes:
    - Similarity is cosine similarity between CLIP text embedding and image embedding.
    - Higher is better.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        name: str = "clip_text_image_similarity",
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
    def _encode_text(self, text: str) -> torch.Tensor:
        if not isinstance(text, str):
            raise TypeError(f"Text input must be str, got {type(text).__name__}.")
        if not text.strip():
            raise ValueError("Text input must be a non-empty string.")

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        text_features = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @torch.no_grad()
    def compute(self, **kwargs: Any) -> Dict[str, float]:
        """
        Compute CLIP cosine similarity between one text and one image.

        Required kwargs:
            text: str
            image / image_path / pred_image / pred_path

        Returns:
            {"clip_text_image_similarity": float}
        """
        text = kwargs.get("text")
        image = (
            kwargs.get("image")
            or kwargs.get("image_path")
            or kwargs.get("pred_image")
            or kwargs.get("pred_path")
        )

        if text is None:
            raise ValueError("Missing `text` in kwargs.")
        if image is None:
            raise ValueError(
                "Missing image input. Use 'image', 'image_path', 'pred_image', or 'pred_path'."
            )

        text_features = self._encode_text(text)
        image_features = self._encode_image(image)

        similarity = F.cosine_similarity(text_features, image_features).item()
        return {self.name: float(similarity)}