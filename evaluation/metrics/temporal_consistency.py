# metrics/temporal_consistency.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from metrics.base_metric import BaseMetric
from metrics.clip_metric import CLIPImageSimilarityMetric
from metrics.lpips_metric import LPIPSMetric


ImageInput = Union[str, Any]


class TemporalCLIPConsistency(BaseMetric):
    """
    Temporal consistency metric based on CLIP image-image similarity.

    Supports two modes:
    1. anchor:
        Compare each frame with one anchor image.
    2. adjacent:
        Compare each frame with the next frame.

    Higher is better.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        mode: str = "anchor",
        name: str = "temporal_clip",
    ) -> None:
        super().__init__(name=name, higher_is_better=True)

        if mode not in {"anchor", "adjacent"}:
            raise ValueError(f"mode must be 'anchor' or 'adjacent', got '{mode}'.")

        self.mode = mode
        self.metric = CLIPImageSimilarityMetric(
            model_name=model_name,
            device=device,
            resize=resize,
            name="clip_image_similarity_internal",
        )

    def _validate_frames(self, frames: Sequence[ImageInput]) -> None:
        if not isinstance(frames, Sequence) or len(frames) < 2:
            raise ValueError("`frames` must be a sequence with at least 2 frames.")

    def compute(self, **kwargs: Any) -> Dict[str, float]:
        """
        Required kwargs:
            frames: Sequence[ImageInput]

        Optional kwargs:
            anchor_image / anchor_path: used in anchor mode
                - if not provided, frames[0] will be used as anchor

        Returns:
            {
                "temporal_clip": float,
                "temporal_clip_num_pairs": float
            }
        """
        frames = kwargs.get("frames")
        if frames is None:
            raise ValueError("Missing `frames` in kwargs.")

        self._validate_frames(frames)

        scores: List[float] = []

        if self.mode == "anchor":
            anchor = kwargs.get("anchor_image", kwargs.get("anchor_path", frames[0]))

            for frame in frames:
                result = self.metric(
                    pred_image=frame,
                    gt_image=anchor,
                )
                scores.append(result["clip_image_similarity_internal"])

        else:  # adjacent
            for i in range(len(frames) - 1):
                result = self.metric(
                    pred_image=frames[i + 1],
                    gt_image=frames[i],
                )
                scores.append(result["clip_image_similarity_internal"])

        if not scores:
            mean_score = 0.0
        else:
            mean_score = sum(scores) / len(scores)

        return {
            self.name: float(mean_score),
            "temporal_clip_num_pairs": float(len(scores)),
        }


class TemporalLPIPSConsistency(BaseMetric):
    """
    Temporal consistency metric based on LPIPS.

    Supports two modes:
    1. anchor:
        Compare each frame with one anchor image.
    2. adjacent:
        Compare each frame with the next frame.

    Lower is better.
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = (224, 224),
        mode: str = "anchor",
        name: str = "temporal_lpips",
    ) -> None:
        super().__init__(name=name, higher_is_better=False)

        if mode not in {"anchor", "adjacent"}:
            raise ValueError(f"mode must be 'anchor' or 'adjacent', got '{mode}'.")

        self.mode = mode
        self.metric = LPIPSMetric(
            net=net,
            device=device,
            resize=resize,
            name="lpips_internal",
        )

    def _validate_frames(self, frames: Sequence[ImageInput]) -> None:
        if not isinstance(frames, Sequence) or len(frames) < 2:
            raise ValueError("`frames` must be a sequence with at least 2 frames.")

    def compute(self, **kwargs: Any) -> Dict[str, float]:
        """
        Required kwargs:
            frames: Sequence[ImageInput]

        Optional kwargs:
            anchor_image / anchor_path: used in anchor mode
                - if not provided, frames[0] will be used as anchor

        Returns:
            {
                "temporal_lpips": float,
                "temporal_lpips_num_pairs": float
            }
        """
        frames = kwargs.get("frames")
        if frames is None:
            raise ValueError("Missing `frames` in kwargs.")

        self._validate_frames(frames)

        scores: List[float] = []

        if self.mode == "anchor":
            anchor = kwargs.get("anchor_image", kwargs.get("anchor_path", frames[0]))

            for frame in frames:
                result = self.metric(
                    pred_image=frame,
                    gt_image=anchor,
                )
                scores.append(result["lpips_internal"])

        else:  # adjacent
            for i in range(len(frames) - 1):
                result = self.metric(
                    pred_image=frames[i + 1],
                    gt_image=frames[i],
                )
                scores.append(result["lpips_internal"])

        if not scores:
            mean_score = 0.0
        else:
            mean_score = sum(scores) / len(scores)

        return {
            self.name: float(mean_score),
            "temporal_lpips_num_pairs": float(len(scores)),
        }