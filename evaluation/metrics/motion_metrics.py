# metrics/motion_metrics.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from metrics.base_metric import BaseMetric
from metrics.clip_metric import CLIPImageSimilarityMetric
from metrics.clip_text_metric import CLIPTextImageSimilarityMetric


class DynamicProgressionMetric(BaseMetric):
    """
    DEVIL-inspired dynamic progression metric.

    Main idea:
    - Flatten all sampled frames from all shots in temporal order
    - Measure adjacent-frame semantic change using CLIP image-image similarity
    - Convert similarity to change score: change = 1 - similarity
    - Also measure long-range change between first and last frame
    - Estimate dynamic coverage: how many adjacent pairs show sufficient change
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        change_threshold: float = 0.08,
        name: str = "dynamic_progression_score",
    ) -> None:
        super().__init__(name=name, higher_is_better=True)

        self.change_threshold = change_threshold
        self.image_metric = CLIPImageSimilarityMetric(
            model_name=model_name,
            device=device,
            resize=resize,
            name="clip_image_similarity_internal",
        )

    def _flatten_frames(self, shots: List[Dict[str, Any]]) -> List[Any]:
        frames: List[Any] = []
        for shot in shots:
            shot_frames = shot.get("sampled_frames", [])
            frames.extend(shot_frames)
        return frames

    def compute(self, **kwargs: Any) -> Dict[str, float]:
        shots = kwargs.get("shots")
        if shots is None:
            raise ValueError("Missing `shots` in kwargs.")
        if not isinstance(shots, list) or len(shots) == 0:
            raise ValueError("`shots` must be a non-empty list.")

        frames = self._flatten_frames(shots)
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames across all shots to compute dynamics.")

        adjacent_changes: List[float] = []

        for i in range(len(frames) - 1):
            sim_result = self.image_metric.compute(
                pred_image=frames[i + 1],
                gt_image=frames[i],
            )
            sim = sim_result["clip_image_similarity_internal"]
            change = max(0.0, 1.0 - sim)
            adjacent_changes.append(float(change))

        mean_adjacent_change = sum(adjacent_changes) / len(adjacent_changes)

        first_last_result = self.image_metric.compute(
            pred_image=frames[-1],
            gt_image=frames[0],
        )
        first_last_sim = first_last_result["clip_image_similarity_internal"]
        long_range_change = max(0.0, 1.0 - first_last_sim)

        dynamic_change_coverage = (
            sum(1 for c in adjacent_changes if c >= self.change_threshold) / len(adjacent_changes)
        )

        progression_score = (
            0.4 * mean_adjacent_change
            + 0.3 * long_range_change
            + 0.3 * dynamic_change_coverage
        )

        return {
            self.name: float(progression_score),
            "dynamic_mean_adjacent_change": float(mean_adjacent_change),
            "dynamic_long_range_change": float(long_range_change),
            "dynamic_change_coverage": float(dynamic_change_coverage),
            "dynamic_num_pairs": float(len(adjacent_changes)),
        }


class DynamicControllabilityMetric(BaseMetric):
    """
    DEVIL-inspired dynamic controllability metric.

    Main idea:
    - Each event is treated as a target semantic change stage
    - For each event, find the best matching shot using text-image similarity
    - If best-match shots progress in temporal order and event scores are sufficiently high,
      the dynamics are considered more controllable
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        event_threshold: float = 0.25,
        frame_aggregation: str = "max",
        name: str = "dynamic_controllability_score",
    ) -> None:
        super().__init__(name=name, higher_is_better=True)

        if frame_aggregation not in {"max", "mean"}:
            raise ValueError(
                f"frame_aggregation must be 'max' or 'mean', got '{frame_aggregation}'."
            )

        self.event_threshold = event_threshold
        self.frame_aggregation = frame_aggregation

        self.text_metric = CLIPTextImageSimilarityMetric(
            model_name=model_name,
            device=device,
            resize=resize,
            name="clip_text_image_similarity_internal",
        )

    def _aggregate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        if self.frame_aggregation == "mean":
            return float(sum(values) / len(values))
        return float(max(values))

    def _score_event_against_shot(self, event_text: str, shot: Dict[str, Any]) -> float:
        values: List[float] = []

        keyframe = shot.get("reference_keyframe")
        if keyframe is not None:
            result = self.text_metric.compute(text=event_text, image_path=keyframe)
            values.append(result["clip_text_image_similarity_internal"])

        for frame in shot.get("sampled_frames", []):
            result = self.text_metric.compute(text=event_text, image_path=frame)
            values.append(result["clip_text_image_similarity_internal"])

        return self._aggregate(values)

    def compute(self, **kwargs: Any) -> Dict[str, float]:
        events = kwargs.get("events")
        shots = kwargs.get("shots")

        if events is None or shots is None:
            raise ValueError("Missing `events` or `shots` in kwargs.")
        if not isinstance(events, list) or len(events) == 0:
            raise ValueError("`events` must be a non-empty list.")
        if not isinstance(shots, list) or len(shots) == 0:
            raise ValueError("`shots` must be a non-empty list.")

        best_scores: List[float] = []
        best_shot_indices: List[int] = []
        success_flags: List[int] = []

        for event_text in events:
            shot_scores: List[Tuple[int, float]] = []

            for shot_idx, shot in enumerate(shots):
                score = self._score_event_against_shot(event_text=event_text, shot=shot)
                shot_scores.append((shot_idx, score))

            best_shot_idx, best_score = max(shot_scores, key=lambda x: x[1])

            best_shot_indices.append(best_shot_idx)
            best_scores.append(float(best_score))
            success_flags.append(1 if best_score >= self.event_threshold else 0)

        num_events = len(events)
        success_ratio = sum(success_flags) / num_events if num_events > 0 else 0.0
        confidence = sum(best_scores) / num_events if num_events > 0 else 0.0

        valid_pairs = 0
        correct_pairs = 0
        for i in range(num_events - 1):
            if success_flags[i] == 1 and success_flags[i + 1] == 1:
                valid_pairs += 1
                if best_shot_indices[i] <= best_shot_indices[i + 1]:
                    correct_pairs += 1

        if valid_pairs == 0:
            event_order_ratio = 1.0
        else:
            event_order_ratio = correct_pairs / valid_pairs

        controllability_score = (
            0.4 * success_ratio
            + 0.3 * event_order_ratio
            + 0.3 * confidence
        )

        return {
            self.name: float(controllability_score),
            "dynamic_event_success_ratio": float(success_ratio),
            "dynamic_event_order_ratio": float(event_order_ratio),
            "dynamic_event_confidence": float(confidence),
            "dynamic_num_events": float(num_events),
        }