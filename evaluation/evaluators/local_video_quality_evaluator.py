# evaluators/local_video_quality_evaluator.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from evaluators.base_evaluator import BaseEvaluator


class LocalVideoQualityEvaluator(BaseEvaluator):
    """
    Evaluate local-video quality generation quality from both:
    1. StoryEval-inspired dimensions
       - event_completion_score
       - event_order_score
    2. DEVIL-inspired dimensions
       - dynamic_progression_score
       - dynamic_controllability_score

    Supported metric categories:
    A. text-image event metrics
       Required capability:
           metric.compute(text=..., image_path=...)
       Example:
           CLIPTextImageSimilarityMetric

    B. dynamics metrics
       Required capability:
           metric.compute(shots=...)
       or
           metric.compute(events=..., shots=...)
       Examples:
           DynamicProgressionMetric
           DynamicControllabilityMetric

    Recommended sample format:
        {
            "storyboard_id": "story_001",
            "storyboard_description": "A boy enters the kitchen, picks up a cup, and drinks water.",
            "events": [
                "The boy enters the kitchen.",
                "He picks up a cup.",
                "He drinks water."
            ],
            "shots": [
                {
                    "shot_id": "story_001_shot_01",
                    "reference_keyframe": "outputs/.../shot_01/init.png",
                    "sampled_frames": [
                        "outputs/.../shot_01/frame_0001.png",
                        "outputs/.../shot_01/frame_0008.png"
                    ]
                },
                {
                    "shot_id": "story_001_shot_02",
                    "reference_keyframe": "outputs/.../shot_02/init.png",
                    "sampled_frames": [
                        "outputs/.../shot_02/frame_0001.png",
                        "outputs/.../shot_02/frame_0008.png"
                    ]
                }
            ]
        }
    """

    def __init__(
        self,
        metrics: List[Any],
        sample_id_key: str = "storyboard_id",
        storyboard_description_key: str = "storyboard_description",
        events_key: str = "events",
        shots_key: str = "shots",
        shot_id_key: str = "shot_id",
        frames_key: str = "sampled_frames",
        keyframe_key: str = "reference_keyframe",
        event_threshold: float = 0.25,
        frame_aggregation: str = "max",
    ) -> None:
        super().__init__(metrics=metrics)

        if frame_aggregation not in {"max", "mean"}:
            raise ValueError(
                f"frame_aggregation must be 'max' or 'mean', got '{frame_aggregation}'."
            )

        self.sample_id_key = sample_id_key
        self.storyboard_description_key = storyboard_description_key
        self.events_key = events_key
        self.shots_key = shots_key
        self.shot_id_key = shot_id_key
        self.frames_key = frames_key
        self.keyframe_key = keyframe_key
        self.event_threshold = event_threshold
        self.frame_aggregation = frame_aggregation

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        required_keys = [
            self.sample_id_key,
            self.storyboard_description_key,
            self.events_key,
            self.shots_key,
        ]
        missing_keys = [k for k in required_keys if k not in sample]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in long video sample: {missing_keys}. "
                f"Sample keys: {list(sample.keys())}"
            )

        if not isinstance(sample[self.storyboard_description_key], str):
            raise TypeError(
                f"'{self.storyboard_description_key}' must be a string, "
                f"got {type(sample[self.storyboard_description_key]).__name__}."
            )

        events = sample[self.events_key]
        if not isinstance(events, list) or len(events) == 0:
            raise ValueError(f"'{self.events_key}' must be a non-empty list.")

        for idx, event in enumerate(events):
            if not isinstance(event, str):
                raise TypeError(
                    f"Event at index {idx} must be a string, got {type(event).__name__}."
                )

        shots = sample[self.shots_key]
        if not isinstance(shots, list) or len(shots) == 0:
            raise ValueError(f"'{self.shots_key}' must be a non-empty list.")

        for idx, shot in enumerate(shots):
            if not isinstance(shot, dict):
                raise TypeError(
                    f"Shot at index {idx} must be a dict, got {type(shot).__name__}."
                )

            for key in [self.shot_id_key, self.frames_key, self.keyframe_key]:
                if key not in shot:
                    raise KeyError(
                        f"Shot at index {idx} is missing required key '{key}'."
                    )

            frames = shot[self.frames_key]
            if not isinstance(frames, list) or len(frames) == 0:
                raise ValueError(
                    f"Shot '{shot.get(self.shot_id_key, idx)}' must have a non-empty '{self.frames_key}'."
                )

    def _aggregate_values(self, values: List[float]) -> float:
        if not values:
            return 0.0
        if self.frame_aggregation == "mean":
            return float(sum(values) / len(values))
        return float(max(values))

    def _score_event_against_shot(
        self,
        event_text: str,
        shot: Dict[str, Any],
        metric: Any,
    ) -> float:
        """
        Score one event against one shot using a text-image metric.
        """
        values: List[float] = []

        for frame_path in shot[self.frames_key]:
            result = metric.compute(text=event_text, image_path=frame_path)
            metric_value = list(result.values())[0]
            values.append(float(metric_value))

        keyframe_path = shot[self.keyframe_key]
        keyframe_result = metric.compute(text=event_text, image_path=keyframe_path)
        keyframe_value = list(keyframe_result.values())[0]
        values.append(float(keyframe_value))

        return self._aggregate_values(values)

    def _evaluate_story_for_metric(
        self,
        sample: Dict[str, Any],
        metric: Any,
    ) -> Dict[str, Any]:
        """
        StoryEval-inspired event evaluation using one text-image metric.
        """
        events: List[str] = sample[self.events_key]
        shots: List[Dict[str, Any]] = sample[self.shots_key]

        event_success_flags: List[int] = []
        event_best_shot_indices: List[int] = []
        event_best_scores: List[float] = []

        for event_text in events:
            shot_scores: List[Tuple[int, float]] = []

            for shot_idx, shot in enumerate(shots):
                shot_score = self._score_event_against_shot(
                    event_text=event_text,
                    shot=shot,
                    metric=metric,
                )
                shot_scores.append((shot_idx, shot_score))

            best_shot_idx, best_score = max(shot_scores, key=lambda x: x[1])
            success = 1 if best_score >= self.event_threshold else 0

            event_success_flags.append(success)
            event_best_shot_indices.append(best_shot_idx)
            event_best_scores.append(float(best_score))

        num_events = len(events)
        event_completion_score = (
            float(sum(event_success_flags) / num_events) if num_events > 0 else 0.0
        )

        completed_positions = [
            event_best_shot_indices[i]
            for i in range(num_events)
            if event_success_flags[i] == 1
        ]

        if len(completed_positions) <= 1:
            event_order_score = 1.0
        else:
            is_non_decreasing = all(
                completed_positions[i] <= completed_positions[i + 1]
                for i in range(len(completed_positions) - 1)
            )
            event_order_score = 1.0 if is_non_decreasing else 0.0

        result: Dict[str, Any] = {
            f"{metric.name}_event_completion_score": float(event_completion_score),
            f"{metric.name}_event_order_score": float(event_order_score),
            f"{metric.name}_num_events": float(num_events),
        }

        for i in range(num_events):
            result[f"{metric.name}_event_{i+1}_success"] = float(event_success_flags[i])
            result[f"{metric.name}_event_{i+1}_best_shot_idx"] = float(event_best_shot_indices[i])
            result[f"{metric.name}_event_{i+1}_best_score"] = float(event_best_scores[i])

        return result

    def _is_story_metric(self, metric: Any) -> bool:
        """
        Heuristic:
        text-image story metrics usually contain 'text' in metric name.
        """
        name = getattr(metric, "name", "").lower()
        return "text" in name

    def _is_dynamic_progression_metric(self, metric: Any) -> bool:
        name = getattr(metric, "name", "").lower()
        return "dynamic_progression" in name

    def _is_dynamic_controllability_metric(self, metric: Any) -> bool:
        name = getattr(metric, "name", "").lower()
        return "dynamic_controllability" in name

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate one local-video quality sample.

        Returns a combined result dict containing:
        - story-related outputs
        - dynamics-related outputs
        """
        self._validate_sample(sample)

        sample_id = sample[self.sample_id_key]
        shots = sample[self.shots_key]
        events = sample[self.events_key]

        result: Dict[str, Any] = {
            self.sample_id_key: sample_id,
            "num_shots": len(shots),
            "num_events": len(events),
            "event_threshold": float(self.event_threshold),
            "frame_aggregation": self.frame_aggregation,
        }

        for metric in self.metrics:
            if self._is_story_metric(metric):
                metric_result = self._evaluate_story_for_metric(sample=sample, metric=metric)
                result.update(metric_result)

            elif self._is_dynamic_progression_metric(metric):
                metric_result = metric.compute(shots=shots)
                result.update(metric_result)

            elif self._is_dynamic_controllability_metric(metric):
                metric_result = metric.compute(events=events, shots=shots)
                result.update(metric_result)

            else:
                raise ValueError(
                    f"Unsupported metric '{getattr(metric, 'name', metric)}' for LocalVideoQualityEvaluator. "
                    "Expected a story metric or dynamic metric."
                )

        return result

    # def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     results: List[Dict[str, Any]] = []
    #     for sample in dataset:
    #         results.append(self.evaluate_sample(sample))
    #     return results

    def summarize(
        self,
        results: List[Dict[str, Any]],
        prefix: str = "local_video_quality",
    ) -> Dict[str, float]:
        """
        Aggregate dataset-level summary over all numeric fields except identifiers/metadata.
        """
        if not results:
            return {}

        excluded_keys = {
            self.sample_id_key,
            "frame_aggregation",
        }

        candidate_keys = set()
        for item in results:
            for k, v in item.items():
                if k in excluded_keys:
                    continue
                if isinstance(v, (int, float)):
                    candidate_keys.add(k)

        summary: Dict[str, float] = {}

        for key in sorted(candidate_keys):
            values = [float(item[key]) for item in results if key in item]
            if not values:
                continue

            mean_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)

            if len(values) > 1:
                variance = sum((x - mean_value) ** 2 for x in values) / len(values)
                std_value = variance ** 0.5
            else:
                std_value = 0.0

            summary[f"{prefix}_{key}_mean"] = mean_value
            summary[f"{prefix}_{key}_std"] = std_value
            summary[f"{prefix}_{key}_min"] = min_value
            summary[f"{prefix}_{key}_max"] = max_value

        return summary