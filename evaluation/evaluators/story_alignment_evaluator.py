# evaluators/story_alignment_evaluator.py

from __future__ import annotations

from typing import Any, Dict, List

from evaluators.base_evaluator import BaseEvaluator


class StoryAlignmentEvaluator(BaseEvaluator):
    """
    Evaluate semantic alignment between text inputs and generated visual outputs.

    Current supported alignment targets:
    1. appearance_prompt -> reference_keyframe
    2. motion_prompt -> sampled_frames

    Recommended sample format:
        {
            "shot_id": "shot_001",
            "appearance_prompt": "A young boy in a red shirt stands in a kitchen.",
            "motion_prompt": "The boy picks up a cup and drinks water.",
            "reference_keyframe": "outputs/drawvideo/shot_001/init.png",
            "sampled_frames": [
                "outputs/drawvideo/shot_001/frame_0001.png",
                "outputs/drawvideo/shot_001/frame_0008.png",
                "outputs/drawvideo/shot_001/frame_0016.png"
            ]
        }
    """

    def __init__(
        self,
        metrics: List[Any],
        appearance_prompt_key: str = "appearance_prompt",
        motion_prompt_key: str = "motion_prompt",
        keyframe_key: str = "reference_keyframe",
        frames_key: str = "sampled_frames",
        sample_id_key: str = "shot_id",
        frame_aggregation: str = "mean",
    ) -> None:
        """
        Args:
            metrics: List of text-image metric instances.
            appearance_prompt_key: Key for static text description.
            motion_prompt_key: Key for local story text.
            keyframe_key: Key for generated initial keyframe path.
            frames_key: Key for sampled frame paths.
            sample_id_key: Key for sample ID.
            frame_aggregation: How to aggregate motion_prompt vs sampled_frames scores.
                               Choices: 'mean', 'max'
        """
        super().__init__(metrics=metrics)

        if frame_aggregation not in {"mean", "max"}:
            raise ValueError(
                f"frame_aggregation must be 'mean' or 'max', got '{frame_aggregation}'."
            )

        self.appearance_prompt_key = appearance_prompt_key
        self.motion_prompt_key = motion_prompt_key
        self.keyframe_key = keyframe_key
        self.frames_key = frames_key
        self.sample_id_key = sample_id_key
        self.frame_aggregation = frame_aggregation

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        required_keys = [
            self.sample_id_key,
            self.appearance_prompt_key,
            self.motion_prompt_key,
            self.keyframe_key,
            self.frames_key,
        ]
        missing_keys = [k for k in required_keys if k not in sample]

        if missing_keys:
            raise KeyError(
                f"Missing required keys in sample: {missing_keys}. "
                f"Sample keys: {list(sample.keys())}"
            )

        if not isinstance(sample[self.appearance_prompt_key], str):
            raise TypeError(
                f"'{self.appearance_prompt_key}' must be a string, "
                f"got {type(sample[self.appearance_prompt_key]).__name__}."
            )

        if not isinstance(sample[self.motion_prompt_key], str):
            raise TypeError(
                f"'{self.motion_prompt_key}' must be a string, "
                f"got {type(sample[self.motion_prompt_key]).__name__}."
            )

        frames = sample[self.frames_key]
        if not isinstance(frames, list):
            raise TypeError(
                f"'{self.frames_key}' must be a list, got {type(frames).__name__}."
            )
        if len(frames) == 0:
            raise ValueError(f"'{self.frames_key}' must not be empty.")

    def _aggregate_scores(self, scores: List[float]) -> float:
        if not scores:
            return 0.0

        if self.frame_aggregation == "mean":
            return float(sum(scores) / len(scores))

        return float(max(scores))

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate semantic alignment for one sample.

        Returns:
            Example:
                {
                    "shot_id": "shot_001",
                    "num_frames": 3,
                    "clip_text_image_similarity_static_alignment": 0.31,
                    "clip_text_image_similarity_story_alignment": 0.28
                }
        """
        self._validate_sample(sample)

        sample_id = sample[self.sample_id_key]
        appearance_prompt = sample[self.appearance_prompt_key]
        motion_prompt = sample[self.motion_prompt_key]
        keyframe = sample[self.keyframe_key]
        frames = sample[self.frames_key]

        result: Dict[str, Any] = {
            self.sample_id_key: sample_id,
            "num_frames": len(frames),
            "frame_aggregation": self.frame_aggregation,
        }

        for metric in self.metrics:
            # 1) appearance_prompt -> initial keyframe
            static_result = metric.compute(
                text=appearance_prompt,
                image_path=keyframe,
            )

            for k, v in static_result.items():
                result[f"{k}_static_alignment"] = v

            # 2) motion_prompt -> sampled frames
            frame_scores: Dict[str, List[float]] = {}
            for frame_path in frames:
                frame_result = metric.compute(
                    text=motion_prompt,
                    image_path=frame_path,
                )
                for k, v in frame_result.items():
                    frame_scores.setdefault(k, []).append(v)

            for k, values in frame_scores.items():
                result[f"{k}_story_alignment"] = self._aggregate_scores(values)
                result[f"{k}_story_num_frames"] = float(len(values))

        return result

    # def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     Evaluate all samples in the dataset.
    #     """
    #     results: List[Dict[str, Any]] = []
    #
    #     for sample in dataset:
    #         result = self.evaluate_sample(sample)
    #         results.append(result)
    #
    #     return results

    def summarize(
        self,
        results: List[Dict[str, Any]],
        prefix: str = "story_alignment",
    ) -> Dict[str, float]:
        """
        Aggregate dataset-level summary for all numeric metric outputs
        generated by this evaluator.

        This is slightly more generic than metric.aggregate(), because
        each metric here expands into multiple derived fields:
            - xxx_static_alignment
            - xxx_story_alignment
        """
        if not results:
            return {}

        summary: Dict[str, float] = {}

        # collect all numeric fields except identifiers / metadata
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