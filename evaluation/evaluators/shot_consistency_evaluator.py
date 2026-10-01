# evaluators/shot_consistency_evaluator.py

from __future__ import annotations

from typing import Any, Dict, List

from evaluators.base_evaluator import BaseEvaluator


class ShotConsistencyEvaluator(BaseEvaluator):
    """
    Evaluate intra-shot consistency.

    Main question:
        Are frames within one shot visually and semantically consistent?

    Typical usage:
        - Compare sampled frames against an explicit appearance anchor
        - Or compare adjacent frames for smoothness/stability

    Recommended sample format:
        {
            "shot_id": "video001_shot03",
            "reference_keyframe": "outputs/drawvideo/video001_shot03/init.png",
            "sampled_frames": [
                "outputs/drawvideo/video001_shot03/frame_0001.png",
                "outputs/drawvideo/video001_shot03/frame_0008.png",
                "outputs/drawvideo/video001_shot03/frame_0016.png"
            ]
        }
    """

    def __init__(
        self,
        metrics: List[Any],
        frames_key: str = "sampled_frames",
        anchor_key: str = "reference_keyframe",
        sample_id_key: str = "shot_id",
    ) -> None:
        """
        Args:
            metrics: List of temporal metric instances.
            frames_key: Key in sample dict for the frame list.
            anchor_key: Key in sample dict for the anchor image path.
            sample_id_key: Key in sample dict for unique sample ID.
        """
        super().__init__(metrics=metrics)
        self.frames_key = frames_key
        self.anchor_key = anchor_key
        self.sample_id_key = sample_id_key

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        required_keys = [self.sample_id_key, self.frames_key, self.anchor_key]
        missing_keys = [k for k in required_keys if k not in sample]

        if missing_keys:
            raise KeyError(
                f"Missing required keys in sample: {missing_keys}. "
                f"Sample keys: {list(sample.keys())}"
            )

        frames = sample[self.frames_key]
        if not isinstance(frames, list):
            raise TypeError(
                f"'{self.frames_key}' must be a list, got {type(frames).__name__}."
            )

        if len(frames) < 2:
            raise ValueError(
                f"'{self.frames_key}' must contain at least 2 frames, got {len(frames)}."
            )

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate one shot sample.

        Returns:
            Example:
                {
                    "shot_id": "video001_shot03",
                    "frames_key": "sampled_frames",
                    "anchor_key": "reference_keyframe",
                    "num_frames": 8,
                    "temporal_clip": 0.84,
                    "temporal_clip_num_pairs": 8.0,
                    "temporal_lpips": 0.12,
                    "temporal_lpips_num_pairs": 8.0
                }
        """
        self._validate_sample(sample)

        sample_id = sample[self.sample_id_key]
        frames = sample[self.frames_key]
        anchor = sample[self.anchor_key]

        result: Dict[str, Any] = {
            self.sample_id_key: sample_id,
            "frames_key": self.frames_key,
            "anchor_key": self.anchor_key,
            "num_frames": len(frames),
        }

        for metric in self.metrics:
            metric_result = metric.compute(
                frames=frames,
                anchor_path=anchor,
            )
            result.update(metric_result)

        return result

    # def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     Evaluate all shot samples in the dataset.
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
        prefix: str = "shot_consistency",
    ) -> Dict[str, float]:
        """
        Aggregate dataset-level summary across all temporal metrics.
        """
        if not results:
            return {}

        summary: Dict[str, float] = {}

        for metric in self.metrics:
            metric_name = metric.name

            metric_results = []
            for item in results:
                if metric_name in item:
                    metric_results.append({metric_name: item[metric_name]})

            metric_summary = metric.aggregate(metric_results, prefix=prefix)
            summary.update(metric_summary)

        return summary
