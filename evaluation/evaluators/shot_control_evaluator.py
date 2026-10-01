# evaluators/shot_control_evaluator.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluators.base_evaluator import BaseEvaluator


class ShotControlEvaluator(BaseEvaluator):
    """
    Evaluate shot-level structural controllability.

    Main question:
        Does the generated result faithfully follow the input sketch / control signal?

    Current default use case:
        input sketch  --> generated initial keyframe

    Supported sample fields (recommended):
        {
            "shot_id": "video001_shot03",
            "sketch": "data/sketches/video001_shot03.png",
            "reference_keyframe": "outputs/drawvideo/video001_shot03/init.png"
        }

    Optional fields for future extension:
        {
            "reference_keyframe": "...",
            "control_frame": "...",
            ...
        }
    """

    def __init__(
        self,
        metrics: List[Any],
        input_key: str = "sketch",
        target_key: str = "reference_keyframe",
        sample_id_key: str = "shot_id",
    ) -> None:
        """
        Args:
            metrics: A list of metric instances, e.g. [LPIPSMetric(), CLIPMetric()]
            input_key: Sample key for the control input image.
            target_key: Sample key for the generated target image.
            sample_id_key: Sample key for unique sample identifier.
        """
        super().__init__(metrics=metrics)
        self.input_key = input_key
        self.target_key = target_key
        self.sample_id_key = sample_id_key

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        """Validate required fields in one sample."""
        required_keys = [self.sample_id_key, self.input_key, self.target_key]
        missing_keys = [k for k in required_keys if k not in sample]

        if missing_keys:
            raise KeyError(
                f"Missing required keys in sample: {missing_keys}. "
                f"Sample keys: {list(sample.keys())}"
            )

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate one sample.

        Returns:
            {
                "shot_id": "...",
                "input_key": "sketch",
                "target_key": "reference_keyframe",
                "lpips": 0.123,
                "clip_image_similarity": 0.812
            }
        """
        self._validate_sample(sample)

        sample_id = sample[self.sample_id_key]
        input_value = sample[self.input_key]
        target_value = sample[self.target_key]

        result: Dict[str, Any] = {
            self.sample_id_key: sample_id,
            "input_key": self.input_key,
            "target_key": self.target_key,
        }

        for metric in self.metrics:
            metric_result = metric(
                pred_path=target_value,
                gt_path=input_value,
            )
            result.update(metric_result)

        return result

    # def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     Evaluate all samples in the dataset.
    #
    #     Returns:
    #         A list of per-sample result dictionaries.
    #     """
    #     results: List[Dict[str, Any]] = []
    #
    #     for sample in dataset:
    #         result = self.evaluate_sample(sample)
    #         results.append(result)
    #
    #     return results

    def summarize(self, results: List[Dict[str, Any]], prefix: str = "shot_control") -> Dict[str, float]:
        """
        Aggregate dataset-level statistics across all metrics.

        Args:
            results: Output from evaluate_dataset()
            prefix: Prefix used in summary keys

        Returns:
            Example:
                {
                    "shot_control_lpips_mean": 0.123,
                    "shot_control_lpips_std": 0.015,
                    "shot_control_lpips_min": 0.089,
                    "shot_control_lpips_max": 0.151
                }
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
