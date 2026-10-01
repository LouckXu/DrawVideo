# metrics/base_metric.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import math


class BaseMetric(ABC):
    """
    Base class for all evaluation metrics.

    Design goals:
    1. Provide a unified interface for all metrics.
    2. Support single-sample computation via `compute`.
    3. Support dataset-level aggregation via `aggregate`.
    4. Keep output format consistent: Dict[str, float].

    Example:
        class MyMetric(BaseMetric):
            def compute(self, pred_path: str, gt_path: str) -> Dict[str, float]:
                score = ...
                return {self.name: score}
    """

    def __init__(self, name: str, higher_is_better: bool) -> None:
        """
        Args:
            name: Metric name used in result dictionaries.
            higher_is_better: Whether a larger value means better performance.
        """
        if not name:
            raise ValueError("Metric name must be a non-empty string.")

        self.name = name
        self.higher_is_better = higher_is_better

    @abstractmethod
    def compute(self, **kwargs: Any) -> Dict[str, float]:
        """
        Compute the metric for one sample.

        Returns:
            A dictionary like:
                {"lpips": 0.1234}
                {"clip_image_similarity": 0.8123}
        """
        raise NotImplementedError

    def validate_result(self, result: Dict[str, float]) -> Dict[str, float]:
        """
        Validate metric output format.

        Rules:
        - Must be a dict
        - Keys must be strings
        - Values must be finite numbers

        Returns:
            The validated result dict.
        """
        if not isinstance(result, dict):
            raise TypeError(
                f"{self.__class__.__name__}.compute() must return a dict, "
                f"got {type(result).__name__}."
            )

        for key, value in result.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Metric result key must be str, got {type(key).__name__}."
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Metric result value for '{key}' must be int or float, "
                    f"got {type(value).__name__}."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Metric result value for '{key}' must be finite, got {value}."
                )

        return {k: float(v) for k, v in result.items()}

    def __call__(self, **kwargs: Any) -> Dict[str, float]:
        """
        Allow metric instances to be called like a function.

        Example:
            result = metric(pred_path="a.png", gt_path="b.png")
        """
        result = self.compute(**kwargs)
        return self.validate_result(result)

    def aggregate(
        self,
        results: List[Dict[str, float]],
        prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Aggregate a list of per-sample metric results into dataset-level summary.

        By default, computes mean/std/min/max for each metric key.

        Args:
            results: A list of dictionaries returned by `compute`.
            prefix: Optional prefix such as "shot", "dataset", or "local_video_quality".

        Returns:
            Example:
                {
                    "dataset_lpips_mean": 0.123,
                    "dataset_lpips_std": 0.015,
                    "dataset_lpips_min": 0.089,
                    "dataset_lpips_max": 0.151
                }
        """
        if not results:
            return {}

        validated_results = [self.validate_result(r) for r in results]

        keys = sorted({key for result in validated_results for key in result.keys()})
        summary: Dict[str, float] = {}

        for key in keys:
            values = [result[key] for result in validated_results if key in result]
            if not values:
                continue

            mean_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)

            if len(values) > 1:
                variance = sum((x - mean_value) ** 2 for x in values) / len(values)
                std_value = math.sqrt(variance)
            else:
                std_value = 0.0

            base_name = f"{prefix}_{key}" if prefix else key

            summary[f"{base_name}_mean"] = mean_value
            summary[f"{base_name}_std"] = std_value
            summary[f"{base_name}_min"] = min_value
            summary[f"{base_name}_max"] = max_value

        return summary

    def metadata(self) -> Dict[str, Any]:
        """
        Return basic metadata for logging or experiment tracking.
        """
        return {
            "name": self.name,
            "higher_is_better": self.higher_is_better,
            "class_name": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"higher_is_better={self.higher_is_better})"
        )