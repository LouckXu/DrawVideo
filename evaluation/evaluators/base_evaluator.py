# evaluators/base_evaluator.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.

    Responsibilities:
    1. Hold a list of metric instances.
    2. Define a unified interface for evaluating one sample.
    3. Provide default dataset-level evaluation logic.
    4. Provide default summarization logic.

    Typical subclasses:
        - ShotControlEvaluator
        - ShotConsistencyEvaluator
        - StoryAlignmentEvaluator
        - LocalVideoQualityEvaluator
    """

    def __init__(self, metrics: Optional[List[Any]] = None) -> None:
        """
        Args:
            metrics: A list of metric instances. Each metric is expected
                     to have:
                        - a `name` attribute
                        - a callable interface or `compute()` method
                        - an `aggregate()` method
        """
        self.metrics = metrics or []
        self._validate_metrics()

    def _validate_metrics(self) -> None:
        """Basic validation for metric instances."""
        for metric in self.metrics:
            if not hasattr(metric, "name"):
                raise AttributeError(
                    f"Metric {metric} must have a 'name' attribute."
                )

            if not callable(metric):
                raise TypeError(
                    f"Metric '{getattr(metric, 'name', str(metric))}' must be callable."
                )

            if not hasattr(metric, "aggregate"):
                raise AttributeError(
                    f"Metric '{getattr(metric, 'name', str(metric))}' must implement 'aggregate()'."
                )

    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate one sample.

        Args:
            sample: A sample dictionary from dataset annotations.

        Returns:
            A result dictionary for one sample, for example:
                {
                    "shot_id": "shot_001",
                    "lpips": 0.123,
                    "clip_image_similarity": 0.812
                }
        """
        raise NotImplementedError

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all samples in a dataset.

        Args:
            dataset: A list of sample dicts.
            show_progress: Whether to show simple progress prints.

        Returns:
            A list of per-sample result dicts.
        """
        results: List[Dict[str, Any]] = []
        total = len(dataset)

        for idx, sample in enumerate(dataset, start=1):
            if show_progress:
                print(f"[{self.__class__.__name__}] Evaluating {idx}/{total}...")

            result = self.evaluate_sample(sample)
            results.append(result)

        return results

    def summarize(
        self,
        results: List[Dict[str, Any]],
        prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Aggregate metric results across all evaluated samples.

        Args:
            results: Per-sample results returned by `evaluate_dataset`.
            prefix: Prefix for output summary keys.

        Returns:
            Example:
                {
                    "eval_lpips_mean": 0.123,
                    "eval_lpips_std": 0.015,
                    "eval_lpips_min": 0.089,
                    "eval_lpips_max": 0.151,
                    "eval_clip_image_similarity_mean": 0.812,
                    ...
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

            if not metric_results:
                continue

            metric_summary = metric.aggregate(metric_results, prefix=prefix)
            summary.update(metric_summary)

        return summary

    def evaluate_and_summarize(
        self,
        dataset: List[Dict[str, Any]],
        prefix: str = "eval",
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method: evaluate the dataset and summarize results.

        Returns:
            {
                "results": [...],
                "summary": {...}
            }
        """
        results = self.evaluate_dataset(dataset, show_progress=show_progress)
        summary = self.summarize(results, prefix=prefix)

        return {
            "results": results,
            "summary": summary,
        }

    def metric_names(self) -> List[str]:
        """Return the list of metric names."""
        return [metric.name for metric in self.metrics]

    def add_metric(self, metric: Any) -> None:
        """Add one metric after initialization."""
        self.metrics.append(metric)
        self._validate_metrics()

    def __repr__(self) -> str:
        metric_names = ", ".join(self.metric_names()) if self.metrics else "None"
        return f"{self.__class__.__name__}(metrics=[{metric_names}])"