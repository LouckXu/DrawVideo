# scripts/run_eval_all.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluators.shot_control_evaluator import ShotControlEvaluator
from evaluators.shot_consistency_evaluator import ShotConsistencyEvaluator
from evaluators.story_alignment_evaluator import StoryAlignmentEvaluator
from evaluators.local_video_quality_evaluator import LocalVideoQualityEvaluator

from metrics.lpips_metric import LPIPSMetric
from metrics.clip_metric import CLIPImageSimilarityMetric
from metrics.edge_metric import EdgeOverlapMetric
from metrics.temporal_consistency import (
    TemporalCLIPConsistency,
    TemporalLPIPSConsistency,
)
from metrics.clip_text_metric import CLIPTextImageSimilarityMetric
from metrics.motion_metrics import (
    DynamicProgressionMetric,
    DynamicControllabilityMetric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Draw Video evaluations."
    )

    parser.add_argument(
        "--method-name",
        type=str,
        default="drawvideo",
        help="Method name used in output filenames and summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save all evaluation outputs.",
    )

    # dataset jsons
    parser.add_argument(
        "--shot-control-json",
        type=str,
        default=None,
        help="Path to shot control dataset JSON.",
    )
    parser.add_argument(
        "--shot-consistency-json",
        type=str,
        default=None,
        help="Path to shot consistency dataset JSON.",
    )
    parser.add_argument(
        "--story-alignment-json",
        type=str,
        default=None,
        help="Path to story alignment dataset JSON.",
    )
    parser.add_argument(
        "--local-video-quality-json",
        type=str,
        default=None,
        help="Path to local-video quality dataset JSON.",
    )

    # shared metric settings
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for metric computation, e.g. 'cuda' or 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--resize-h",
        type=int,
        default=224,
        help="Resize height for image-based metrics.",
    )
    parser.add_argument(
        "--resize-w",
        type=int,
        default=224,
        help="Resize width for image-based metrics.",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model name.",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone.",
    )

    # temporal / story options
    parser.add_argument(
        "--temporal-mode",
        type=str,
        default="anchor",
        choices=["anchor", "adjacent"],
        help="Temporal comparison mode for shot consistency.",
    )
    parser.add_argument(
        "--frame-aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Frame aggregation mode for story alignment.",
    )
    parser.add_argument(
        "--event-frame-aggregation", choices=["mean", "max"], default="max",
        help="Frame aggregation for event matching (paper: max).",
    )

    # local-video quality options
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.25,
        help="Threshold for considering an event completed.",
    )
    parser.add_argument(
        "--dynamic-change-threshold",
        type=float,
        default=0.08,
        help="Threshold for considering adjacent change as meaningful dynamic progression.",
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Whether to print evaluation progress.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_dataset(dataset: Any, dataset_name: str) -> List[Dict[str, Any]]:
    if not isinstance(dataset, list):
        raise TypeError(
            f"{dataset_name} must be a list of sample dicts, got {type(dataset).__name__}."
        )
    if len(dataset) == 0:
        raise ValueError(f"{dataset_name} is empty.")
    for idx, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise TypeError(
                f"{dataset_name}[{idx}] must be a dict, got {type(sample).__name__}."
            )
    return dataset


def maybe_load_dataset(path_str: Optional[str], dataset_name: str) -> Optional[List[Dict[str, Any]]]:
    if path_str is None:
        return None
    annotation_path = Path(path_str).resolve()
    dataset = load_json(annotation_path)
    resolve_annotation_paths(dataset, annotation_path.parent)
    return validate_dataset(dataset, dataset_name)


def resolve_annotation_paths(value, root: Path):
    """Resolve portable annotation paths only when files are consumed."""
    if isinstance(value, list):
        for item in value:
            resolve_annotation_paths(item, root)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"sketch", "reference_keyframe", "shot_video"} and isinstance(item, str):
                value[key] = str((root / item).resolve())
            elif key == "sampled_frames":
                value[key] = [str((root / frame).resolve()) for frame in item]
            else:
                resolve_annotation_paths(item, root)


def run_shot_control(
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    print("[Info] Initializing shot control metrics...")
    metrics = [
        LPIPSMetric(
            net=args.lpips_net,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            name="lpips",
        ),
        CLIPImageSimilarityMetric(
            model_name=args.clip_model_name,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            name="clip_image_similarity",
        ),
        EdgeOverlapMetric(
            resize=(args.resize_h, args.resize_w),
            name="edge_f1",
        ),
    ]

    evaluator = ShotControlEvaluator(metrics=metrics)

    print("[Info] Running shot control evaluation...")
    output = evaluator.evaluate_and_summarize(
        dataset=dataset,
        prefix="shot_control",
        show_progress=args.show_progress,
    )

    results = output["results"]
    summary = {
        "method_name": args.method_name,
        "num_samples": len(results),
        **output["summary"],
    }

    raw_path = output_dir / f"{args.method_name}_shot_control_results.json"
    summary_path = output_dir / f"{args.method_name}_shot_control_summary.json"

    save_json(results, raw_path)
    save_json(summary, summary_path)

    return {
        "results_path": str(raw_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def run_shot_consistency(
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    print("[Info] Initializing shot consistency metrics...")
    metrics = [
        TemporalCLIPConsistency(
            model_name=args.clip_model_name,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            mode=args.temporal_mode,
            name="temporal_clip",
        ),
        TemporalLPIPSConsistency(
            net=args.lpips_net,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            mode=args.temporal_mode,
            name="temporal_lpips",
        ),
    ]

    evaluator = ShotConsistencyEvaluator(metrics=metrics)

    print("[Info] Running shot consistency evaluation...")
    output = evaluator.evaluate_and_summarize(
        dataset=dataset,
        prefix="shot_consistency",
        show_progress=args.show_progress,
    )

    results = output["results"]
    summary = {
        "method_name": args.method_name,
        "num_samples": len(results),
        "temporal_mode": args.temporal_mode,
        **output["summary"],
    }

    raw_path = output_dir / f"{args.method_name}_shot_consistency_results.json"
    summary_path = output_dir / f"{args.method_name}_shot_consistency_summary.json"

    save_json(results, raw_path)
    save_json(summary, summary_path)

    return {
        "results_path": str(raw_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def run_story_alignment(
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    print("[Info] Initializing story alignment metrics...")
    metrics = [
        CLIPTextImageSimilarityMetric(
            model_name=args.clip_model_name,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            name="clip_text_image_similarity",
        )
    ]

    evaluator = StoryAlignmentEvaluator(
        metrics=metrics,
        frame_aggregation=args.frame_aggregation,
    )

    print("[Info] Running story alignment evaluation...")
    output = evaluator.evaluate_and_summarize(
        dataset=dataset,
        prefix="story_alignment",
        show_progress=args.show_progress,
    )

    results = output["results"]
    summary = {
        "method_name": args.method_name,
        "num_samples": len(results),
        "frame_aggregation": args.frame_aggregation,
        **output["summary"],
    }

    raw_path = output_dir / f"{args.method_name}_story_alignment_results.json"
    summary_path = output_dir / f"{args.method_name}_story_alignment_summary.json"

    save_json(results, raw_path)
    save_json(summary, summary_path)

    return {
        "results_path": str(raw_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def run_local_video_quality(
    dataset: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    print("[Info] Initializing local-video quality metrics...")
    metrics = [
        # StoryEval-inspired
        CLIPTextImageSimilarityMetric(
            model_name=args.clip_model_name,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            name="clip_text_image_similarity",
        ),
        # DEVIL-inspired
        DynamicProgressionMetric(
            model_name=args.clip_model_name,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            change_threshold=args.dynamic_change_threshold,
            name="dynamic_progression_score",
        ),
        DynamicControllabilityMetric(
            model_name=args.clip_model_name,
            device=args.device,
            resize=(args.resize_h, args.resize_w),
            event_threshold=args.event_threshold,
            frame_aggregation=args.event_frame_aggregation,
            name="dynamic_controllability_score",
        ),
    ]

    evaluator = LocalVideoQualityEvaluator(
        metrics=metrics,
        event_threshold=args.event_threshold,
        frame_aggregation=args.event_frame_aggregation,
    )

    print("[Info] Running local-video quality evaluation...")
    output = evaluator.evaluate_and_summarize(
        dataset=dataset,
        prefix="local_video_quality",
        show_progress=args.show_progress,
    )

    results = output["results"]
    summary = {
        "method_name": args.method_name,
        "num_samples": len(results),
        "event_threshold": args.event_threshold,
        "frame_aggregation": args.event_frame_aggregation,
        "dynamic_change_threshold": args.dynamic_change_threshold,
        **output["summary"],
    }

    raw_path = output_dir / f"{args.method_name}_local_video_quality_results.json"
    summary_path = output_dir / f"{args.method_name}_local_video_quality_summary.json"

    save_json(results, raw_path)
    save_json(summary, summary_path)

    return {
        "results_path": str(raw_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (
        args.shot_control_json is None
        and args.shot_consistency_json is None
        and args.story_alignment_json is None
        and args.local_video_quality_json is None
    ):
        raise ValueError(
            "At least one dataset JSON must be provided: "
            "--shot_control_json / --shot_consistency_json / "
            "--story_alignment_json / --local_video_quality_json"
        )

    all_outputs: Dict[str, Any] = {
        "method_name": args.method_name,
    }

    # shot control
    if args.shot_control_json is not None:
        print(f"[Info] Loading shot control dataset from: {args.shot_control_json}")
        shot_control_dataset = maybe_load_dataset(args.shot_control_json, "shot_control_json")
        print(f"[Info] Loaded {len(shot_control_dataset)} shot control samples.")
        all_outputs["shot_control"] = run_shot_control(
            dataset=shot_control_dataset,
            args=args,
            output_dir=output_dir,
        )

    # shot consistency
    if args.shot_consistency_json is not None:
        print(f"[Info] Loading shot consistency dataset from: {args.shot_consistency_json}")
        shot_consistency_dataset = maybe_load_dataset(args.shot_consistency_json, "shot_consistency_json")
        print(f"[Info] Loaded {len(shot_consistency_dataset)} shot consistency samples.")
        all_outputs["shot_consistency"] = run_shot_consistency(
            dataset=shot_consistency_dataset,
            args=args,
            output_dir=output_dir,
        )

    # story alignment
    if args.story_alignment_json is not None:
        print(f"[Info] Loading story alignment dataset from: {args.story_alignment_json}")
        story_alignment_dataset = maybe_load_dataset(args.story_alignment_json, "story_alignment_json")
        print(f"[Info] Loaded {len(story_alignment_dataset)} story alignment samples.")
        all_outputs["story_alignment"] = run_story_alignment(
            dataset=story_alignment_dataset,
            args=args,
            output_dir=output_dir,
        )

    # long video
    if args.local_video_quality_json is not None:
        print(f"[Info] Loading local-video quality dataset from: {args.local_video_quality_json}")
        local_video_quality_dataset = maybe_load_dataset(args.local_video_quality_json, "local_video_quality_json")
        print(f"[Info] Loaded {len(local_video_quality_dataset)} local-video quality samples.")
        all_outputs["local_video_quality"] = run_local_video_quality(
            dataset=local_video_quality_dataset,
            args=args,
            output_dir=output_dir,
        )

    # merged summary
    merged_summary: Dict[str, Any] = {
        "method_name": args.method_name,
    }

    for section_name in ["shot_control", "shot_consistency", "story_alignment", "local_video_quality"]:
        if section_name in all_outputs:
            section_summary = all_outputs[section_name]["summary"]
            for k, v in section_summary.items():
                merged_summary[f"{section_name}_{k}"] = v

    merged_summary_path = output_dir / f"{args.method_name}_all_eval_summary.json"
    save_json(merged_summary, merged_summary_path)

    all_outputs["merged_summary_path"] = str(merged_summary_path)
    all_outputs["merged_summary"] = merged_summary

    manifest_path = output_dir / f"{args.method_name}_eval_manifest.json"
    save_json(all_outputs, manifest_path)

    print("[Done] All requested evaluations finished.")
    print(json.dumps(merged_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
