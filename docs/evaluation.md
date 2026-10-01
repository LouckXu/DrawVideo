# Evaluation

Install metric dependencies in your Python environment:

```bash
python -m pip install -e '.[evaluation]'
```

## Prepare Annotations

After generation, extract six uniformly sampled frames per shot, including the first and last:

```bash
python evaluation/prepare_annotations.py \
  --manifest outputs/example/generation_manifest.json \
  --output-dir outputs/evaluation_annotations
```

This creates `shot_control.json`, `shot_consistency.json`, and `story_alignment.json`. Image paths in these files are relative to the annotation directory and are resolved when evaluating them.

## Run Metrics

```bash
python evaluation/evaluate.py \
  --method-name drawvideo \
  --shot-control-json outputs/evaluation_annotations/shot_control.json \
  --shot-consistency-json outputs/evaluation_annotations/shot_consistency.json \
  --story-alignment-json outputs/evaluation_annotations/story_alignment.json \
  --output-dir outputs/evaluation_results
```

| Paper metric | Implementation |
| --- | --- |
| LPIPS | `metrics/lpips_metric.py` |
| CLIP Image Similarity | `metrics/clip_metric.py` |
| Edge-F1 | `metrics/edge_metric.py` |
| Temporal CLIP / Temporal LPIPS | `metrics/temporal_consistency.py` |
| Static Alignment / Story Alignment | `evaluators/story_alignment_evaluator.py` |
| Event Completion | `evaluators/local_video_quality_evaluator.py` |
| Dynamic Controllability / Dynamic Progression | `metrics/motion_metrics.py` |

Temporal comparisons use the reference keyframe as an anchor. Story Alignment averages frame scores; event matching uses the maximum frame score. The default event threshold is 0.25 and dynamic-change threshold is 0.08. CLIP uses `openai/clip-vit-base-patch32` and LPIPS uses AlexNet.

## Event-Based Evaluation

Specify an ordered `events` array in the input storyboard, using predefined target events from its motion conditions. These are evaluation targets, not automatically inferred from the generated output. Use the same targets for every compared method.

```json
{
  "events": [
    "The character takes a step forward.",
    "The character turns to aim the bow.",
    "The character releases an arrow."
  ]
}
```

The generation manifest retains the input events. Annotation preparation then also writes `local_video_quality.json`. Add it to the metric command:

```bash
python evaluation/evaluate.py \
  --method-name drawvideo \
  --local-video-quality-json outputs/evaluation_annotations/local_video_quality.json \
  --event-frame-aggregation max \
  --output-dir outputs/evaluation_results
```

The supplied evaluation implementation associates events with their best-matching storyboard shots and combines success, ordering, and confidence for Dynamic Controllability. Dynamic Progression is an auxiliary measure of visible temporal change.

The evaluator retains detailed component outputs alongside the aggregate scores. It does not generate a human-study score. Baseline inference code is maintained by the respective method authors and is not redistributed in this repository.
