"""Prepare portable evaluation annotations from a DrawVideo generation manifest."""

import argparse
import json
import os
from pathlib import Path

from utils.video import extract_uniform_frames


def prepare(manifest_path, output_dir, num_samples=6):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    shot_control, shot_consistency, story_alignment, quality_shots = [], [], [], []
    for shot in manifest["shots"]:
        reference_keyframe = (manifest_path.parent / shot["reference_keyframe"]).resolve()
        shot_video = (manifest_path.parent / shot["shot_video"]).resolve()
        frames = extract_uniform_frames(
            str(shot_video), str(shot_video.parent / "evaluation_frames"),
            num_samples=num_samples, include_first=True, include_last=True)
        reference = os.path.relpath(reference_keyframe, output_dir)
        sampled_frames = [os.path.relpath(frame, output_dir) for frame in frames]
        common = {"shot_id": shot["shot_id"], "reference_keyframe": reference}
        sketch = (manifest_path.parent / shot["sketch"]).resolve()
        shot_control.append({**common, "sketch": os.path.relpath(sketch, output_dir)})
        consistency = {**common, "sampled_frames": sampled_frames}
        shot_consistency.append(consistency)
        story_alignment.append({**consistency,
                                "appearance_prompt": shot["appearance_prompt"],
                                "motion_prompt": shot["motion_prompt"]})
        quality_shots.append(consistency)
    groups = {"shot_control": shot_control, "shot_consistency": shot_consistency,
              "story_alignment": story_alignment}
    if manifest.get("events"):
        groups["local_video_quality"] = [{
            "storyboard_id": manifest["storyboard_id"],
            "storyboard_description": manifest.get("storyboard_description") or
                                      " ".join(shot["motion_prompt"] for shot in manifest["shots"]),
            "events": manifest["events"], "shots": quality_shots,
        }]
    for name, annotations in groups.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(annotations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return list(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()
    groups = prepare(args.manifest, args.output_dir, args.num_samples)
    print("Prepared: " + ", ".join(groups))
    if "local_video_quality" not in groups:
        print("No predefined events were supplied; event evaluation annotations were not generated.")


if __name__ == "__main__":
    main()
