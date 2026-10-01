import argparse
import json
from pathlib import Path

from .pipeline import STAGES, generate, load_storyboard, read_json, run_stage


def main():
    parser = argparse.ArgumentParser(description="DrawVideo storyboard generation.")
    parser.add_argument("stage", choices=("generate", *STAGES))
    parser.add_argument("--storyboard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drawvideo"))
    parser.add_argument("--config", type=Path, default=Path("configs/generation.json"))
    parser.add_argument("--force", action="store_true", help="Regenerate existing stage outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print the plan without model calls.")
    args = parser.parse_args()
    storyboard = load_storyboard(args.storyboard)
    settings = read_json(args.config)
    video_settings = settings["first_last_frame_video_generation"]
    if video_settings["num_frames"] < 5 or (video_settings["num_frames"] - 1) % 4:
        parser.error("num_frames must be 4n+1 (e.g. 81).")
    if args.dry_run:
        print(json.dumps({"storyboard_id": storyboard["storyboard_id"],
                          "shot_order": [shot["shot_id"] for shot in storyboard["shots"]],
                          "stages": list(STAGES) if args.stage == "generate" else [args.stage],
                          "derivative_keyframes_per_shot": 5,
                          "local_clips_per_shot": 5,
                          "frames_per_local_clip": video_settings["num_frames"]}, indent=2))
        return
    if args.stage == "generate":
        generate(args.storyboard, args.output_dir, args.config, args.force)
    else:
        run_stage(args.stage, args.storyboard, args.output_dir, settings, args.force)


if __name__ == "__main__":
    main()
