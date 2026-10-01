"""Import one downloaded SketchLongVideo package into the paper-aligned schema."""

import argparse
import json
from pathlib import Path
import shutil


def import_storyboard(video_package, output_dir):
    video_package = Path(video_package)
    output_dir = Path(output_dir)
    sketch_dir = output_dir / "sketches"
    sketch_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for index, sketch in enumerate(sorted((video_package / "sketch").glob("*.png")), 1):
        appearance = video_package / "static_prompt" / (sketch.stem + ".txt")
        motion = video_package / "story" / (sketch.stem + ".txt")
        if not appearance.is_file() or not motion.is_file():
            raise FileNotFoundError(f"Missing paired appearance or motion text for {sketch.name}.")
        shot_id = f"shot_{index:03d}"
        shutil.copy2(sketch, sketch_dir / f"{shot_id}.png")
        shots.append({"shot_id": shot_id, "sketch": f"sketches/{shot_id}.png",
                      "appearance_prompt": appearance.read_text(encoding="utf-8").strip(),
                      "motion_prompt": motion.read_text(encoding="utf-8").strip(),
                      "source_sample_id": sketch.stem})
    if not shots:
        raise ValueError("The video package contains no sketch PNGs.")
    manifest = {"storyboard_id": video_package.name, "shots": shots}
    path = output_dir / "storyboard.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(import_storyboard(args.video_package, args.output_dir))


if __name__ == "__main__":
    main()
