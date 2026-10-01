"""Ordered storyboard orchestration; generation backends are imported lazily."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .structured_prompt_decomposition import decompose_prompts, build_dynamic_prompt

STAGES = ("decompose", "color", "keyframes", "video")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_storyboard(storyboard_path):
    storyboard_path = Path(storyboard_path).resolve()
    storyboard = read_json(storyboard_path)
    shots = storyboard.get("shots", [])
    if not shots:
        raise ValueError("The storyboard must contain at least one shot.")
    shot_ids = []
    for shot in shots:
        shot_id = shot["shot_id"]
        if not shot_id or Path(shot_id).name != shot_id or shot_id in (".", ".."):
            raise ValueError("shot_id must be a nonempty directory name.")
        shot_ids.append(shot_id)
        for field in ("appearance_prompt", "motion_prompt"):
            if not isinstance(shot.get(field), str) or not shot[field].strip():
                raise ValueError(f"{shot_id}: missing {field}.")
        sketch = (storyboard_path.parent / shot["sketch"]).resolve()
        if not sketch.is_file():
            raise FileNotFoundError(f"Sketch not found for {shot_id}.")
    if len(set(shot_ids)) != len(shot_ids):
        raise ValueError("shot_id values must be unique within a storyboard.")
    return storyboard


def concat_videos(video_paths, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The list file uses relative paths, so it remains independent of the host.
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temp_dir:
        list_path = Path(temp_dir) / "clips.txt"
        lines = ["file '" + os.path.relpath(path, temp_dir).replace("'", "'\\''") + "'"
                 for path in video_paths]
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(list_path), "-c", "copy", str(output_path)], check=True)


def run_stage(stage, storyboard_path, output_dir, settings, force=False):
    storyboard_path = Path(storyboard_path).resolve()
    output_dir = Path(output_dir).resolve()
    storyboard = load_storyboard(storyboard_path)
    backend = None
    if stage == "color":
        from .sketch_coloring import SketchColoring
        backend = SketchColoring(settings["sketch_coloring"])
    elif stage == "keyframes":
        from .derivative_keyframes import DerivativeKeyframesGeneration
        backend = DerivativeKeyframesGeneration(settings["derivative_keyframes_generation"])
    elif stage == "video":
        from .first_last_frame_video import generate_video
    shot_videos = []
    for shot in storyboard["shots"]:
        shot_dir = output_dir / "shots" / shot["shot_id"]
        shot_dir.mkdir(parents=True, exist_ok=True)
        sketch = (storyboard_path.parent / shot["sketch"]).resolve()
        decomposition_path = shot_dir / "structured_prompts.json"
        reference_keyframe = shot_dir / "reference_keyframe.png"
        if stage == "decompose":
            if force or not decomposition_path.is_file():
                write_json(decomposition_path, decompose_prompts(
                    shot["appearance_prompt"], shot["motion_prompt"],
                    settings["structured_prompt_decomposition"]))
            continue
        decomposition = read_json(decomposition_path)
        if stage == "color":
            if force or not reference_keyframe.is_file():
                backend.color_one(decomposition["enhanced_appearance_prompt"],
                                  str(sketch), str(reference_keyframe))
        elif stage == "keyframes":
            for entry in decomposition["derivative_keyframes"]:
                derivative_keyframe = shot_dir / f"derivative_keyframe_{entry['index']:02d}.png"
                if force or not derivative_keyframe.is_file():
                    backend.generate_one(entry["conversion_prompt"],
                                         str(reference_keyframe), str(derivative_keyframe))
        elif stage == "video":
            local_clips = []
            first_keyframe = reference_keyframe
            video_settings = settings["first_last_frame_video_generation"]
            for entry in decomposition["derivative_keyframes"]:
                index = entry["index"]
                last_keyframe = shot_dir / f"derivative_keyframe_{index:02d}.png"
                local_clip = shot_dir / f"local_clip_{index:02d}.mp4"
                if force or not local_clip.is_file():
                    generated = generate_video(
                        build_dynamic_prompt(entry["dynamic_prompt"]),
                        video_settings["negative_prompt"], str(first_keyframe),
                        str(last_keyframe), settings=video_settings)
                    if not generated or not Path(generated).is_file():
                        raise RuntimeError(f"Video generation failed for {shot['shot_id']} clip {index}.")
                    shutil.copy2(generated, local_clip)
                local_clips.append(local_clip)
                first_keyframe = last_keyframe
            shot_video = shot_dir / "shot_video.mp4"
            concat_videos(local_clips, shot_video)
            shot_videos.append(shot_video)
    if stage == "video":
        concat_videos(shot_videos, output_dir / "storyboard_video.mp4")
        manifest = dict(storyboard)
        manifest["shots"] = []
        for shot in storyboard["shots"]:
            shot_dir = Path("shots") / shot["shot_id"]
            manifest["shots"].append({
                **shot,
                "sketch": os.path.relpath(storyboard_path.parent / shot["sketch"], output_dir),
                "reference_keyframe": (shot_dir / "reference_keyframe.png").as_posix(),
                "derivative_keyframes": [(shot_dir / f"derivative_keyframe_{i:02d}.png").as_posix()
                                         for i in range(1, 6)],
                "local_clips": [(shot_dir / f"local_clip_{i:02d}.mp4").as_posix()
                                for i in range(1, 6)],
                "shot_video": (shot_dir / "shot_video.mp4").as_posix(),
                "structured_prompts": (shot_dir / "structured_prompts.json").as_posix(),
            })
        manifest["storyboard_video"] = "storyboard_video.mp4"
        write_json(output_dir / "generation_manifest.json", manifest)


def generate(storyboard_path, output_dir, config_path, force=False):
    # Separate processes release model memory between the FLUX and Wan stages.
    for stage in STAGES:
        command = [sys.executable, "-m", "drawvideo", stage, "--storyboard",
                   str(Path(storyboard_path).resolve()), "--output-dir",
                   str(Path(output_dir).resolve()), "--config", str(Path(config_path).resolve())]
        if force:
            command.append("--force")
        subprocess.run(command, check=True)
