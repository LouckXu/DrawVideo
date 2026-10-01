import contextlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from drawvideo import pipeline
from drawvideo.structured_prompt_decomposition import validate_decomposition, build_dynamic_prompt

ROOT = Path(__file__).resolve().parents[1]


def decomposition():
    return {
        "enhanced_appearance_prompt": "A fairy in a green forest.",
        "derivative_keyframes": [{
            "index": i, "conversion_prompt": f"Action state {i}.",
            "dynamic_prompt": {field: f"{field} {i}" for field in (
                "positive", "animation_action", "body_motion",
                "facial_expression_change", "style_consistency")},
        } for i in range(1, 6)],
    }


class PipelineTests(unittest.TestCase):
    def test_example_and_configuration(self):
        storyboard = pipeline.load_storyboard(ROOT / "examples/storyboard/storyboard.json")
        self.assertEqual([shot["shot_id"] for shot in storyboard["shots"]], ["shot_001", "shot_002"])
        settings = pipeline.read_json(ROOT / "configs/generation.json")
        self.assertEqual(settings["first_last_frame_video_generation"]["num_frames"], 81)
        self.assertEqual(settings["sketch_coloring"]["preprocessing_resolution"], 1024)

    def test_five_ordered_prompt_pairs(self):
        value = decomposition()
        self.assertEqual(validate_decomposition(value), value)
        dynamic = value["derivative_keyframes"][0]["dynamic_prompt"]
        self.assertIn("animation_action 1", build_dynamic_prompt(dynamic))
        value["derivative_keyframes"].reverse()
        with self.assertRaises(ValueError):
            validate_decomposition(value)

    def test_portable_evaluation_annotations(self):
        fake_video_utils = types.ModuleType("utils.video")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = [root / f"frame_{i:02d}.png" for i in range(6)]
            fake_video_utils.extract_uniform_frames = lambda *args, **kwargs: [str(path) for path in frames]
            spec = importlib.util.spec_from_file_location(
                "prepare_annotations", ROOT / "evaluation/prepare_annotations.py")
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"utils.video": fake_video_utils}):
                spec.loader.exec_module(module)
            pipeline.write_json(root / "generation_manifest.json", {
                "storyboard_id": "test", "events": ["The character moves."],
                "shots": [{"shot_id": "shot_001", "sketch": "sketch.png",
                           "reference_keyframe": "reference_keyframe.png", "shot_video": "shot_video.mp4",
                           "appearance_prompt": "A character.", "motion_prompt": "The character moves."}],
            })
            output_dir = root / "annotations"
            groups = module.prepare(root / "generation_manifest.json", output_dir)
            self.assertEqual(groups, ["shot_control", "shot_consistency", "story_alignment", "local_video_quality"])
            control = pipeline.read_json(output_dir / "shot_control.json")[0]
            self.assertEqual(control["reference_keyframe"], "../reference_keyframe.png")
            self.assertEqual(control["sketch"], "../sketch.png")
            alignment = pipeline.read_json(output_dir / "story_alignment.json")[0]
            self.assertEqual(len(alignment["sampled_frames"]), 6)
            self.assertEqual(alignment["motion_prompt"], "The character moves.")

    def test_shared_reference_and_adjacent_video_pairs(self):
        calls, video_calls, concatenations = [], [], []
        fake_derivative = types.ModuleType("drawvideo.derivative_keyframes")

        class Keyframes:
            def __init__(self, settings):
                pass

            def generate_one(self, conversion_prompt, reference, output):
                calls.append((reference, output))
                Path(output).write_bytes(b"image")

        fake_derivative.DerivativeKeyframesGeneration = Keyframes
        fake_video = types.ModuleType("drawvideo.first_last_frame_video")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            for shot_id in ("shot_001", "shot_002"):
                shot_dir = output_dir / "shots" / shot_id
                pipeline.write_json(shot_dir / "structured_prompts.json", decomposition())
                (shot_dir / "reference_keyframe.png").write_bytes(b"image")

            def generate_video(dynamic, negative, first, last, *, settings):
                video_calls.append((first, last, settings["num_frames"]))
                raw_clip = output_dir / "raw_clip.mp4"
                raw_clip.write_bytes(b"video")
                return str(raw_clip)

            fake_video.generate_video = generate_video

            def concat(paths, output):
                concatenations.append((list(paths), output))
                Path(output).write_bytes(b"concatenated")

            settings = pipeline.read_json(ROOT / "configs/generation.json")
            storyboard_path = ROOT / "examples/storyboard/storyboard.json"
            with patch.dict(sys.modules, {"drawvideo.derivative_keyframes": fake_derivative,
                                          "drawvideo.first_last_frame_video": fake_video}), \
                    patch.object(pipeline, "concat_videos", concat):
                pipeline.run_stage("keyframes", storyboard_path, output_dir, settings)
                pipeline.run_stage("video", storyboard_path, output_dir, settings)
            self.assertEqual(len(calls), 10)
            self.assertTrue(all(Path(reference).name == "reference_keyframe.png"
                                for reference, _ in calls))
            self.assertEqual(len(video_calls), 10)
            for shot_index in range(2):
                shot_calls = video_calls[shot_index * 5:(shot_index + 1) * 5]
                self.assertEqual(Path(shot_calls[0][0]).name, "reference_keyframe.png")
                for index, (first, last, frames) in enumerate(shot_calls, 1):
                    self.assertEqual(Path(last).name, f"derivative_keyframe_{index:02d}.png")
                    if index > 1:
                        self.assertEqual(Path(first).name, f"derivative_keyframe_{index - 1:02d}.png")
                    self.assertEqual(frames, 81)
            self.assertEqual([len(paths) for paths, _ in concatenations], [5, 5, 2])
            manifest = pipeline.read_json(output_dir / "generation_manifest.json")
            for shot in manifest["shots"]:
                self.assertFalse(Path(shot["reference_keyframe"]).is_absolute())
                self.assertFalse(Path(shot["sketch"]).is_absolute())

    @unittest.skipUnless(__import__("shutil").which("ffmpeg"), "ffmpeg is not installed")
    def test_ffmpeg_concatenation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clip = root / "local_clip.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=red:s=64x64:r=16", "-t", "0.5", "-c:v", "mpeg4",
                            str(clip)], check=True)
            output = root / "storyboard_video.mp4"
            pipeline.concat_videos([clip, clip], output)
            result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of", "json", str(output)],
                                    check=True, capture_output=True, text=True)
            self.assertAlmostEqual(float(json.loads(result.stdout)["format"]["duration"]), 1.0, places=2)


class BackendContractTests(unittest.TestCase):
    def test_sampling_parameters_reach_comfyui_nodes(self):
        from PIL import Image

        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext
        with patch.dict(sys.modules, {"torch": fake_torch}):
            modules = [importlib.import_module("drawvideo." + name) for name in (
                "sketch_coloring", "derivative_keyframes", "first_last_frame_video")]
        settings = pipeline.read_json(ROOT / "configs/generation.json")
        records = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (128, 128), "white").save(source)

            class Node:
                def __init__(self, name):
                    self.name = name

                def __getattr__(self, method):
                    def call(**kwargs):
                        records.append((self.name, method, kwargs))
                        if method == "save_images":
                            return {"ui": {"images": [{"filename": "source.png"}]}}
                        if self.name == "SaveVideo":
                            return types.SimpleNamespace(ui=types.SimpleNamespace(
                                values=[{"filename": "source.png"}]))
                        return ("value_0", "value_1", "value_2")
                    return call

            mappings = {}
            for name in ("DualCLIPLoaderGGUF", "CLIPTextEncode", "UnetLoaderGGUF", "VAELoader",
                         "ControlNetLoader", "LoadImage", "FluxGuidance", "AIO_Preprocessor",
                         "ControlNetApplyAdvanced", "EmptySD3LatentImage", "KSampler", "VAEDecode",
                         "SaveImage", "ImageStitch", "FluxKontextImageScale", "VAEEncode",
                         "DualCLIPLoader", "UNETLoader", "ReferenceLatent", "ConditioningZeroOut",
                         "CLIPLoader", "ModelSamplingSD3", "WanFirstLastFrameToVideo",
                         "KSamplerAdvanced", "CreateVideo", "SaveVideo"):
                mappings[name] = type(name, (Node,), {"__init__": lambda self, n=name: Node.__init__(self, n)})
            for module in modules:
                with patch.dict(module.NODE_CLASS_MAPPINGS, mappings, clear=True), \
                        patch.object(module, "initialize_comfyui"), \
                        patch.object(module, "output_file", return_value=str(source)):
                    if module.__name__.endswith("sketch_coloring"):
                        module.SketchColoring(settings["sketch_coloring"]).color_one(
                            "appearance", str(source), str(root / "reference_keyframe.png"))
                    elif module.__name__.endswith("derivative_keyframes"):
                        module.DerivativeKeyframesGeneration(settings["derivative_keyframes_generation"]).generate_one(
                            "conversion", str(source), str(root / "derivative_keyframe_01.png"))
                    else:
                        module.generate_video("dynamic", "negative", str(source), str(source),
                                              settings=settings["first_last_frame_video_generation"])
        samples = [kwargs for name, method, kwargs in records if name == "KSampler" and method == "sample"]
        self.assertEqual([(sample["steps"], sample["cfg"]) for sample in samples], [(15, 7.0), (20, 1.0)])
        preprocessing = [kwargs for name, _, kwargs in records if name == "AIO_Preprocessor"]
        self.assertEqual(preprocessing[0]["resolution"], 1024)
        first_last = [kwargs for name, _, kwargs in records if name == "WanFirstLastFrameToVideo"]
        self.assertEqual(first_last[0]["length"], 81)
        advanced = [kwargs for name, method, kwargs in records if name == "KSamplerAdvanced" and method == "sample"]
        self.assertEqual([(sample["start_at_step"], sample["end_at_step"]) for sample in advanced], [(0, 10), (10, 20)])


if __name__ == "__main__":
    unittest.main()
