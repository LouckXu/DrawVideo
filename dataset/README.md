# SketchLongVideo Utilities

The [dataset download][sketchlongvideo] opens the shared `SketchLongVideo Dataset` folder, containing three source subsets: `self-collected` (20 sequences, 201 triplets), `AnimeShooter` (96 sequences, 932 triplets), and `AI-generated` (10 sequences, 100 triplets).

Each `video_XXX` package aligns files by a shared stem:

```text
SketchLongVideo Dataset/
  self-collected/
  AnimeShooter/
  AI-generated/
    video_122/
      sketch/video_122_keyframe_0001.png
      static_prompt/video_122_keyframe_0001.txt
      story/video_122_keyframe_0001.txt
```

In the paper and public generation API, these text conditions are called **appearance prompt** and **motion prompt**.

Some folders also contain macOS `.DS_Store` metadata files. They are not dataset samples and are ignored by the importer.

## Import a Storyboard

```bash
python dataset/import_storyboard.py \
  --video-package "datasets/SketchLongVideo Dataset/AI-generated/video_122" \
  --output-dir outputs/storyboards/video_122
```

This copies sketches to `sketches/shot_001.png`, etc., writes an ordered `storyboard.json`, and preserves the prompt text and source identifiers. Run generation on that JSON using the main pipeline.

## Dataset Construction Components

`sketch_extraction.py` implements the grayscale, inversion, 3 x 3 erosion, and color-dodge conversion used in the paper:

```bash
python dataset/sketch_extraction.py datasets/color_keyframes -o outputs/extracted_sketches
```

`condition_recognition.py` queries LLaVA-OneVision for subject, style, scene, and action, then organizes them into initial appearance and motion descriptions:

```bash
python -m pip install -e '.[dataset]'
python dataset/condition_recognition.py \
  --input datasets/color_keyframes \
  --output-root outputs/recognized_conditions
```

The recognition output fields are `appearance_prompt` and `motion_prompt`. This is the initial recognition component; it does not replace the complete curated dataset or the subsequent motion-narrative preparation. The inference-time Qwen2.5 module is in `src/drawvideo/structured_prompt_decomposition.py`.

Original third-party videos and model weights are not bundled here. Source media remain subject to their original rights and usage terms.

[sketchlongvideo]: https://unisyd-my.sharepoint.com/:f:/g/personal/chuanzhi_xu_sydney_edu_au/IgDJVV8IRQmbSLkgMj7NnchaAS29iBpAs3t2L4Lkr0-0qLQ?e=BhsbWB
