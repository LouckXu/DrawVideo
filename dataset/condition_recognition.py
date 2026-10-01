from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: transformers. "
        "Please install the required packages and activate the correct Python environment."
    ) from exc


MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "input"
DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"
STYLE_BIAS = (
    "classic 2D cartoon animation frame, clean colored lineart, flat cel-style coloring, "
    "solid local colors, minimal shading"
)

SUBJECT_PROMPT = (
    "Describe the main visible subject without guessing any franchise or person name. "
    "Mention visible face shape, eyebrows, hair, skin tone, clothes, and clear colors in 1 to 2 sentences."
)
STYLE_PROMPT = (
    "Describe the visual rendering style of this image in one short sentence. "
    "Mention whether it looks like a 2D cartoon frame, anime frame, flat cel coloring, or simple colored line art. "
    "Focus on visible style only."
)
SCENE_PROMPT = (
    "Describe only the background and framing in 1 to 2 sentences. "
    "Mention major colors, large background marks or objects, and whether the shot is centered or close-up. "
    "Do not guess a franchise name. Do not describe the character, clothes, hands, or emotion."
)
ACTION_PROMPT = (
    "Describe only the pose, action, and apparent emotion of the main visible subject in 1 short sentence. "
    "Start with a verb phrase such as 'Standing ...', 'Looking ...', or 'Holding ...'. "
    "Focus only on this single image, avoid off-screen speculation, and do not mention the background."
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\s*assistant\s*[:：-]?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _ensure_sentence(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _split_sentences(text: str) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _drop_sentences_with_keywords(text: str, keywords: set[str]) -> str:
    kept: list[str] = []
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _remove_background_clauses(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"\s+and is standing against\s+[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+standing against\s+[^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+against\s+[^.]+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _to_prompt_fragment(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    text = re.sub(r"^The main visible subject is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^The subject is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^The image is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^This image is\s+", "", text, flags=re.IGNORECASE)
    text = text.replace(". ", ", ")
    return text.strip(" .,\t")


def _extract_background_phrase(scene_text: str) -> str:
    scene_text = _clean_text(scene_text)
    if not scene_text:
        return "a simple background"

    sentences = _split_sentences(scene_text)
    first_sentence = sentences[0] if sentences else scene_text
    standing_match = re.search(r"standing against\s+(.+?)(?:[.!?]|$)", first_sentence, flags=re.IGNORECASE)
    if standing_match:
        phrase = standing_match.group(1).strip(" .,\t")
        return phrase or "a simple background"

    first_sentence = re.sub(r"^The background is\s+", "", first_sentence, flags=re.IGNORECASE)
    first_sentence = re.sub(r"^Background:\s*", "", first_sentence, flags=re.IGNORECASE)
    first_sentence = re.split(r",?\s+and the subject\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    first_sentence = re.split(r",?\s+and the character\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    first_sentence = re.split(r",?\s+with the subject\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    first_sentence = re.split(r",?\s+with the character\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    first_sentence = re.split(r",?\s+while the subject\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    first_sentence = re.split(r",?\s+while the character\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    first_sentence = first_sentence.strip(" .,\t")
    return first_sentence or "a simple background"


def _extract_composition_phrase(scene_text: str) -> str:
    for sentence in _split_sentences(scene_text):
        lowered = sentence.lower()
        if any(token in lowered for token in {"centered", "close-up", "close up", "framing", "shot", "foreground", "focus"}):
            if "not in focus" in lowered:
                return "centered framing"
            return sentence.strip(" .,\t")
    return ""


def _derive_subject_label(subject_text: str) -> str:
    lowered = subject_text.lower()
    candidates = [
        ("cartoon child", "The cartoon child"),
        ("cartoon boy", "The cartoon boy"),
        ("cartoon girl", "The cartoon girl"),
        ("cartoon character", "The cartoon character"),
        ("child", "The child"),
        ("boy", "The boy"),
        ("girl", "The girl"),
        ("man", "The man"),
        ("woman", "The woman"),
        ("character", "The character"),
    ]
    for needle, label in candidates:
        if needle in lowered:
            return label
    return "The main character"


def _build_appearance_prompt(analysis: dict[str, Any]) -> str:
    subject_fragment = _to_prompt_fragment(_remove_background_clauses(analysis["subject_description"]))
    style_fragment = _to_prompt_fragment(analysis["style_description"])
    background_fragment = f"background with {analysis['background_phrase']}"
    composition_fragment = _to_prompt_fragment(_extract_composition_phrase(analysis["scene_description"]))

    parts = [
        subject_fragment or "main character",
        style_fragment or STYLE_BIAS,
        STYLE_BIAS if style_fragment else "",
        background_fragment,
        composition_fragment,
        "preserve the same composition and camera angle",
        "keep the same subject placement",
        "clear local colors",
        "consistent palette",
        "high quality 2D frame look",
        "no extra characters or objects",
    ]
    parts = [part for part in parts if part]
    return _ensure_sentence(", ".join(parts))


def _build_motion_prompt(analysis: dict[str, Any]) -> str:
    subject_label = analysis["subject_label"]
    background_phrase = analysis["background_phrase"]
    action_text = _drop_sentences_with_keywords(analysis["action_description"], {"background"})
    action_text = re.sub(r"^The main visible subject is\s+", "", action_text, flags=re.IGNORECASE)
    action_text = re.sub(r"^The subject is\s+", "", action_text, flags=re.IGNORECASE)
    action_text = action_text[:1].upper() + action_text[1:] if action_text else action_text
    action_sentence = _ensure_sentence(action_text)
    intro = _ensure_sentence(f"{subject_label} appears in the frame against {background_phrase}")
    return " ".join(part for part in [intro, action_sentence] if part).strip()


def _resolve_device(preferred_device: str) -> str:
    if preferred_device != "auto":
        return preferred_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(
    model_name: str,
    device: str,
    hf_home: Path,
) -> tuple[AutoProcessor, LlavaOnevisionForConditionalGeneration]:
    os.environ.setdefault("HF_HOME", str(hf_home))
    hf_home.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_name)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return processor, model


def _generate_answer(
    image: Image.Image,
    prompt_text: str,
    device: str,
    processor: AutoProcessor,
    model: LlavaOnevisionForConditionalGeneration,
    max_new_tokens: int = 120,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_only = output_ids[:, prompt_length:]
    decoded = processor.batch_decode(generated_only, skip_special_tokens=True)[0]
    return _clean_text(decoded)


def _analyze_one_image(
    image_path: Path,
    device: str,
    processor: AutoProcessor,
    model: LlavaOnevisionForConditionalGeneration,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    subject_description = _generate_answer(image, SUBJECT_PROMPT, device, processor, model)
    style_description = _generate_answer(image, STYLE_PROMPT, device, processor, model)
    scene_description = _generate_answer(image, SCENE_PROMPT, device, processor, model)
    action_description = _generate_answer(image, ACTION_PROMPT, device, processor, model)

    return {
        "image_name": image_path.name,
        "subject_description": _ensure_sentence(subject_description),
        "style_description": _ensure_sentence(style_description),
        "scene_description": _ensure_sentence(scene_description),
        "action_description": _ensure_sentence(action_description),
        "subject_label": _derive_subject_label(subject_description),
        "background_phrase": _extract_background_phrase(scene_description),
    }


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported images found in: {input_path}")
        return images

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _resolve_runtime_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _build_output_path(image_path: Path, output_root: Path) -> Path:
    try:
        relative_image = image_path.relative_to(INPUT_ROOT)
        return output_root / relative_image.with_suffix(".json")
    except ValueError:
        return output_root / image_path.parent.name / f"{image_path.stem}.json"


def run_recognition(
    input_path: str | Path,
    output_root: str | Path = "output",
    model_name: str = MODEL_NAME,
    hf_home: str | Path = DEFAULT_HF_HOME,
    device: str = "auto",
    max_images: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_path = _resolve_runtime_path(input_path)
    output_root = _resolve_runtime_path(output_root)
    hf_home = Path(hf_home)

    images = _collect_images(input_path)
    if max_images is not None:
        images = images[:max_images]

    resolved_device = _resolve_device(device)
    processor, model = _load_model(model_name, resolved_device, hf_home)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    manifest_items: list[dict[str, str]] = []

    for image_path in tqdm(images, desc="Generating appearance and motion prompts"):
        output_path = _build_output_path(image_path, output_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            manifest_items.append(
                {
                    "image_path": str(image_path),
                    "output_json": str(output_path),
                    "status": "skipped_existing",
                }
            )
            continue

        analysis = _analyze_one_image(image_path, resolved_device, processor, model)
        appearance_prompt = _build_appearance_prompt(analysis)
        motion_prompt = _build_motion_prompt(analysis)

        result = {
            "image_id": image_path.stem,
            "appearance_prompt": appearance_prompt,
            "motion_prompt": motion_prompt,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

        manifest_items.append(
            {
                "image_path": str(image_path),
                "output_json": str(output_path),
                "status": "generated",
            }
        )

    return {
        "input_path": str(input_path),
        "output_root": str(output_root),
        "model_name": model_name,
        "device": resolved_device,
        "hf_home": str(hf_home),
        "num_images": len(manifest_items),
        "items": manifest_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate appearance and motion prompts for one image or one folder of images."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image file or one folder containing images.",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Directory for output JSON files.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Vision-language model id.",
    )
    parser.add_argument(
        "--hf-home",
        default=str(DEFAULT_HF_HOME),
        help="HuggingFace model cache directory.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for debugging batch runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_recognition(
        input_path=args.input,
        output_root=args.output_root,
        model_name=args.model_name,
        hf_home=args.hf_home,
        device=args.device,
        max_images=args.max_images,
        overwrite=args.overwrite,
    )
    print(f"Processed {manifest['num_images']} image(s)")
    print(f"Saved outputs to {manifest['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
