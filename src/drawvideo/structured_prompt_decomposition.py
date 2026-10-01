"""Qwen2.5 prompt organization for A_k, C_k^i, and D_k^i."""

import json
from urllib import request


SYSTEM_PROMPT = """Organize a storyboard shot for sketch-conditioned video generation.
Return JSON only, using exactly this schema:
{
  "enhanced_appearance_prompt": "appearance and composition description",
  "derivative_keyframes": [
    {
      "index": 1,
      "conversion_prompt": "one frozen action state for image-to-image generation",
      "dynamic_prompt": {
        "positive": "visual description of the transition",
        "animation_action": "action from the previous state to this state",
        "body_motion": "body movement during the transition",
        "facial_expression_change": "expression change during the transition",
        "style_consistency": "retain the reference image style"
      }
    }
  ]
}
Enhance the appearance prompt with the supplied identity, clothing, scene,
lighting, camera composition and style. Preserve the provided content.
Decompose the motion prompt into exactly five ordered action states, indexed
1, 2, 3, 4, 5. Follow the actual supplied motion; do not impose an unrelated
action sequence. Each conversion prompt describes one still keyframe.
Each dynamic prompt describes the transition from the previous state to the
corresponding keyframe; state 1 follows the colored reference keyframe.
Preserve the same subject, props, scene and camera within the shot. Do not add
characters, objects, events or effects absent from the supplied conditions.
All five derivative images will share the same reference image, not the
previous derivative image. Use English. No Markdown or explanation.
"""


def validate_decomposition(payload):
    if not isinstance(payload.get("enhanced_appearance_prompt"), str):
        raise ValueError("Missing enhanced_appearance_prompt.")
    if not payload["enhanced_appearance_prompt"].strip():
        raise ValueError("enhanced_appearance_prompt must not be empty.")
    derivative_keyframes = payload.get("derivative_keyframes", [])
    if [entry.get("index") for entry in derivative_keyframes] != [1, 2, 3, 4, 5]:
        raise ValueError("Expected exactly five ordered derivative keyframes (1-5).")
    for entry in derivative_keyframes:
        if not isinstance(entry.get("conversion_prompt"), str) or not entry["conversion_prompt"].strip():
            raise ValueError("Each derivative keyframe needs a conversion_prompt.")
        dynamic_prompt = entry.get("dynamic_prompt", {})
        for field in ("positive", "animation_action", "body_motion",
                      "facial_expression_change", "style_consistency"):
            if not isinstance(dynamic_prompt.get(field), str) or not dynamic_prompt[field].strip():
                raise ValueError(f"Missing dynamic_prompt.{field}.")
    return payload


def decompose_prompts(appearance_prompt, motion_prompt, settings):
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "appearance_prompt": appearance_prompt,
                "motion_prompt": motion_prompt,
            })},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": settings["temperature"], "num_predict": 2500},
    }
    endpoint = settings["ollama_url"].rstrip("/") + "/api/chat"
    req = request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=settings["timeout_seconds"]) as response:
        result = json.load(response)
    return validate_decomposition(json.loads(result["message"]["content"]))


def build_dynamic_prompt(dynamic_prompt):
    return " ".join(dynamic_prompt[field].strip() for field in (
        "positive", "animation_action", "body_motion",
        "facial_expression_change", "style_consistency",
    ))
