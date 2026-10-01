"""Shared initialization for the three ComfyUI generation stages."""

import asyncio
import os
from pathlib import Path
import sys

NODE_CLASS_MAPPINGS = {}
_initialized = False


def get_value_at_index(value, index):
    if isinstance(value, dict):
        return value["result"][index]
    return value[index]


def initialize_comfyui():
    global _initialized
    if _initialized:
        return
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        raise RuntimeError("Set COMFYUI_PATH to your ComfyUI installation directory.")
    comfyui_root = Path(comfyui_path).expanduser().resolve()
    if not (comfyui_root / "nodes.py").is_file():
        raise FileNotFoundError("COMFYUI_PATH does not contain a ComfyUI installation.")
    sys.path.insert(0, str(comfyui_root))
    os.chdir(comfyui_root)
    saved_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        import execution
        import nodes
        import server
        from utils.extra_config import load_extra_path_config

        extra_model_paths = comfyui_root / "extra_model_paths.yaml"
        if extra_model_paths.is_file():
            load_extra_path_config(str(extra_model_paths))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        prompt_server = server.PromptServer(loop)
        execution.PromptQueue(prompt_server)
        loop.run_until_complete(nodes.init_extra_nodes())
        NODE_CLASS_MAPPINGS.update(nodes.NODE_CLASS_MAPPINGS)
        _initialized = True
    finally:
        sys.argv = saved_argv


def output_file(filename, subfolder=""):
    import folder_paths

    return os.path.join(folder_paths.get_output_directory(), subfolder, filename)
