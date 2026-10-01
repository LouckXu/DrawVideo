# Setup

## Environment

Use Python 3.10 or newer in the CUDA-enabled environment used for ComfyUI. Install [ComfyUI](https://github.com/Comfy-Org/ComfyUI) following its upstream instructions, including its PyTorch and Python dependencies. The experiments in the paper used an NVIDIA RTX 5090; hardware requirements vary with model precision and ComfyUI's memory management.

Install these custom node packages into `ComfyUI/custom_nodes/`, with their dependencies:

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF): `UnetLoaderGGUF`, `DualCLIPLoaderGGUF`.
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux): `AIO_Preprocessor`, `CannyEdgePreprocessor`.

The remaining nodes are supplied by ComfyUI. The pipeline requires `ImageStitch`, `FluxKontextImageScale`, `ReferenceLatent`, `WanFirstLastFrameToVideo`, `CreateVideo`, and `SaveVideo` in addition to the standard loaders and samplers.

From the DrawVideo repository root:

```bash
python -m pip install -e .
export COMFYUI_PATH="$(pwd)/../ComfyUI"
```

Set `COMFYUI_PATH` to your own installation directory if it is not adjacent to DrawVideo. Paths are supplied at runtime, never embedded in source files. No ComfyUI web server is required: the generation stages call its Python nodes directly.

Install [Ollama](https://ollama.com/) separately, start its service, and obtain the text model:

```bash
ollama pull qwen2.5:7b
```

The endpoint, model tag, and temperature are configured under `structured_prompt_decomposition` in `configs/generation.json`.

Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are available on your command path. They are used for video concatenation and inspection.

## Model Files

The following filenames are the default ComfyUI loader names in `configs/generation.json`. Obtain compatible weights from their publishers or quantization providers and follow their terms. Model names in the config can be changed to match your installed files; changing precision or weights changes the generation configuration.

| Stage | ComfyUI model directory | Configured filename |
| --- | --- | --- |
| Sketch Coloring | `diffusion_models` | `FLUX1/flux1-dev-F16.gguf` |
| Sketch Coloring | `text_encoders` | `t5/t5-v1_1-xxl-encoder-Q4_K_S.gguf` |
| FLUX text encoding | `text_encoders` | `clip_l.safetensors` |
| FLUX VAE | `vae` | `ae.safetensors` |
| Sketch Coloring | `controlnet` | `flux-canny-controlnet-v3.safetensors` |
| Derivative Keyframes Generation | `diffusion_models` | `flux1-dev-kontext_fp8_scaled.safetensors` |
| Derivative Keyframes Generation | `text_encoders` | `t5/t5xxl_fp16.safetensors` |
| Video Generation | `diffusion_models` | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` |
| Video Generation | `diffusion_models` | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` |
| Video Generation | `text_encoders` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| Video Generation | `vae` | `wan_2.1_vae.safetensors` |

ComfyUI's `extra_model_paths.yaml` is supported for model directories outside its installation.

## Sampling Settings

| Setting | Sketch Coloring | Derivative Keyframes Generation | First-last-frame Video Generation |
| --- | --- | --- | --- |
| Steps | 15 | 20 | 20 |
| CFG | 7.0 | 1.0 | 4.0 |
| FLUX guidance | 3.5 | 2.5 | - |
| Sampler | DPM++ 2M | Euler | Euler |
| Scheduler | SGM uniform | simple | simple |
| Denoising strength | 0.8 | 1.0 | - |

Canny preprocessing uses resolution 1024 and ControlNet strength 0.95 over the full sampling range. Reference images retain their aspect ratio with dimensions below 600 and aligned to a 16-pixel latent grid. Video defaults are 640 x 480, 81 frames, 16 FPS, and shift 8.0. The high-noise stage runs steps 0-10; the low-noise stage runs steps 10-20. For square outputs, set `height` to 640. Each shot uses five prompt pairs and five local clips.

Generation seeds are sampled per image or clip. Separate stage processes release model memory between the image and video backbones.

## Lightweight Checks

Without model weights, run the input plan and unit tests:

```bash
python -m drawvideo generate --storyboard examples/storyboard/storyboard.json --dry-run
python -m unittest discover -s tests -v
```

These checks validate orchestration and parameter wiring, not generated video quality. The backend tests replace model nodes with lightweight test doubles.
