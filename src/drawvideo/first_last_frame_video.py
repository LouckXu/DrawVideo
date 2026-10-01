import os
import random
import shutil
import torch

from .comfyui_runtime import NODE_CLASS_MAPPINGS, initialize_comfyui, get_value_at_index, output_file

def generate_video(
    dynamic_prompt: str,
    negative_prompt: str,
    first_keyframe: str,
    last_keyframe: str,
    *,
    settings: dict,
) -> str:
    """
    Generate a video from a first frame and last frame using Wan 2.2 FLF2V.

    Args:
        dynamic_prompt: Structured dynamic prompt D_k^i.
        negative_prompt: Negative conditioning text.
        first_keyframe: First keyframe I_k^{i-1}.
        last_keyframe: Last keyframe I_k^i.
        settings: Wan2.2 model names and sampling parameters.

    Returns:
        Absolute path to the generated video file, or empty string on failure.
    """
    initialize_comfyui()
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        text_encoder = cliploader.load_clip(
            clip_name=settings["models"]["text_encoder"],
            type="wan",
            device="default",
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        high_noise_model = unetloader.load_unet(
            unet_name=settings["models"]["high_noise_model"],
            weight_dtype="default",
        )

        low_noise_model = unetloader.load_unet(
            unet_name=settings["models"]["low_noise_model"],
            weight_dtype="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        negative_conditioning = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(text_encoder, 0),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae = vaeloader.load_vae(vae_name=settings["models"]["vae"])

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        first_keyframe_image = loadimage.load_image(image=first_keyframe)
        last_keyframe_image = loadimage.load_image(image=last_keyframe)

        dynamic_conditioning = cliptextencode.encode(
            text=dynamic_prompt,
            clip=get_value_at_index(text_encoder, 0),
        )

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        wanfirstlastframetovideo = NODE_CLASS_MAPPINGS["WanFirstLastFrameToVideo"]()
        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        SaveVideoClass = NODE_CLASS_MAPPINGS["SaveVideo"]
        savevideo = SaveVideoClass()

        # SaveVideo is a new-style ComfyNode; inject a stub so metadata saving
        # does not crash when called outside the ComfyUI server context.
        SaveVideoClass.hidden = type("_Hidden", (), {"extra_pnginfo": None, "prompt": None})()

        for q in range(1):
            high_noise_sampling = modelsamplingsd3.patch(
                shift=settings["shift"], model=get_value_at_index(high_noise_model, 0)
            )

            low_noise_sampling = modelsamplingsd3.patch(
                shift=settings["shift"], model=get_value_at_index(low_noise_model, 0)
            )

            keyframe_conditioning = wanfirstlastframetovideo.EXECUTE_NORMALIZED(
                width=settings["width"],
                height=settings["height"],
                length=settings["num_frames"],
                batch_size=1,
                positive=get_value_at_index(dynamic_conditioning, 0),
                negative=get_value_at_index(negative_conditioning, 0),
                vae=get_value_at_index(vae, 0),
                start_image=get_value_at_index(first_keyframe_image, 0),
                end_image=get_value_at_index(last_keyframe_image, 0),
            )

            high_noise_samples = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64 - 1),
                steps=settings["steps"],
                cfg=settings["cfg"],
                sampler_name=settings["sampler"],
                scheduler=settings["scheduler"],
                start_at_step=0,
                end_at_step=settings["high_noise_end_step"],
                return_with_leftover_noise="enable",
                model=get_value_at_index(high_noise_sampling, 0),
                positive=get_value_at_index(keyframe_conditioning, 0),
                negative=get_value_at_index(keyframe_conditioning, 1),
                latent_image=get_value_at_index(keyframe_conditioning, 2),
            )

            low_noise_samples = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64 - 1),
                steps=settings["steps"],
                cfg=settings["cfg"],
                sampler_name=settings["sampler"],
                scheduler=settings["scheduler"],
                start_at_step=settings["high_noise_end_step"],
                end_at_step=settings["steps"],
                return_with_leftover_noise="disable",
                model=get_value_at_index(low_noise_sampling, 0),
                positive=get_value_at_index(keyframe_conditioning, 0),
                negative=get_value_at_index(keyframe_conditioning, 1),
                latent_image=get_value_at_index(high_noise_samples, 0),
            )

            clip_frames = vaedecode.decode(
                samples=get_value_at_index(low_noise_samples, 0),
                vae=get_value_at_index(vae, 0),
            )

            local_clip = createvideo.EXECUTE_NORMALIZED(
                fps=settings["fps"], images=get_value_at_index(clip_frames, 0)
            )

            saved_clip = savevideo.EXECUTE_NORMALIZED(
                filename_prefix="video/local_clip",
                format="auto",
                codec="auto",
                video=get_value_at_index(local_clip, 0),
            )

    try:
        saved_result = saved_clip.ui.values[0]
        filename = saved_result["filename"]
        subfolder = saved_result.get("subfolder", "")
        saved_path = output_file(filename, subfolder)
        print(f"Video saved: {saved_path}")
        return saved_path
    except Exception as e:
        print(f"Could not extract video path: {e}")
    return ""
