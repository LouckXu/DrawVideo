import os
import random
import shutil
import torch

from .comfyui_runtime import NODE_CLASS_MAPPINGS, initialize_comfyui, get_value_at_index, output_file

def compute_output_dimensions(
    orig_w: int,
    orig_h: int,
    max_exclusive: int = 600,
    multiple: int = 16,
) -> tuple[int, int]:
    """
    Scale the image proportionally so that both width and height are strictly
    less than max_exclusive, then round down each dimension to the nearest
    multiple of 'multiple' (required by the FLUX latent grid).
    """
    if orig_w < 1 or orig_h < 1:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")
    scale = min(1.0, (max_exclusive - 1) / orig_w, (max_exclusive - 1) / orig_h)
    wf = orig_w * scale
    hf = orig_h * scale
    w = max(multiple, (int(wf) // multiple) * multiple)
    h = max(multiple, (int(hf) // multiple) * multiple)
    while w >= max_exclusive:
        w -= multiple
    while h >= max_exclusive:
        h -= multiple
    return max(multiple, w), max(multiple, h)


class SketchColoring:
    """
    Loads FLUX + ControlNet Canny models once and reuses them across multiple
    colour_one() calls, suitable for batch colouring of sketch keyframes.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self._loaded = False
        self.text_encoders = None
        self.image_model = None
        self.vae = None
        self.controlnet = None
        self.cliptextencode = None
        self.loadimage = None
        self.fluxguidance = None
        self.aio_preprocessor = None
        self.controlnetapplyadvanced = None
        self.emptysd3latentimage = None
        self.ksampler = None
        self.vaedecode = None
        self.saveimage = None

    def setup(self) -> None:
        if self._loaded:
            return
        initialize_comfyui()
        with torch.inference_mode():
            dualcliploadergguf = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
            self.text_encoders = dualcliploadergguf.load_clip(
                clip_name1=self.settings["models"]["text_encoder"],
                clip_name2=self.settings["models"]["clip_encoder"],
                type="flux",
            )

            self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            self.image_model = unetloadergguf.load_unet(
                unet_name=self.settings["models"]["diffusion_model"]
            )

            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            self.vae = vaeloader.load_vae(vae_name=self.settings["models"]["vae"])

            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            self.controlnet = controlnetloader.load_controlnet(
                control_net_name=self.settings["models"]["controlnet"]
            )

            self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
            self.controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            self.emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
            self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            self.saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
        self._loaded = True
        print("Models and nodes loaded. Starting batch colouring.")

    def color_one(
        self,
        enhanced_appearance_prompt: str,
        sketch_path: str,
        reference_keyframe_path: str,
    ) -> str:
        self.setup()
        sketch_path = os.path.abspath(sketch_path)

        from PIL import Image as _PILImage
        with _PILImage.open(sketch_path) as _img:
            _orig_w, _orig_h = _img.size
        out_w, out_h = compute_output_dimensions(
            _orig_w, _orig_h, max_exclusive=self.settings["max_dimension_exclusive"], multiple=16)
        canny_res = self.settings["preprocessing_resolution"]
        print(f"Input: {_orig_w}x{_orig_h}  ->  Output: {out_w}x{out_h}  (canny_res={canny_res})")

        with torch.inference_mode():
            appearance_conditioning = self.cliptextencode.encode(
                text=enhanced_appearance_prompt,
                clip=get_value_at_index(self.text_encoders, 0),
            )
            negative_conditioning = self.cliptextencode.encode(
                text="", clip=get_value_at_index(self.text_encoders, 0)
            )

            sketch_image = self.loadimage.load_image(image=sketch_path)

            guided_appearance = self.fluxguidance.EXECUTE_NORMALIZED(
                guidance=self.settings["guidance"], conditioning=get_value_at_index(appearance_conditioning, 0)
            )

            sketch_edges = self.aio_preprocessor.execute(
                preprocessor="CannyEdgePreprocessor",
                resolution=canny_res,
                image=get_value_at_index(sketch_image, 0),
            )

            structural_conditioning = self.controlnetapplyadvanced.apply_controlnet(
                strength=self.settings["controlnet_strength"],
                start_percent=self.settings["control_start"],
                end_percent=self.settings["control_end"],
                positive=get_value_at_index(guided_appearance, 0),
                negative=get_value_at_index(negative_conditioning, 0),
                control_net=get_value_at_index(self.controlnet, 0),
                image=get_value_at_index(sketch_edges, 0),
                vae=get_value_at_index(self.vae, 0),
            )

            reference_latent = self.emptysd3latentimage.EXECUTE_NORMALIZED(
                width=out_w, height=out_h, batch_size=1,
            )

            reference_samples = self.ksampler.sample(
                seed=random.randint(1, 2**64 - 1),
                steps=self.settings["steps"],
                cfg=self.settings["cfg"],
                sampler_name=self.settings["sampler"],
                scheduler=self.settings["scheduler"],
                denoise=self.settings["denoise"],
                model=get_value_at_index(self.image_model, 0),
                positive=get_value_at_index(structural_conditioning, 0),
                negative=get_value_at_index(structural_conditioning, 1),
                latent_image=get_value_at_index(reference_latent, 0),
            )

            reference_image = self.vaedecode.decode(
                samples=get_value_at_index(reference_samples, 0),
                vae=get_value_at_index(self.vae, 0),
            )

            saved_reference = self.saveimage.save_images(
                filename_prefix="reference_keyframe",
                images=get_value_at_index(reference_image, 0),
            )

            saved_image = saved_reference["ui"]["images"][0]
            saved_path = output_file(saved_image["filename"], saved_image.get("subfolder", ""))
            final_path = os.path.abspath(reference_keyframe_path)

            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.copy(saved_path, final_path)
            print(f"Saved: {final_path}")
            return final_path
