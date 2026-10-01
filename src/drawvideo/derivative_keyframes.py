import os
import random
import shutil
import torch

from .comfyui_runtime import NODE_CLASS_MAPPINGS, initialize_comfyui, get_value_at_index, output_file

class DerivativeKeyframesGeneration:
    """
    Loads FLUX.1 Kontext models once and generates derivative keyframes
    from a single reference image by reusing the same model instance.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self._loaded = False
        self.loadimage = None
        self.imagestitch = None
        self.fluxkontextimagescale = None
        self.vae = None
        self.vaeencode = None
        self.text_encoders = None
        self.cliptextencode = None
        self.keyframe_model = None
        self.referencelatent = None
        self.fluxguidance = None
        self.conditioningzeroout = None
        self.ksampler = None
        self.vaedecode = None
        self.saveimage = None

    def setup(self) -> None:
        if self._loaded:
            return
        initialize_comfyui()
        with torch.inference_mode():
            self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            self.imagestitch = NODE_CLASS_MAPPINGS["ImageStitch"]()
            self.fluxkontextimagescale = NODE_CLASS_MAPPINGS["FluxKontextImageScale"]()

            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            self.vae = vaeloader.load_vae(vae_name=self.settings["models"]["vae"])

            self.vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()

            dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            self.text_encoders = dualcliploader.load_clip(
                clip_name1=self.settings["models"]["clip_encoder"],
                clip_name2=self.settings["models"]["text_encoder"],
                type="flux",
                device="default",
            )

            self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

            unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            self.keyframe_model = unetloader.load_unet(
                unet_name=self.settings["models"]["diffusion_model"],
                weight_dtype="default",
            )

            self.referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
            self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            self.saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        self._loaded = True
        print("FLUX Kontext models loaded.")

    def generate_one(
        self,
        conversion_prompt: str,
        reference_keyframe_path: str,
        derivative_keyframe_path: str,
    ) -> str:
        self.setup()
        src_abs = os.path.abspath(reference_keyframe_path)

        with torch.inference_mode():
            reference_image = self.loadimage.load_image(image=src_abs)

            prepared_reference = self.imagestitch.EXECUTE_NORMALIZED(
                direction="right",
                match_image_size=True,
                spacing_width=0,
                spacing_color="white",
                image1=get_value_at_index(reference_image, 0),
            )

            scaled_reference = self.fluxkontextimagescale.EXECUTE_NORMALIZED(
                image=get_value_at_index(prepared_reference, 0)
            )

            reference_latent = self.vaeencode.encode(
                pixels=get_value_at_index(scaled_reference, 0),
                vae=get_value_at_index(self.vae, 0),
            )

            conversion_conditioning = self.cliptextencode.encode(
                text=conversion_prompt,
                clip=get_value_at_index(self.text_encoders, 0),
            )

            reference_conditioning = self.referencelatent.EXECUTE_NORMALIZED(
                conditioning=get_value_at_index(conversion_conditioning, 0),
                latent=get_value_at_index(reference_latent, 0),
            )

            guided_conversion = self.fluxguidance.EXECUTE_NORMALIZED(
                guidance=self.settings["guidance"],
                conditioning=get_value_at_index(reference_conditioning, 0),
            )

            negative_conditioning = self.conditioningzeroout.zero_out(
                conditioning=get_value_at_index(conversion_conditioning, 0)
            )

            derivative_samples = self.ksampler.sample(
                seed=random.randint(1, 2**64 - 1),
                steps=self.settings["steps"],
                cfg=self.settings["cfg"],
                sampler_name=self.settings["sampler"],
                scheduler=self.settings["scheduler"],
                denoise=self.settings["denoise"],
                model=get_value_at_index(self.keyframe_model, 0),
                positive=get_value_at_index(guided_conversion, 0),
                negative=get_value_at_index(negative_conditioning, 0),
                latent_image=get_value_at_index(reference_latent, 0),
            )

            derivative_image = self.vaedecode.decode(
                samples=get_value_at_index(derivative_samples, 0),
                vae=get_value_at_index(self.vae, 0),
            )

            saved_derivative = self.saveimage.save_images(
                filename_prefix="derivative_keyframe",
                images=get_value_at_index(derivative_image, 0),
            )

            saved_image = saved_derivative["ui"]["images"][0]
            saved_path = output_file(saved_image["filename"], saved_image.get("subfolder", ""))

            os.makedirs(os.path.dirname(derivative_keyframe_path), exist_ok=True)
            shutil.copy(saved_path, derivative_keyframe_path)
            print(f"Saved: {derivative_keyframe_path}")
            return derivative_keyframe_path
