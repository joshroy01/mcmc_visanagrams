import html
import inspect
import re
import urllib.parse as ul
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from diffusers.loaders import LoraLoaderMixin
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    is_accelerate_available,
    is_accelerate_version,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
# from .if_pipeline_output import IFPipelineOutput
from .if_safety_checker import IFSafetyChecker
from .if_watermarker import IFWatermarker
from ..utils.latents import extract_latents_stage_2, apply_views_to_latents
from ..utils.output import make_canvas_stage_2

if TYPE_CHECKING:
    from mcmc_visanagrams.context import ContextList

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
        >>> from diffusers.utils import pt_to_pil
        >>> import torch

        >>> pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        >>> image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

        >>> # save intermediate image
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        ... )
        >>> super_res_1_pipe.enable_model_cpu_offload()

        >>> image = super_res_1_pipe(
        ...     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds
        ... ).images
        >>> image[0].save("./if_stage_II.png")
        ```
"""


class IFSuperResolutionPipeline(DiffusionPipeline, LoraLoaderMixin):
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    image_noising_scheduler: DDPMScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]

    watermarker: Optional[IFWatermarker]

    bad_punct_regex = re.compile(r"[" + "#®•©™&@·º½¾¿¡§~" + r"\)" + r"\(" + r"\]" + r"\[" + r"\}" +
                                 r"\{" + r"\|" + "\\" + r"\/" + r"\*" + r"]{1,}")  # noqa

    _optional_components = [
        "tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"
    ]
    model_cpu_offload_seq = "text_encoder->unet"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        image_noising_scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        if unet.config.in_channels != 6:
            logger.warning(
                "It seems like you have loaded a checkpoint that shall not be used for super resolution from {unet.config._name_or_path} as it accepts {unet.config.in_channels} input channels instead of 6. Please make sure to pass a super resolution checkpoint as the `'unet'`: IFSuperResolutionPipeline.from_pretrained(unet=super_resolution_unet, ...)`."
            )

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            image_noising_scheduler=image_noising_scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.

        NOTE(dylan.colli): I'm unsure if this method is necessary. However, the image tapestry
        notebook implementation of the IFPipeline redefines this method so I'm including it in the
        super-resolution stage as well.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        models = [
            self.text_encoder,
            self.unet,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative
        execution of the `unet`.

        NOTE(dylan.colli): I'm unsure if this method is necessary. However, the image tapestry
        notebook implementation of the IFPipeline redefines this method so I'm including it in the
        super-resolution stage as well.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache(
            )  # otherwise we don't see the memory savings (but they probably exist)

        hook = None

        if self.text_encoder is not None:
            _, hook = cpu_offload_with_hook(self.text_encoder, device, prev_module_hook=hook)

            # Accelerate will move the next model to the device _before_ calling the offload hook of the
            # previous model. This will cause both models to be present on the device at the same time.
            # IF uses T5 for its text encoder which is really large. We can manually call the offload
            # hook for the text encoder to ensure it's moved to the cpu before the unet is moved to
            # the GPU.
            self.text_encoder_offload_hook = hook

        _, hook = cpu_offload_with_hook(self.unet, device, prev_module_hook=hook)

        # if the safety checker isn't called, `unet_offload_hook` will have to be called to manually offload the unet
        self.unet_offload_hook = hook

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.remove_all_hooks
    def remove_all_hooks(self):
        if is_accelerate_available():
            from accelerate.hooks import remove_hook_from_module
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        for model in [self.text_encoder, self.unet, self.safety_checker]:
            if model is not None:
                remove_hook_from_module(model, recurse=True)

        self.unet_offload_hook = None
        self.text_encoder_offload_hook = None
        self.final_offload_hook = None

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warning("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.

        NOTE(dylan.colli): I'm unsure if this method is necessary. However, the image tapestry
        notebook implementation of the IFPipeline redefines this method so I'm including it in the
        super-resolution stage as well.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
        """
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest",
                                             return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1:-1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}")

            attention_mask = text_inputs.attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt,
                                                                 seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image),
                                                          return_tensors="pt").to(device)
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()

        return image, nsfw_detected, watermark_detected

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        batch_size,
        noise_level,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (callback_steps
                is None) or (callback_steps is not None and
                             (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}.")

        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level`: {noise_level} must be a valid timestep in `self.noising_scheduler`, [0, {self.image_noising_scheduler.config.num_train_timesteps})"
            )

        if isinstance(image, list):
            check_image_type = image[0]
        else:
            check_image_type = image

        if (not isinstance(check_image_type, torch.Tensor)
                and not isinstance(check_image_type, PIL.Image.Image)
                and not isinstance(check_image_type, np.ndarray)):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, or List[...] but is"
                f" {type(check_image_type)}")

        if isinstance(image, list):
            image_batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            image_batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image):
            image_batch_size = 1
        elif isinstance(image, np.ndarray):
            image_batch_size = image.shape[0]
        else:
            assert False

        if batch_size != image_batch_size:
            raise ValueError(
                f"image batch size: {image_batch_size} must be same as prompt batch size {batch_size}"
            )

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline.prepare_intermediate_images
    def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, device,
                                    generator):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        intermediate_images = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        return intermediate_images

    def preprocess_image(self, image, num_images_per_prompt, device):
        if not isinstance(image, torch.Tensor) and not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            image = [np.array(i).astype(np.float32) / 127.5 - 1.0 for i in image]

            image = np.stack(image, axis=0)  # to np
            image = torch.from_numpy(image.transpose(0, 3, 1, 2))
        elif isinstance(image[0], np.ndarray):
            image = np.stack(image, axis=0)  # to np
            if image.ndim == 5:
                image = image[0]

            image = torch.from_numpy(image.transpose(0, 3, 1, 2))
        elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
            dims = image[0].ndim

            if dims == 3:
                image = torch.stack(image, dim=0)
            elif dims == 4:
                image = torch.concat(image, dim=0)
            else:
                raise ValueError(f"Image must have 3 or 4 dimensions, instead got {dims}")

        image = image.to(device=device, dtype=self.unet.dtype)

        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self,
                 context: 'ContextList',
                 sampler,
                 prompt: Union[str, List[str]] = None,
                 height: int = None,
                 width: int = None,
                 image: Union[PIL.Image.Image, np.ndarray, torch.FloatTensor] = None,
                 num_inference_steps: int = 50,
                 timesteps: List[int] = None,
                 guidance_scale: float = 4.0,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: int = 1,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 noise_level: int = 250,
                 clean_caption: bool = True,
                 mcmc_iteration_cutoff: int = 50,
                 base_img_size: int = 128,
                 using_va_method: bool = False,
                 using_mcmc_sampling: bool = False,
                 unconditional: bool = False):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to None):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to None):
                The width in pixels of the generated image.
            image (`PIL.Image.Image`, `np.ndarray`, `torch.FloatTensor`):
                The image to be upscaled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*, defaults to None):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            noise_level (`int`, *optional*, defaults to 250):
                The amount of noise to add to the upscaled image. Must be in the range `[0, 1000)`
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            or watermarked content, according to the `safety_checker`.
        """
        # NOTE: You'll notice this ugly formatting throughout this function. I (Dylan) have
        # separated out chunks of code that I've directly copied from the IFPipeline. Specifically,
        # I compared the original IFPipeline (part of the diffusers library) to the IFPipeline
        # included as part of the MCMC notebook and I copied the portions of code that were added
        # for the MCMC sampling.
        # ------------------------------------------------------------------------------------------
        sizes, prompt, weights, views = context.collapse(is_stage_2=True)

        if unconditional:
            prompt = ["" for _ in prompt]

        device = self._execution_device
        weights = torch.Tensor(weights).to(device)[:, None, None, None]
        # print("Sizes:", sizes)
        # ------------------------------------------------------------------------------------------

        # 1. Check inputs. Raise error if not correct
        # This is further up in the function than in IFPipeline since batch_size is checked in the
        # check_inputs() method.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # self.check_inputs(
        #     prompt,
        #     image,
        #     batch_size,
        #     noise_level,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        # )

        # 2. Define call parameters
        # Woah. Didn't realize that you could rely on truthiness of int/None to select the one
        # that's specified like this. Good tip!
        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        # ------------------------------------------------------------------------------------------
        # TODO(dylan.colli): This isn't the same way as the IFPipeline specifies the number of
        # channels for MCMC but it was in the super resolution pipeline originally. I believe this
        # is done as in stage 2, the intermediate_images (initially pure noise) is concatenated
        # channel-wise with the noised output of stage 1. I bet this is done as a sort of
        # "guidance".
        num_channels = self.unet.config.in_channels // 2
        # num_channels = self.unet.config.in_channels
        # intermediate_images = self.prepare_intermediate_images(
        #     batch_size * num_images_per_prompt,
        #     num_channels,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        # )
        # With one prompt, the size of the latents_canvas is [1, num_channels, 256, 256]
        latents_canvas = self.prepare_intermediate_images(
            1,
            num_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # TODO: Should I upscale here?
        # 7. Prepare upscaled image and noise level
        # image is [1, 3, 128, 128]
        image = self.preprocess_image(image, num_images_per_prompt, device)

        # If truly upscaling, size is [1, 3, 256, 256]
        upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)
        # If not upscaling, size is [1, 3, 128, 128]
        # upscaled = image
        # ------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------
        # Dylan copied this from the IFPipeline
        # canvas_size is 256
        _, _, canvas_size, _ = latents_canvas.size()
        # intermediate_images is [num_contexts, num_channels, 128, 128] when only using 1 context.
        intermediate_images = extract_latents_stage_2(latents_canvas,
                                                      sizes,
                                                      views=views,
                                                      target_size=base_img_size)
        # ------------------------------------------------------------------------------------------

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ------------------------------------------------------------------------------------------
        # Dylan copied this from the IFPipeline
        alphas = 1 - self.scheduler.betas
        alphas_cumprod = np.cumprod(alphas)
        scalar = np.sqrt(1 / (1 - alphas_cumprod))

        # ------------------------------------------------------------------------------------------

        # # 7. Prepare upscaled image and noise level
        # image = self.preprocess_image(image, num_images_per_prompt, device)
        # upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

        # Repeat the upscaled image to match the number of contexts provided (the batch dimension)
        # upscaled = upscaled.repeat(len(sizes), 1, 1, 1)
        # upscaled_dummy_latents = extract_latents_stage_2(upscaled,
        #                                                  sizes,
        #                                                  views=views,
        #                                                  target_size=base_img_size)
        # print("Upscaled shape post-latent-extraction:", upscaled.shape)

        # This seems to suggest that the upscaled (and subsequently downscaled extracted latents) is
        # still fairly "good" and representative of the image that was passed in.
        # import matplotlib.pyplot as plt
        # from mcmc_visanagrams.utils.display import image_from_latents
        # for i in range(upscaled.shape[0]):
        #     plt.figure()
        #     up_np = upscaled[i].to(torch.double).detach().cpu().numpy().transpose(1, 2, 0)
        #     up_np = ((up_np + 1) / 2 * 255).astype(np.uint8)
        #     plt.imshow(up_np)
        #     plt.show()
        # plt.figure()
        # up_np = upscaled[0].to(torch.double).detach().cpu().numpy().transpose(1, 2, 0)
        # up_np = ((up_np + 1) / 2 * 255).astype(np.uint8)
        # plt.imshow(up_np)
        # plt.show()
        # return

        # NEED TO APPLY NOISE BEFORE EXTRACTING THE LATENTS!!!
        noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
        noise = randn_tensor(upscaled.shape,
                             generator=generator,
                             device=upscaled.device,
                             dtype=upscaled.dtype)
        upscaled: torch.Tensor = self.image_noising_scheduler.add_noise(upscaled,
                                                                        noise,
                                                                        timesteps=noise_level)
        # TODO: The above step noises the input image since stage 1 produces a clean, but low
        # resolution input image. I'm thinking I might need to extract the latents from this noised,
        # upscaled image and not concatenate intermediate_images and upscaled.

        upscaled = extract_latents_stage_2(upscaled, sizes, views=views, target_size=base_img_size)
        noise_level = torch.cat([noise_level] * upscaled.shape[0])

        if do_classifier_free_guidance:
            noise_level = torch.cat([noise_level] * 2)

        # ------------------------------------------------------------------------------------------
        # Begin copied section from IFPipeline by Dylan.
        # Compute the gradient function for MCMC sampling
        def gradient_fn(x, t, text_embeddings):
            # Compute normal classifier-free guidance update
            x = extract_latents_stage_2(x, sizes, views=views, target_size=base_img_size)

            # Dylan added this line for stage 2.
            x = torch.cat([x, upscaled], dim=1)

            model_input = (torch.cat([x] * 2) if do_classifier_free_guidance else x)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # NOTE: Getting an error here that says:
            #   "class_labels should be provided when num_class_embeds > 0"
            # What is indicating the number of class embeds?
            # Seems like the only thing that's different with this UNET call versus the call made in
            # the denoising loop is that the denoising loop passes the class_labels=noise_level.
            # predict the noise residual
            noise_pred = self.unet(
                model_input.type(prompt_embeds.dtype),
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
                # Dylan added this line for stage 2
                class_labels=noise_level,
            )[0]

            # s = noise_pred.size()
            # noise_pred = noise_pred.reshape(2, -1, *s[1:])
            # noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
            # noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
            # noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2,
            #                                                             dim=1)
            # (noise_pred_text, predicted_variance,
            #  noise_pred_uncond) = self._extract_noise_from_prediction(noise_pred)

            # # print("WARNING: Do I need to halve the number of channels here?")
            # noise_pred_uncond_canvas = make_canvas_stage_2(
            #     noise_pred_uncond,
            #     canvas_size,
            #     sizes,
            #     in_channels=self.unet.config.in_channels // 2,
            #     views=views,
            #     target_size=base_img_size)
            # noise_pred_text_canvas = make_canvas_stage_2(noise_pred_text,
            #                                              canvas_size,
            #                                              sizes,
            #                                              in_channels=self.unet.config.in_channels //
            #                                              2,
            #                                              views=views,
            #                                              base_img_size=base_img_size)
            # noise_pred = noise_pred_uncond_canvas + 7.5 * (noise_pred_text_canvas -
            #                                                noise_pred_uncond_canvas)

            # NOTE: I've verified that the noise extraction utilized by the
            # self._extract_noise_from_prediction behaves the same as the noise extraction that was
            # included here (in the MCMC notebook).
            # - Additionally, there were two `make_canvas` calls here in the MCMC notebook but that
            #   is accomplisehd in the `self._adjust_noise_pred_va_method` function.
            # - As the classifier-free guidance method is a linear combination (and thus independent
            #   across separate dimensions since no broadcasting is done), we can invert the views
            #   after the classifier-free guidance is done.

            # print("Warning!! Should this value be guidance_scale or weights instead of hard-coded?")
            noise_pred = self._classifier_free_guidance(noise_pred, 7.5, model_input)
            # print("Noise pred grad fn:", noise_pred.shape)

            if using_va_method:
                noise_pred = self._adjust_noise_pred_va_method(noise_pred, views, sizes)

                # print("Noise pred shape after adjustment (in grad fn):", noise_pred.shape)

                # And we don't need the predicted variance so we get rid of it
                noise_pred, _ = noise_pred.split(model_input.shape[1] // 2, dim=1)

            # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
            scale = scalar[t]
            noise_pred_normalized = noise_pred / (noise_pred**2).mean().sqrt()
            return -1 * scale * noise_pred_normalized

        def sync_fn(x):
            # TODO: Should we use num_channels instead of self.unet.config.in_channels?
            x_canvas = make_canvas_stage_2(x,
                                           canvas_size,
                                           sizes,
                                           in_channels=self.unet.config.in_channels,
                                           views=views,
                                           base_size=base_img_size)
            x = extract_latents_stage_2(x_canvas, sizes, views=views, target_size=base_img_size)
            return x

        def noise_fn():
            noise_canvas = torch.randn_like(latents_canvas)
            noise = extract_latents_stage_2(noise_canvas,
                                            sizes,
                                            views=views,
                                            target_size=base_img_size)
            return noise

        sampler._gradient_function = gradient_fn
        sampler._sync_function = sync_fn
        sampler._noise_function = noise_fn
        # ------------------------------------------------------------------------------------------

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self,
                   "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # TODO: I feel like it could be beneficial to switch which of the nearest neighbors
                # is used when downsampling in the extract latents function (but only per iteration
                # in the denoising loop, otherwise the gradient function would be messed up if we
                # did it per-function call). I feel like this might be the reason we get some visual
                # artifacts in the output images.
                # print("Intermediate images shape:", intermediate_images.shape)
                # print("Upscaled shape:", upscaled.shape)
                # This is failing due to intermediate images being [128, 128] and upscaled being
                # [256, 256]
                # NOTE: I'm going to try out commenting this out. If this doesn't work, I may need
                # to upscale the latents (if the UNET is expecting 256x256 input).
                model_input = torch.cat([intermediate_images, upscaled], dim=1)

                # If using classifier-free guidance, need to double the input along the batch
                # dimension so that the model can predict the noise residual for both the
                # unconditional and conditional noise.
                model_input = torch.cat([model_input] *
                                        2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)

                # Going to try to repeat the prompt embeddings as I think they are split into 2 in
                # the UNET model due to the concatenation of intermediate_images with upscaled.
                # prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                # This didn't work.

                # print("Model input shape:", model_input.shape)
                # print("Prompt embedding shape:", prompt_embeds.shape)
                # print("Cross attention kwargs:", cross_attention_kwargs)
                # print("Class labels:", noise_level)

                # predict the noise residual
                noise_pred: torch.Tensor = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = self._classifier_free_guidance(noise_pred, weights, model_input)
                    # print("Noise pred shape after classifer free guidance:", noise_pred.shape)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(intermediate_images.shape[1], dim=1)

                if using_va_method:
                    noise_pred = self._adjust_noise_pred_va_method(noise_pred, views, sizes)
                    intermediate_images = make_canvas_stage_2(intermediate_images,
                                                              canvas_size,
                                                              sizes,
                                                              views=views,
                                                              base_size=base_img_size)

                #     print("Adjusted shapes")
                #     print("\tnoise_pred:", noise_pred.shape)
                #     print("\tintermediate_images:", intermediate_images.shape)

                # print("Noise prediction shape input to scheduler:", noise_pred.shape)
                # print("intermediate_images shape input to scheduler:", intermediate_images.shape)

                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.scheduler.step(noise_pred,
                                                          t,
                                                          intermediate_images,
                                                          **extra_step_kwargs,
                                                          return_dict=False)[0]

                if using_va_method:
                    intermediate_images = extract_latents_stage_2(intermediate_images,
                                                                  sizes,
                                                                  views,
                                                                  target_size=base_img_size)

                # Dylan copied this conditional from IFPipeline.
                if using_mcmc_sampling and t > mcmc_iteration_cutoff:
                    # print(f"\nDoing MCMC for iteration {t}!!!\n")
                    # if False:
                    # The score functions in the last 50 steps don't really change the image
                    intermediate_images_canvas = make_canvas_stage_2(
                        intermediate_images,
                        canvas_size,
                        sizes,
                        in_channels=self.unet.config.in_channels // 2,
                        views=views,
                        base_size=base_img_size)
                    intermediate_images = sampler.sample_step(intermediate_images_canvas, t,
                                                              prompt_embeds)
                    intermediate_images = extract_latents_stage_2(intermediate_images,
                                                                  sizes,
                                                                  views=views,
                                                                  target_size=base_img_size)
                elif not using_va_method:
                    # print(f"\nSkipping MCMC for iteration {t}!!!\n")
                    intermediate_images_canvas = make_canvas_stage_2(
                        intermediate_images,
                        canvas_size,
                        sizes,
                        in_channels=self.unet.config.in_channels // 2,
                        views=views,
                        base_size=base_img_size)
                    intermediate_images = extract_latents_stage_2(intermediate_images_canvas,
                                                                  sizes,
                                                                  views=views,
                                                                  target_size=base_img_size)
                else:
                    pass

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                               (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

                #-----------------------------------------------------------------------------------
                # Dylan copied this from IFPipeline.
                # cast dtype
                intermediate_images = intermediate_images.type(prompt_embeds.dtype)
                #-----------------------------------------------------------------------------------

                # print()

        image = intermediate_images
        image = make_canvas_stage_2(image,
                                    canvas_size,
                                    sizes,
                                    in_channels=self.unet.config.in_channels // 2,
                                    views=views,
                                    base_size=base_img_size)
        return image

    def _extract_noise_from_prediction(self, model_output: torch.Tensor, model_input: torch.Tensor):
        """Extracts the conditioned and unconditioned noise estimates from model output

        NOTE: Assumes classifier-free guidance was used

        NOTE: I have verified (in test_noise_extraction.py) that both methods of noise extraction
        work the same way. I don't know why it was changed.
        """
        # Since the model takes in the (upscaled) stage 1 output as well as what we're denoising, we
        # need to chop off that portion of the output.
        channels = model_input.shape[1] // 2
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)

        noise_pred_uncond, _ = noise_pred_uncond.split(channels, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(channels, dim=1)

        return (noise_pred_text, predicted_variance, noise_pred_uncond)

    def _classifier_free_guidance(self, noise_pred: torch.Tensor, weights: torch.Tensor,
                                  model_input: torch.Tensor) -> torch.Tensor:
        # print("Noise prediction output from UNET shape:", noise_pred.shape)

        (noise_pred_text, predicted_variance,
         noise_pred_uncond) = self._extract_noise_from_prediction(noise_pred, model_input)

        noise_pred = noise_pred_uncond + weights * (noise_pred_text - noise_pred_uncond)

        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        return noise_pred

    def _adjust_noise_pred_va_method(self,
                                     noise_pred: torch.Tensor,
                                     views,
                                     sizes,
                                     reduction: str = 'mean'):
        if reduction != 'mean':
            raise ValueError("Only 'mean' reduction is supported for now.")
        # print("Noise pred shape before adjustment:", noise_pred.shape)

        # As make_canvas averages the input across the batch (zeroth) dimension, this is the same
        # operation as the VA mean method.
        noise_pred = make_canvas_stage_2(noise_pred,
                                         256,
                                         sizes,
                                         in_channels=6,
                                         base_size=256,
                                         views=views)

        # print("Noise pred shape after adjustment:", noise_pred.shape)

        return noise_pred
