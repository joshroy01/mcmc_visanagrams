"""
    file copy from diffusion library from Huggingface: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
"""
import numpy as np
import torch
import torch.nn as nn

from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

from diffusers.utils import logging

logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class IFSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModelWithProjection(config.vision_config)

        self.p_head = nn.Linear(config.vision_config.projection_dim, 1)
        self.w_head = nn.Linear(config.vision_config.projection_dim, 1)

    @torch.no_grad()
    def forward(self, clip_input, images, p_threshold=0.5, w_threshold=0.5):
        image_embeds = self.vision_model(clip_input)[0]

        nsfw_detected = self.p_head(image_embeds)
        nsfw_detected = nsfw_detected.flatten()
        nsfw_detected = nsfw_detected > p_threshold
        nsfw_detected = nsfw_detected.tolist()

        if any(nsfw_detected):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed.")

        for idx, nsfw_detected_ in enumerate(nsfw_detected):
            if nsfw_detected_:
                images[idx] = np.zeros(images[idx].shape)

        watermark_detected = self.w_head(image_embeds)
        watermark_detected = watermark_detected.flatten()
        watermark_detected = watermark_detected > w_threshold
        watermark_detected = watermark_detected.tolist()

        if any(watermark_detected):
            logger.warning(
                "Potential watermarked content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed.")

        for idx, watermark_detected_ in enumerate(watermark_detected):
            if watermark_detected_:
                images[idx] = np.zeros(images[idx].shape)

        return images, nsfw_detected, watermark_detected
