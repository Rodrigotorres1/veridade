"""Extrator de embeddings com o FaRL congelado.

Usado em dois lugares, sempre com o mesmo código:
  1. no Colab, para extrair as features do bootstrap dataset (uma vez só);
  2. na DAG do Airflow, para extrair as features das imagens novas que
     chegam pelo MQTT (em CPU, em lotes pequenos).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.config import CLIP_PROCESSOR_ID, FARL_HF_ID


class FarlExtractor:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPVisionModelWithProjection.from_pretrained(FARL_HF_ID).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        try:
            self.processor = CLIPImageProcessor.from_pretrained(FARL_HF_ID)
        except OSError:  # repo do FaRL pode não ter preprocessor_config.json
            self.processor = CLIPImageProcessor.from_pretrained(CLIP_PROCESSOR_ID)

    @torch.no_grad()
    def embed(self, imagens: list[Image.Image]) -> np.ndarray:
        """Recebe imagens PIL (rosto já recortado) e devolve (n, 512) normalizado L2,
        igual ao "normalized and projected output" descrito no artigo."""
        batch = self.processor(images=[im.convert("RGB") for im in imagens], return_tensors="pt")
        pix = batch["pixel_values"].to(self.device)
        if self.device == "cuda":
            pix = pix.half()
        emb = self.model(pixel_values=pix).image_embeds.float()
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()
