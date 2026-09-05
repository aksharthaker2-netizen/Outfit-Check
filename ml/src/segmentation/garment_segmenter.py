"""
garment_segmenter.py — pretrained clothes-parsing model wrapper.

Uses SegFormer fine-tuned for human clothes parsing (18 classes) via
HuggingFace transformers. Inference-only, runs on CPU. First run downloads
model weights from HuggingFace Hub — requires internet once, then cached
locally.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

MODEL_NAME = "mattmdjaga/segformer_b2_clothes"

# Model's own label indices (from its config) -> our simplified garment categories.
# Model classes we care about: 4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress,
# 9=Left-shoe, 10=Right-shoe. Everything else (background, hair, face, skin,
# limbs, bag, etc.) is ignored for garment purposes.
class GarmentCategory(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    SHOES = "shoes"


LABEL_TO_CATEGORY = {
    4: GarmentCategory.TOP,
    5: GarmentCategory.BOTTOM,
    6: GarmentCategory.BOTTOM,
    7: GarmentCategory.DRESS,
    9: GarmentCategory.SHOES,
    10: GarmentCategory.SHOES,
}


@dataclass
class SegmentationResult:
    # Boolean mask per category, shape (H, W), True where that garment is present.
    # A category is absent (no key) if it wasn't detected in the image.
    masks: dict[GarmentCategory, np.ndarray]
    image_width: int
    image_height: int


class GarmentSegmenter:
    def __init__(self, device: str = None):
        # device-agnostic: uses GPU if available (e.g. if run on Colab later),
        # otherwise CPU — fine for single-image inference either way.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image_path: str) -> SegmentationResult:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (1, num_labels, H', W')

        # Upsample logits back to original image resolution
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        pred_labels = upsampled.argmax(dim=1)[0].cpu().numpy()  # shape (H, W)

        masks: dict[GarmentCategory, np.ndarray] = {}
        for label_idx, category in LABEL_TO_CATEGORY.items():
            label_mask = (pred_labels == label_idx)
            if not label_mask.any():
                continue
            if category in masks:
                masks[category] = masks[category] | label_mask  # merge e.g. left+right shoe
            else:
                masks[category] = label_mask

        return SegmentationResult(masks=masks, image_width=w, image_height=h)