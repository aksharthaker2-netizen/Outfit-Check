"""
color_extractor.py — dominant color extraction per garment mask via k-means.

Runs k-means ONLY on pixels inside a garment's mask, so results reflect the
garment's actual color, not background/skin/wall pixels.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


@dataclass
class DominantColor:
    rgb: tuple[int, int, int]
    weight: float  # fraction of masked pixels belonging to this color cluster


def extract_dominant_colors(
    image_path: str, mask: np.ndarray, k: int = 3
) -> list[DominantColor]:
    """
    image_path: path to the original RGB image.
    mask: boolean array, shape (H, W), True where the garment is present
          (from GarmentSegmenter output).
    k: number of color clusters to extract.
    """
    image = np.array(Image.open(image_path).convert("RGB"))

    masked_pixels = image[mask]  # shape (N, 3)
    if len(masked_pixels) == 0:
        return []

    # If there are fewer pixels than k, reduce k to avoid a sklearn error.
    effective_k = min(k, len(masked_pixels))

    kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(masked_pixels)

    colors = []
    for cluster_idx in range(effective_k):
        cluster_mask = labels == cluster_idx
        weight = cluster_mask.sum() / len(masked_pixels)
        rgb = tuple(int(v) for v in kmeans.cluster_centers_[cluster_idx])
        colors.append(DominantColor(rgb=rgb, weight=weight))

    # Largest cluster first — that's the garment's primary color.
    colors.sort(key=lambda c: c.weight, reverse=True)
    return colors