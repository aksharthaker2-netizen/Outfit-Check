import sys
import os
import cv2
import numpy as np

from ml.src.segmentation.garment_segmenter import GarmentSegmenter, GarmentCategory
from ml.src.segmentation.color_extractor import extract_dominant_colors
from ml.src.scoring.color_harmony import score_color_pair

DEBUG_DIR = "ml/notebooks/debug_output"


def save_mask_visualization(image_path, masks):
    image = cv2.imread(image_path)
    overlay = image.copy()

    colors = {
        GarmentCategory.TOP: (0, 255, 0),
        GarmentCategory.BOTTOM: (255, 0, 0),
        GarmentCategory.DRESS: (0, 255, 255),
        GarmentCategory.SHOES: (0, 0, 255),
    }

    for category, mask in masks.items():
        color = colors.get(category, (255, 255, 255))
        overlay[mask] = color

    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    out_path = os.path.join(DEBUG_DIR, "P3_segmentation_debug.png")
    cv2.imwrite(out_path, blended)
    print(f"Segmentation debug image saved to: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.notebooks.P3_segmentation_color <path_to_image>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("Loading segmentation model (first run downloads weights, needs internet)...")
    segmenter = GarmentSegmenter()

    print("Running segmentation...")
    result = segmenter.segment(image_path)

    if not result.masks:
        print("No garments detected.")
        return

    print(f"Detected garment categories: {[c.value for c in result.masks.keys()]}")

    garment_colors = {}
    for category, mask in result.masks.items():
        colors = extract_dominant_colors(image_path, mask, k=3)
        garment_colors[category] = colors
        print(f"\n{category.value} dominant colors:")
        for c in colors:
            print(f"  RGB {c.rgb} — weight {c.weight:.2f}")

    save_mask_visualization(image_path, result.masks)

    # Score harmony between every pair of garments' primary colors
    categories = list(garment_colors.keys())
    if len(categories) >= 2:
        print("\nColor harmony between garments:")
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat_a, cat_b = categories[i], categories[j]
                if not garment_colors[cat_a] or not garment_colors[cat_b]:
                    continue
                primary_a = garment_colors[cat_a][0].rgb
                primary_b = garment_colors[cat_b][0].rgb
                harmony = score_color_pair(primary_a, primary_b)
                print(f"  {cat_a.value} vs {cat_b.value}: "
                      f"score={harmony.score:.2f}, relationship={harmony.relationship}")


if __name__ == "__main__":
    main()