import sys
import os
import cv2

from ml.src.pose import PoseDetector
from ml.src.anthropometry import classify_body_shape, BodyShape
from ml.src.segmentation import GarmentSegmenter, GarmentCategory, extract_dominant_colors
from ml.src.scoring import score_outfit
from ml.src.recommendations import generate_recommendations


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.notebooks.P5_recommendations <path_to_image>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("Running pose detection...")
    with PoseDetector(static_image_mode=True) as detector:
        frame = cv2.imread(image_path)
        pose_result = detector.process(frame)

    if not pose_result.detected:
        print("No person detected. Exiting.")
        return

    shape_result = classify_body_shape(pose_result)
    print(f"Classified body shape: {shape_result.shape.value} "
          f"(confidence: {shape_result.confidence:.2f})")

    print("\nRunning garment segmentation...")
    segmenter = GarmentSegmenter()
    seg_result = segmenter.segment(image_path)

    if GarmentCategory.TOP not in seg_result.masks or GarmentCategory.BOTTOM not in seg_result.masks:
        print("Need both a top and bottom garment detected to score. Exiting.")
        return

    top_rgb = extract_dominant_colors(image_path, seg_result.masks[GarmentCategory.TOP], k=1)[0].rgb
    bottom_rgb = extract_dominant_colors(image_path, seg_result.masks[GarmentCategory.BOTTOM], k=1)[0].rgb
    print(f"Top primary color: RGB{top_rgb}")
    print(f"Bottom primary color: RGB{bottom_rgb}")

    result = score_outfit(shape_result.shape, top_rgb, bottom_rgb)
    print(f"\nFinal score: {result.final_score:.2f} "
          f"(base harmony: {result.base_harmony_score:.2f}, fit factor: {result.fit_factor:.2f})")
    print(f"Relationship: {result.relationship}")

    print("\n--- Recommendations ---")
    recommendations = generate_recommendations(result, shape_result.shape)
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. [{rec.category.value}] {rec.message}")
        from ml.src.recommendations import phrase_recommendations
    print("\n--- LLM-Phrased Version ---")
    natural_text = phrase_recommendations(recommendations)
    print(natural_text)


if __name__ == "__main__":
    main()