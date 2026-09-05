import sys
import os
import cv2

from ml.src.pose import PoseDetector
from ml.src.anthropometry import (
    classify_body_shape,
    classify_body_shape_with_waist,
    estimate_waist_width,
)
from ml.src.segmentation import GarmentSegmenter, GarmentCategory


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.notebooks.P2b_waist_bodyshape <path_to_image>")
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

    old_result = classify_body_shape(pose_result)
    print(f"\n--- OLD (3-category, landmarks only) ---")
    print(f"Shape: {old_result.shape.value}  "
          f"(shoulder/hip ratio: {old_result.shoulder_hip_ratio:.2f}, "
          f"confidence: {old_result.confidence:.2f})")

    print("\nRunning garment segmentation for waist estimate...")
    segmenter = GarmentSegmenter()
    seg_result = segmenter.segment(image_path)

    from ml.src.anthropometry import estimate_hip_width, estimate_shoulder_width

    waist_estimate = None
    hip_width_from_mask = None
    shoulder_width_from_mask = None

    if GarmentCategory.TOP in seg_result.masks and GarmentCategory.BOTTOM in seg_result.masks:
        landmarks = pose_result.landmarks
        left_shoulder, right_shoulder = landmarks.get(11), landmarks.get(12)
        left_hip, right_hip = landmarks.get(23), landmarks.get(24)

        if all(lm is not None for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2

            waist_estimate = estimate_waist_width(seg_result.masks[GarmentCategory.TOP], shoulder_y, hip_y)
            hip_width_from_mask = estimate_hip_width(seg_result.masks[GarmentCategory.BOTTOM], hip_y)
            shoulder_width_from_mask = estimate_shoulder_width(seg_result.masks[GarmentCategory.TOP], shoulder_y)

    print(f"Hip width from mask: {hip_width_from_mask}")
    print(f"Shoulder width from mask: {shoulder_width_from_mask}")

    new_result = classify_body_shape_with_waist(
        pose_result, waist_estimate, hip_width_from_mask, shoulder_width_from_mask
    )
    print(f"\nShape: {new_result.shape.value}  (ratio: {new_result.shoulder_hip_ratio:.2f}, confidence: {new_result.confidence:.2f})")


if __name__ == "__main__":
    main()