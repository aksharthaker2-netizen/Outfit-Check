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

import cv2
import os

def save_debug_mask(image_path, seg_result, out_name="P2b_debug.png"):
    image = cv2.imread(image_path)
    overlay = image.copy()

    from ml.src.segmentation import GarmentCategory
    colors = {
        GarmentCategory.TOP: (0, 255, 0),
        GarmentCategory.BOTTOM: (255, 0, 0),
    }
    for category, mask in seg_result.masks.items():
        color = colors.get(category, (255, 255, 255))
        overlay[mask] = color

    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    out_dir = "ml/notebooks/debug_output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, blended)
    print(f"Debug mask saved to: {out_path}")


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
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_debug_mask(image_path, seg_result, out_name=f"P2b_debug_{base_name}.png")
    print(f"Detected categories: {list(seg_result.masks.keys())}")

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

            print(f"DEBUG: shoulder_y={shoulder_y:.3f}  hip_y={hip_y:.3f}  scan_range_px=({int(shoulder_y*seg_result.image_height)}, {int(hip_y*seg_result.image_height)})")
            waist_estimate = estimate_waist_width(seg_result.masks[GarmentCategory.TOP], shoulder_y, hip_y)
            if waist_estimate:
                print(f"DEBUG: narrowest row found at y={waist_estimate.row_used_y_normalized:.3f} "
                f"(scan range was {shoulder_y:.3f} to {hip_y:.3f})")
            hip_width_from_mask = estimate_hip_width(seg_result.masks[GarmentCategory.BOTTOM], hip_y)
            shoulder_width_from_mask = estimate_shoulder_width(seg_result.masks[GarmentCategory.TOP], shoulder_y)

    print(f"Hip width from mask: {hip_width_from_mask}")
    print(f"Shoulder width from mask: {shoulder_width_from_mask}")

    if waist_estimate:
        print(f"Waist width from mask: {waist_estimate.waist_width_normalized}  (confidence: {waist_estimate.confidence:.2f})")
    else:
        print("Waist width from mask: None")

    new_result = classify_body_shape_with_waist(
        pose_result, waist_estimate, hip_width_from_mask, shoulder_width_from_mask
    )
    print(f"\nShape: {new_result.shape.value}  (ratio: {new_result.shoulder_hip_ratio:.2f}, confidence: {new_result.confidence:.2f})")


if __name__ == "__main__":
    main()