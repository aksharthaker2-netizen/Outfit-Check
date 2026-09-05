"""
body_shape_classifier.py — landmark-ratio body-shape classification.

LIMITATION (flagged, not silently assumed): MediaPipe Pose has no waist
landmark, so hourglass and apple categories cannot be distinguished from
pose landmarks alone. This classifier covers 3 of 5 taxonomy categories
(rectangle, pear, inverted_triangle) until P3's segmentation mask supplies
a waist-width proxy to complete the taxonomy. See TODO below.
"""

from dataclasses import dataclass
from enum import Enum
from ml.src.pose.detector import PoseResult

# Ratio threshold: shoulder/hip within this band counts as "balanced" (rectangle).
# Outside the band, whichever is wider determines pear vs inverted-triangle.
BALANCED_RATIO_LOW = 0.9
BALANCED_RATIO_HIGH = 1.1


class BodyShape(str, Enum):
    RECTANGLE = "rectangle"
    PEAR = "pear"                    # hips wider than shoulders
    INVERTED_TRIANGLE = "inverted_triangle"  # shoulders wider than hips
    UNKNOWN = "unknown"
    # TODO (P3/P4): add HOURGLASS and APPLE once a waist-width proxy is
    # available from garment/body segmentation. Requires: waist_width_ratio
    # relative to both shoulder_width and hip_width.


@dataclass
class BodyShapeResult:
    shape: BodyShape
    shoulder_hip_ratio: float | None
    confidence: float


def classify_body_shape(pose_result: PoseResult) -> BodyShapeResult:
    landmarks = pose_result.landmarks

    left_shoulder = landmarks.get(11)
    right_shoulder = landmarks.get(12)
    left_hip = landmarks.get(23)
    right_hip = landmarks.get(24)

    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(lm is None for lm in required):
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    VISIBILITY_MIN = 0.6
    if any(lm.visibility < VISIBILITY_MIN for lm in required):
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    hip_width = abs(left_hip.x - right_hip.x)

    if hip_width == 0:
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    ratio = shoulder_width / hip_width

    if BALANCED_RATIO_LOW <= ratio <= BALANCED_RATIO_HIGH:
        shape = BodyShape.RECTANGLE
    elif ratio > BALANCED_RATIO_HIGH:
        shape = BodyShape.INVERTED_TRIANGLE
    else:
        shape = BodyShape.PEAR

    confidence = sum(lm.visibility for lm in required) / len(required)

    return BodyShapeResult(shape=shape, shoulder_hip_ratio=ratio, confidence=confidence)

# --- Add these to the existing BodyShape enum ---
class BodyShape(str, Enum):
    RECTANGLE = "rectangle"
    PEAR = "pear"
    INVERTED_TRIANGLE = "inverted_triangle"
    HOURGLASS = "hourglass"   # NEW — requires waist-width signal
    APPLE = "apple"           # NEW — requires waist-width signal
    UNKNOWN = "unknown"


# --- Add these new thresholds ---
WAIST_NARROW_RATIO_THRESHOLD = 0.85  # waist/shoulder or waist/hip below this = "narrow waist"
WAIST_WIDE_RATIO_THRESHOLD = 0.95    # waist/shoulder or waist/hip above this = "no waist definition / wide"


# --- New function, added alongside the existing classify_body_shape ---
def classify_body_shape_with_waist(
    pose_result: "PoseResult", waist_estimate: "WaistEstimate | None"
) -> BodyShapeResult:
    """
    Full 5-category classification using shoulder/hip ratio (from landmarks)
    PLUS waist width (from segmentation, via waist_estimator.py).

    Falls back to the 3-category classify_body_shape() logic if
    waist_estimate is None (e.g. segmentation wasn't run, or the top
    garment mask didn't yield a usable waist reading).
    """
    landmarks = pose_result.landmarks

    left_shoulder = landmarks.get(11)
    right_shoulder = landmarks.get(12)
    left_hip = landmarks.get(23)
    right_hip = landmarks.get(24)

    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(lm is None for lm in required):
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    VISIBILITY_MIN = 0.6
    if any(lm.visibility < VISIBILITY_MIN for lm in required):
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    hip_width = abs(left_hip.x - right_hip.x)

    if hip_width == 0:
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    shoulder_hip_ratio = shoulder_width / hip_width
    landmark_confidence = sum(lm.visibility for lm in required) / len(required)

    # No usable waist signal — fall back to the original 3-category logic.
    if waist_estimate is None:
        if BALANCED_RATIO_LOW <= shoulder_hip_ratio <= BALANCED_RATIO_HIGH:
            shape = BodyShape.RECTANGLE
        elif shoulder_hip_ratio > BALANCED_RATIO_HIGH:
            shape = BodyShape.INVERTED_TRIANGLE
        else:
            shape = BodyShape.PEAR
        return BodyShapeResult(shape=shape, shoulder_hip_ratio=shoulder_hip_ratio, confidence=landmark_confidence)

    # Waist signal available — full 5-category logic.
    waist_width = waist_estimate.waist_width_normalized
    waist_shoulder_ratio = waist_width / shoulder_width if shoulder_width else 1.0
    waist_hip_ratio = waist_width / hip_width if hip_width else 1.0

    is_narrow_waist = (
        waist_shoulder_ratio < WAIST_NARROW_RATIO_THRESHOLD
        and waist_hip_ratio < WAIST_NARROW_RATIO_THRESHOLD
    )
    is_wide_waist = (
        waist_shoulder_ratio > WAIST_WIDE_RATIO_THRESHOLD
        or waist_hip_ratio > WAIST_WIDE_RATIO_THRESHOLD
    )

    if BALANCED_RATIO_LOW <= shoulder_hip_ratio <= BALANCED_RATIO_HIGH:
        if is_narrow_waist:
            shape = BodyShape.HOURGLASS
        elif is_wide_waist:
            shape = BodyShape.APPLE
        else:
            shape = BodyShape.RECTANGLE
    elif shoulder_hip_ratio > BALANCED_RATIO_HIGH:
        shape = BodyShape.INVERTED_TRIANGLE
    else:
        shape = BodyShape.PEAR

    combined_confidence = (landmark_confidence + waist_estimate.confidence) / 2

    return BodyShapeResult(
        shape=shape,
        shoulder_hip_ratio=shoulder_hip_ratio,
        confidence=combined_confidence,
    )