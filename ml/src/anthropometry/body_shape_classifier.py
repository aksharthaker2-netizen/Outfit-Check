from dataclasses import dataclass
from enum import Enum

from ml.src.pose.detector import PoseResult


class BodyShape(str, Enum):
    RECTANGLE = "rectangle"
    PEAR = "pear"
    INVERTED_TRIANGLE = "inverted_triangle"
    HOURGLASS = "hourglass"
    APPLE = "apple"
    UNKNOWN = "unknown"


BALANCED_RATIO_LOW = 0.9
BALANCED_RATIO_HIGH = 1.3
WAIST_NARROW_RATIO_THRESHOLD = 0.85
WAIST_WIDE_RATIO_THRESHOLD = 0.95
VISIBILITY_MIN = 0.6


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
    if any(lm is None for lm in required) or any(lm.visibility < VISIBILITY_MIN for lm in required):
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


def classify_body_shape_with_waist(
    pose_result: PoseResult,
    waist_estimate,
    hip_width_from_mask: float | None = None,
    shoulder_width_from_mask: float | None = None,
) -> BodyShapeResult:
    landmarks = pose_result.landmarks
    left_shoulder = landmarks.get(11)
    right_shoulder = landmarks.get(12)
    left_hip = landmarks.get(23)
    right_hip = landmarks.get(24)

    required = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(lm is None for lm in required) or any(lm.visibility < VISIBILITY_MIN for lm in required):
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

        # Use mask-based widths ONLY if BOTH are available — mixing a mask
    # value with a landmark value is methodologically worse than using
    # either method alone (confirmed empirically: this produced a more
    # distorted ratio than the original landmark-only bug).
    if shoulder_width_from_mask is not None and hip_width_from_mask is not None:
        shoulder_width = shoulder_width_from_mask
        hip_width = hip_width_from_mask
    else:
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)

    if hip_width == 0 or shoulder_width == 0:
        return BodyShapeResult(shape=BodyShape.UNKNOWN, shoulder_hip_ratio=None, confidence=0.0)

    shoulder_hip_ratio = shoulder_width / hip_width
    landmark_confidence = sum(lm.visibility for lm in required) / len(required)

    if waist_estimate is None:
        if BALANCED_RATIO_LOW <= shoulder_hip_ratio <= BALANCED_RATIO_HIGH:
            shape = BodyShape.RECTANGLE
        elif shoulder_hip_ratio > BALANCED_RATIO_HIGH:
            shape = BodyShape.INVERTED_TRIANGLE
        else:
            shape = BodyShape.PEAR
        return BodyShapeResult(shape=shape, shoulder_hip_ratio=shoulder_hip_ratio, confidence=landmark_confidence)

    waist_width = waist_estimate.waist_width_normalized
    waist_shoulder_ratio = waist_width / shoulder_width
    waist_hip_ratio = waist_width / hip_width

    is_narrow_waist = waist_shoulder_ratio < WAIST_NARROW_RATIO_THRESHOLD and waist_hip_ratio < WAIST_NARROW_RATIO_THRESHOLD
    is_wide_waist = waist_shoulder_ratio > WAIST_WIDE_RATIO_THRESHOLD or waist_hip_ratio > WAIST_WIDE_RATIO_THRESHOLD

    if BALANCED_RATIO_LOW <= shoulder_hip_ratio <= BALANCED_RATIO_HIGH:
        shape = BodyShape.HOURGLASS if is_narrow_waist else (BodyShape.APPLE if is_wide_waist else BodyShape.RECTANGLE)
    elif shoulder_hip_ratio > BALANCED_RATIO_HIGH:
        shape = BodyShape.INVERTED_TRIANGLE
    else:
        shape = BodyShape.PEAR

    combined_confidence = (landmark_confidence + waist_estimate.confidence) / 2
    return BodyShapeResult(shape=shape, shoulder_hip_ratio=shoulder_hip_ratio, confidence=combined_confidence)