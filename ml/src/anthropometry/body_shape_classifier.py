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