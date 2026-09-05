"""
height_estimator.py — reference-object calibration for monocular height estimation.

Core assumption: the reference object and the person are at roughly the same
distance from the camera, on a roughly perpendicular camera axis. This is a
known limitation of single-camera (monocular) measurement — see concept note.
"""

from dataclasses import dataclass
from ml.src.pose.detector import PoseResult, Landmark

# Anthropometric constant: average ratio of (eye-to-nose distance) used to
# extrapolate head-top position beyond the nose, since MediaPipe has no
# literal "top of head" landmark. This is an approximation, not measured
# per-person — a known source of a few cm of systematic error.
HEAD_TOP_EXTRAPOLATION_FACTOR = 2.0


@dataclass
class CalibrationResult:
    pixel_per_cm: float


@dataclass
class HeightEstimate:
    height_cm: float
    confidence: float  # heuristic, based on landmark visibility used in the estimate


def calibrate_from_reference_object(
    object_pixel_height: float, object_real_height_cm: float
) -> CalibrationResult:
    """
    object_pixel_height: measured pixel height of the reference object in the frame
                          (e.g. a piece of A4 paper standing on its short edge,
                          or measured however you're detecting the object).
    object_real_height_cm: the object's known real-world height, e.g. 29.7 for A4.
    """
    if object_pixel_height <= 0:
        raise ValueError("object_pixel_height must be > 0")
    return CalibrationResult(pixel_per_cm=object_pixel_height / object_real_height_cm)


def _estimate_head_top_y(landmarks: dict[int, Landmark]) -> float | None:
    """
    Extrapolates head-top y (normalized) from nose + eye landmarks, since
    MediaPipe has no literal head-top point. Uses the nose-to-eye vertical
    gap, scaled by HEAD_TOP_EXTRAPOLATION_FACTOR, as a proxy for
    nose-to-head-top distance.
    """
    nose = landmarks.get(0)
    left_eye = landmarks.get(2)
    right_eye = landmarks.get(5)

    if nose is None or (left_eye is None and right_eye is None):
        return None

    eye_y_candidates = [e.y for e in (left_eye, right_eye) if e is not None]
    eye_y = sum(eye_y_candidates) / len(eye_y_candidates)

    nose_to_eye = abs(nose.y - eye_y)
    head_top_y = nose.y - (nose_to_eye * HEAD_TOP_EXTRAPOLATION_FACTOR)
    return head_top_y


def estimate_height(
    pose_result: PoseResult, calibration: CalibrationResult
) -> HeightEstimate | None:
    """
    Estimates standing height from head-top (extrapolated) to average ankle y.
    Returns None if required landmarks aren't visible enough to trust.
    """
    landmarks = pose_result.landmarks

    head_top_y = _estimate_head_top_y(landmarks)
    left_ankle = landmarks.get(27)
    right_ankle = landmarks.get(28)

    if head_top_y is None or left_ankle is None or right_ankle is None:
        return None

    VISIBILITY_MIN = 0.6
    ankle_visibilities = [
        lm.visibility for lm in (left_ankle, right_ankle) if lm.visibility >= VISIBILITY_MIN
    ]
    if not ankle_visibilities:
        return None

    ankle_y = (left_ankle.y + right_ankle.y) / 2

    span_normalized = ankle_y - head_top_y  # normalized [0,1] fraction of frame height
    span_pixels = span_normalized * pose_result.frame_height

    height_cm = span_pixels / calibration.pixel_per_cm

    # Heuristic confidence: average visibility of the landmarks actually used.
    confidence = sum(ankle_visibilities) / len(ankle_visibilities)

    return HeightEstimate(height_cm=height_cm, confidence=confidence)