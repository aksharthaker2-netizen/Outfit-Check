"""
detector.py — MediaPipe Pose wrapper.

MediaPipe Pose is a lightweight, CPU-friendly heatmap-regression model.
Given a frame, it returns 33 body landmarks, each with (x, y, z, visibility).
`visibility` is the model's confidence that the landmark is actually present
and unoccluded — the signal the framing check depends on downstream.
"""

from dataclasses import dataclass
import numpy as np
import mediapipe as mp

LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    27: "left_ankle",
    28: "right_ankle",
}


@dataclass
class Landmark:
    x: float          # normalized [0, 1], relative to frame width
    y: float          # normalized [0, 1], relative to frame height
    z: float          # normalized depth, relative to hip midpoint
    visibility: float # [0, 1] confidence the point is present & unoccluded


@dataclass
class PoseResult:
    landmarks: dict[int, Landmark]
    frame_width: int
    frame_height: int
    detected: bool


class PoseDetector:
    """
    Usage:
        detector = PoseDetector()
        result = detector.process(frame_bgr)   # frame from cv2.VideoCapture
        if result.detected:
            nose = result.landmarks[0]
    """

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        # model_complexity: 0 (fastest) / 1 (balanced) / 2 (most accurate, slowest).
        # complexity=1 is right for a live CPU guidance loop — complexity=2
        # is meant for accuracy-critical offline analysis, not real-time feedback.
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB

        results = self._pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return PoseResult(landmarks={}, frame_width=w, frame_height=h, detected=False)

        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)

        return PoseResult(landmarks=landmarks, frame_width=w, frame_height=h, detected=True)

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()