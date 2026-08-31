"""
framing_check.py — decides whether a frame contains a valid, fully-framed
full-body pose, and what guidance to give if not.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .detector import PoseResult, Landmark


class BodyPart(str, Enum):
    HEAD = "head"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"


REQUIRED_LANDMARKS = {
    BodyPart.HEAD: 0,
    BodyPart.LEFT_SHOULDER: 11,
    BodyPart.RIGHT_SHOULDER: 12,
    BodyPart.LEFT_HIP: 23,
    BodyPart.RIGHT_HIP: 24,
    BodyPart.LEFT_ANKLE: 27,
    BodyPart.RIGHT_ANKLE: 28,
}

VISIBILITY_THRESHOLD = 0.6
EDGE_MARGIN = 0.03  # landmark must be >=3% of frame dimension away from every edge


@dataclass
class FramingStatus:
    is_valid: bool
    missing_or_occluded: list[BodyPart] = field(default_factory=list)
    too_close_to_edge: list[BodyPart] = field(default_factory=list)
    guidance: str = ""


def _landmark_near_edge(lm: Landmark) -> Optional[str]:
    if lm.x < EDGE_MARGIN:
        return "left"
    if lm.x > 1.0 - EDGE_MARGIN:
        return "right"
    if lm.y < EDGE_MARGIN:
        return "top"
    if lm.y > 1.0 - EDGE_MARGIN:
        return "bottom"
    return None


def check_framing(pose_result: PoseResult) -> FramingStatus:
    if not pose_result.detected:
        return FramingStatus(is_valid=False, guidance="No person detected — step into frame.")

    missing_or_occluded: list[BodyPart] = []
    too_close_to_edge: list[BodyPart] = []
    edge_hits: dict[str, list[BodyPart]] = {"left": [], "right": [], "top": [], "bottom": []}

    for part, idx in REQUIRED_LANDMARKS.items():
        lm = pose_result.landmarks.get(idx)

        if lm is None or lm.visibility < VISIBILITY_THRESHOLD:
            missing_or_occluded.append(part)
            continue

        edge = _landmark_near_edge(lm)
        if edge is not None:
            too_close_to_edge.append(part)
            edge_hits[edge].append(part)

    is_valid = not missing_or_occluded and not too_close_to_edge
    guidance = _build_guidance(missing_or_occluded, edge_hits)

    return FramingStatus(
        is_valid=is_valid,
        missing_or_occluded=missing_or_occluded,
        too_close_to_edge=too_close_to_edge,
        guidance=guidance,
    )


def _build_guidance(missing: list[BodyPart], edge_hits: dict[str, list[BodyPart]]) -> str:
    # Priority: occlusion/missing first (ambiguous, needs general repositioning),
    # then edge-cropping (directional, specific). Only one instruction at a time.
    if BodyPart.LEFT_ANKLE in missing or BodyPart.RIGHT_ANKLE in missing:
        return "Step back so your feet are visible."
    if BodyPart.HEAD in missing:
        return "Step back so your head is visible."
    if missing:
        return "Step back — your full body isn't visible."

    if edge_hits["bottom"]:
        return "Step back — your feet are too close to the bottom of frame."
    if edge_hits["top"]:
        return "Step back — your head is too close to the top of frame."
    if edge_hits["left"]:
        return "Move right — you're too close to the left edge."
    if edge_hits["right"]:
        return "Move left — you're too close to the right edge."

    return "Good — hold still."