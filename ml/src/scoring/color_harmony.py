"""
color_harmony.py — rule-based color harmony scoring using HSV hue relationships.

Concept: harmony rules (analogous, complementary, triadic) are defined on
hue position on the color wheel, not raw RGB distance. Low-saturation colors
(black/white/gray/beige) are treated as neutrals that harmonize with anything,
per standard styling heuristics.
"""

import colorsys
from dataclasses import dataclass

NEUTRAL_SATURATION_THRESHOLD = 0.15  # below this, treat as neutral regardless of hue
NEUTRAL_VALUE_EXTREMES = (0.12, 0.92)  # near-black or near-white also count as neutral

ANALOGOUS_MAX_DEGREES = 35
TRIADIC_TARGET_DEGREES = 120
TRIADIC_TOLERANCE_DEGREES = 20
COMPLEMENTARY_TARGET_DEGREES = 180
COMPLEMENTARY_TOLERANCE_DEGREES = 20


@dataclass
class HarmonyResult:
    score: float  # 0.0-1.0
    relationship: str  # "neutral_pairing" | "analogous" | "complementary" | "triadic" | "clashing"


def _rgb_to_hsv_degrees(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


def _is_neutral(s: float, v: float) -> bool:
    if s < NEUTRAL_SATURATION_THRESHOLD:
        return True
    if v < NEUTRAL_VALUE_EXTREMES[0] or v > NEUTRAL_VALUE_EXTREMES[1]:
        return True
    return False


def _hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


def score_color_pair(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> HarmonyResult:
    h1, s1, v1 = _rgb_to_hsv_degrees(rgb1)
    h2, s2, v2 = _rgb_to_hsv_degrees(rgb2)

    if _is_neutral(s1, v1) or _is_neutral(s2, v2):
        return HarmonyResult(score=0.9, relationship="neutral_pairing")

    dist = _hue_distance(h1, h2)

    if dist <= ANALOGOUS_MAX_DEGREES:
        # Closer within the analogous band = slightly higher score
        score = 1.0 - (dist / ANALOGOUS_MAX_DEGREES) * 0.15
        return HarmonyResult(score=score, relationship="analogous")

    if abs(dist - COMPLEMENTARY_TARGET_DEGREES) <= COMPLEMENTARY_TOLERANCE_DEGREES:
        closeness = 1.0 - abs(dist - COMPLEMENTARY_TARGET_DEGREES) / COMPLEMENTARY_TOLERANCE_DEGREES
        score = 0.75 + 0.15 * closeness
        return HarmonyResult(score=score, relationship="complementary")

    if abs(dist - TRIADIC_TARGET_DEGREES) <= TRIADIC_TOLERANCE_DEGREES:
        closeness = 1.0 - abs(dist - TRIADIC_TARGET_DEGREES) / TRIADIC_TOLERANCE_DEGREES
        score = 0.65 + 0.15 * closeness
        return HarmonyResult(score=score, relationship="triadic")

    return HarmonyResult(score=0.3, relationship="clashing")