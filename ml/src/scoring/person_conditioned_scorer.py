"""
person_conditioned_scorer.py — fuses color-harmony score with a
body-shape-conditioned fit factor, per the styling principle that value
(brightness) contrast placement between top/bottom visually balances
different body-shape proportions.

v1 LIMITATION (flagged, not hidden): only uses dominant-color brightness.
Says nothing about garment fit, cut, or length — those require features
we don't extract yet. This is a real scoping limitation, to be revisited
once segmentation masks can supply garment-shape features (P3/P4 follow-up).
"""

import colorsys
from dataclasses import dataclass

from ml.src.anthropometry.body_shape_classifier import BodyShape
from ml.src.scoring.color_harmony import score_color_pair, HarmonyResult

# Bounds on how much the body-shape fit factor can move the score.
# Kept modest — value-contrast placement is a secondary signal next to
# color harmony itself, not a dominant one.
FIT_FACTOR_MIN = 0.85
FIT_FACTOR_MAX = 1.15

# How large a brightness (V) difference counts as "strong contrast" for
# the rectangle rule, and as a meaningful directional signal for pear/
# inverted-triangle. On a 0-1 V scale.
CONTRAST_STRONG_THRESHOLD = 0.35


@dataclass
class OutfitScoreResult:
    final_score: float           # 0.0-1.0, clamped
    base_harmony_score: float    # from color_harmony, unmodified
    fit_factor: float            # the body-shape-conditioned multiplier applied
    relationship: str            # harmony relationship label (analogous/complementary/etc.)
    explanation: str             # human-readable reason for the fit factor


def _get_value(rgb: tuple[int, int, int]) -> float:
    r, g, b = [v / 255.0 for v in rgb]
    _, _, v = colorsys.rgb_to_hsv(r, g, b)
    return v


def _compute_fit_factor(
    body_shape: BodyShape, top_rgb: tuple[int, int, int], bottom_rgb: tuple[int, int, int]
) -> tuple[float, str]:
    top_v = _get_value(top_rgb)
    bottom_v = _get_value(bottom_rgb)
    diff = top_v - bottom_v  # positive = top brighter than bottom
    contrast_magnitude = abs(diff)

    if body_shape == BodyShape.PEAR:
        # Reward brighter top / darker bottom (draws eye upward).
        if diff > 0:
            factor = 1.0 + min(diff, 1.0) * (FIT_FACTOR_MAX - 1.0)
            reason = "Brighter top with darker bottom helps balance a pear silhouette."
        else:
            factor = 1.0 + min(abs(diff), 1.0) * (FIT_FACTOR_MIN - 1.0)
            reason = "Darker top with brighter bottom can emphasize hip width on a pear silhouette."
        return factor, reason

    if body_shape == BodyShape.INVERTED_TRIANGLE:
        # Reward darker top / brighter bottom (balances shoulder width).
        if diff < 0:
            factor = 1.0 + min(abs(diff), 1.0) * (FIT_FACTOR_MAX - 1.0)
            reason = "Darker top with brighter bottom helps balance broader shoulders."
        else:
            factor = 1.0 + min(diff, 1.0) * (FIT_FACTOR_MIN - 1.0)
            reason = "Brighter top with darker bottom can emphasize shoulder width on this silhouette."
        return factor, reason

    if body_shape == BodyShape.RECTANGLE:
        # Reward contrast magnitude regardless of direction (creates
        # visual waist definition a rectangle silhouette otherwise lacks).
        if contrast_magnitude >= CONTRAST_STRONG_THRESHOLD:
            factor = 1.0 + min(contrast_magnitude, 1.0) * (FIT_FACTOR_MAX - 1.0)
            reason = "Strong top/bottom contrast creates visual waist definition on a rectangle silhouette."
        else:
            factor = 1.0 - (CONTRAST_STRONG_THRESHOLD - contrast_magnitude) * 0.3
            reason = "Low top/bottom contrast gives little visual definition on a rectangle silhouette."
        return factor, reason

    # UNKNOWN or unhandled shape: no directional basis to adjust — stay neutral.
    return 1.0, "Body shape not confidently classified — no fit adjustment applied."


def score_outfit(
    body_shape: BodyShape,
    top_rgb: tuple[int, int, int],
    bottom_rgb: tuple[int, int, int],
) -> OutfitScoreResult:
    harmony: HarmonyResult = score_color_pair(top_rgb, bottom_rgb)
    fit_factor, explanation = _compute_fit_factor(body_shape, top_rgb, bottom_rgb)

    fit_factor = max(FIT_FACTOR_MIN, min(FIT_FACTOR_MAX, fit_factor))

    final_score = harmony.score * fit_factor
    final_score = max(0.0, min(1.0, final_score))

    return OutfitScoreResult(
        final_score=final_score,
        base_harmony_score=harmony.score,
        fit_factor=fit_factor,
        relationship=harmony.relationship,
        explanation=explanation,
    )