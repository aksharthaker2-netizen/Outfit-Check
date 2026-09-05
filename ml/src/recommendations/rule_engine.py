"""
rule_engine.py — turns a P4 OutfitScoreResult into concrete, actionable
recommendations, grounded strictly in what the pipeline actually computed
(color harmony relationship + body-shape fit-factor reasoning).

v1 LIMITATION (flagged, not hidden): recommendations only cover color
harmony and value-contrast direction, because those are the only two
levers the current scorer reasons about. No garment fit/cut/fabric advice
yet — that needs signals (garment shape, texture) we don't extract yet.
"""

from dataclasses import dataclass
from enum import Enum

from ml.src.anthropometry.body_shape_classifier import BodyShape
from ml.src.scoring.person_conditioned_scorer import OutfitScoreResult

# Thresholds for deciding whether a recommendation is even worth giving.
# Below this, harmony is considered "good enough" not to flag.
HARMONY_GOOD_THRESHOLD = 0.75
# Below this, the fit factor is considered a meaningful penalty worth flagging.
FIT_FACTOR_PENALTY_THRESHOLD = 0.99


class RecommendationCategory(str, Enum):
    COLOR_HARMONY = "color_harmony"
    VALUE_CONTRAST = "value_contrast"
    NONE = "none"


@dataclass
class Recommendation:
    category: RecommendationCategory
    message: str
    priority: int  # 1 = highest priority, shown first


def _harmony_recommendation(relationship: str, harmony_score: float) -> Recommendation | None:
    if harmony_score >= HARMONY_GOOD_THRESHOLD:
        return None

    if relationship == "clashing":
        return Recommendation(
            category=RecommendationCategory.COLOR_HARMONY,
            message=(
                "These colors sit at an awkward distance on the color wheel. "
                "Try a complementary pairing (opposite hues) or a neutral "
                "(black, white, gray, beige) for one piece instead."
            ),
            priority=1,
        )

    # Between clashing and "good" — still worth a lighter nudge.
    return Recommendation(
        category=RecommendationCategory.COLOR_HARMONY,
        message=(
            f"This is a {relationship} pairing, which works but isn't the "
            "strongest combination. A closer analogous match or a neutral "
            "piece could sharpen it further."
        ),
        priority=2,
    )


def _fit_factor_recommendation(
    body_shape: BodyShape, fit_factor: float, explanation: str
) -> Recommendation | None:
    if fit_factor >= FIT_FACTOR_PENALTY_THRESHOLD:
        return None

    if body_shape == BodyShape.PEAR:
        message = (
            "For your body shape, a brighter or lighter top paired with a "
            "darker bottom tends to balance proportions better than the "
            "reverse — consider swapping which piece is lighter."
        )
    elif body_shape == BodyShape.INVERTED_TRIANGLE:
        message = (
            "For your body shape, a darker top paired with a brighter or "
            "lighter bottom tends to balance shoulder width better — "
            "consider swapping which piece is darker."
        )
    elif body_shape == BodyShape.RECTANGLE:
        message = (
            "Your top and bottom are close in brightness, which reads as "
            "flat on a rectangle silhouette. A stronger light/dark contrast "
            "between the two pieces would create more visual definition."
        )
    else:
        return None

    return Recommendation(
        category=RecommendationCategory.VALUE_CONTRAST,
        message=message,
        priority=1,
    )


def generate_recommendations(
    result: OutfitScoreResult, body_shape: BodyShape
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []

    fit_rec = _fit_factor_recommendation(body_shape, result.fit_factor, result.explanation)
    if fit_rec:
        recommendations.append(fit_rec)

    harmony_rec = _harmony_recommendation(result.relationship, result.base_harmony_score)
    if harmony_rec:
        recommendations.append(harmony_rec)

    recommendations.sort(key=lambda r: r.priority)

    if not recommendations:
        recommendations.append(
            Recommendation(
                category=RecommendationCategory.NONE,
                message="This outfit works well as-is — no changes recommended.",
                priority=1,
            )
        )

    return recommendations