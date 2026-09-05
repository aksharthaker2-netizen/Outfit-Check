"""
waist_estimator.py — waist-width proxy from garment segmentation mask.

LIMITATION (flagged, not hidden): accuracy depends on how fitted the top
garment is. A loose/baggy top traces the garment's silhouette, not the
body's, and will overestimate waist width. Still strictly more signal
than zero waist information, which is the honest framing for this v1.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class WaistEstimate:
    waist_width_normalized: float  # fraction of image width, comparable to shoulder/hip width
    row_used_y_normalized: float   # which row (normalized y) the narrowest point was found at
    confidence: float              # heuristic: how many valid rows had mask pixels at all


def estimate_waist_width(
    top_mask: np.ndarray,
    shoulder_y_normalized: float,
    hip_y_normalized: float,
) -> WaistEstimate | None:
    """
    top_mask: boolean array (H, W), True where the top garment is present
              (from GarmentSegmenter's TOP category mask).
    shoulder_y_normalized / hip_y_normalized: average shoulder/hip y from
              pose landmarks, normalized [0,1] relative to frame height.
    """
    h, w = top_mask.shape

    y_start = int(shoulder_y_normalized * h)
    y_end = int(hip_y_normalized * h)

    if y_start >= y_end:
        return None  # malformed input — shoulder should be above hip in image coords

    min_width_px = None
    min_width_row = None
    valid_rows = 0

    for y in range(y_start, y_end):
        row = top_mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        valid_rows += 1
        width_px = xs.max() - xs.min()
        if min_width_px is None or width_px < min_width_px:
            min_width_px = width_px
            min_width_row = y

    total_rows = y_end - y_start
    if min_width_px is None or total_rows == 0:
        return None

    confidence = valid_rows / total_rows

    return WaistEstimate(
        waist_width_normalized=min_width_px / w,
        row_used_y_normalized=min_width_row / h,
        confidence=confidence,
    )