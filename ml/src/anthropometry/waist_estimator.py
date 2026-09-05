from dataclasses import dataclass
import numpy as np


@dataclass
class WaistEstimate:
    waist_width_normalized: float
    row_used_y_normalized: float
    confidence: float


def _measure_mask_width_at_y(mask: np.ndarray, y_normalized: float) -> float | None:
    h, w = mask.shape
    y = int(y_normalized * h)
    y = max(0, min(h - 1, y))
    row = mask[y]
    xs = np.where(row)[0]
    if len(xs) == 0:
        return None
    width_px = xs.max() - xs.min()
    return width_px / w


def estimate_waist_width(
    top_mask: np.ndarray,
    shoulder_y_normalized: float,
    hip_y_normalized: float,
) -> WaistEstimate | None:
    h, w = top_mask.shape
    y_start = int(shoulder_y_normalized * h)
    y_end = int(hip_y_normalized * h)
    if y_start >= y_end:
        return None

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


def estimate_hip_width(bottom_mask: np.ndarray, hip_y_normalized: float) -> float | None:
    return _measure_mask_width_at_y(bottom_mask, hip_y_normalized)


def estimate_shoulder_width(top_mask: np.ndarray, shoulder_y_normalized: float) -> float | None:
    return _measure_mask_width_at_y(top_mask, shoulder_y_normalized)