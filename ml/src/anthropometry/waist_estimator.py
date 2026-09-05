from dataclasses import dataclass
import numpy as np


@dataclass
class WaistEstimate:
    waist_width_normalized: float
    row_used_y_normalized: float
    confidence: float


# Fraction of frame height sampled on EACH side of a target y-row, when
# measuring hip/shoulder width. Guards against single-row occlusion or
# fold artifacts by taking a median over a small window instead of
# trusting one row — confirmed necessary empirically (a test photo
# returned zero mask pixels at the exact hip-landmark row due to
# occlusion from a long top).
ROW_WINDOW_FRACTION = 0.03


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


def _measure_mask_width_windowed(
    mask: np.ndarray, y_normalized: float, window_fraction: float = ROW_WINDOW_FRACTION
) -> float | None:
    """
    Measures mask width at a small window of rows centered on
    y_normalized, and returns the MEDIAN width across all valid rows in
    that window (rows with zero mask pixels are skipped, not counted as
    zero). Median is used specifically because it tolerates a handful of
    outlier rows (occlusion, folds, seams) without being pulled by them
    the way a mean would be.
    """
    h, w = mask.shape
    center_y = int(y_normalized * h)
    window_px = max(1, int(h * window_fraction))

    y_start = max(0, center_y - window_px)
    y_end = min(h, center_y + window_px + 1)

    widths = []
    for y in range(y_start, y_end):
        row = mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        widths.append(xs.max() - xs.min())

    if not widths:
        return None

    median_width_px = float(np.median(widths))
    return median_width_px / w


def estimate_waist_width(
    top_mask: np.ndarray,
    shoulder_y_normalized: float,
    hip_y_normalized: float,
) -> WaistEstimate | None:
    """
    Finds the narrowest row of the TOP garment mask between the shoulder
    and hip y-coordinates (excluding a margin at both ends to avoid
    collar/hem artifacts) — a proxy for waist width.
    """
    h, w = top_mask.shape
    y_start_raw = int(shoulder_y_normalized * h)
    y_end_raw = int(hip_y_normalized * h)

    if y_start_raw >= y_end_raw:
        return None

    full_range = y_end_raw - y_start_raw
    margin_px = int(full_range * 0.15)

    y_start = y_start_raw + margin_px
    y_end = y_end_raw - margin_px

    if y_start >= y_end:
        y_start, y_end = y_start_raw, y_end_raw

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
    """
    Estimates outer hip width from the BOTTOM garment's segmentation mask,
    using a windowed median measurement (see _measure_mask_width_windowed)
    instead of a single row, for robustness against occlusion at the
    exact hip-landmark row.
    """
    return _measure_mask_width_windowed(bottom_mask, hip_y_normalized)


def estimate_shoulder_width(top_mask: np.ndarray, shoulder_y_normalized: float) -> float | None:
    """
    Estimates outer shoulder width from the TOP garment's segmentation
    mask, using the same windowed median approach as estimate_hip_width,
    for methodological consistency.
    """
    return _measure_mask_width_windowed(top_mask, shoulder_y_normalized)