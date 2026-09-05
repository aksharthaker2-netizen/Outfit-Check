"""
main.py — FastAPI backend wrapping the Outfit Check ML pipeline.

Key design point: the GarmentSegmenter (SegFormer model) is loaded ONCE
at server startup, not per-request. Loading a transformer model takes
several seconds; reloading it on every request would make each API call
unacceptably slow. This is the standard "warm model" pattern for served
ML — contrasted with the notebook scripts, which reload the model every
run since that cost doesn't matter for a one-off CLI invocation.
"""

import shutil
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2

from ml.src.pose import PoseDetector
from ml.src.anthropometry import (
    classify_body_shape_with_waist,
    estimate_waist_width,
    estimate_hip_width,
    estimate_shoulder_width,
)
from ml.src.segmentation import GarmentSegmenter, GarmentCategory, extract_dominant_colors
from ml.src.scoring import score_outfit
from ml.src.recommendations import generate_recommendations, phrase_recommendations

app = FastAPI(title="Outfit Check API")

# Allow the React frontend (running on a different port during dev) to
# call this API. Restrict origins in production instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at import time (i.e. once when the server process starts),
# reused across every request — see module docstring.
segmenter = GarmentSegmenter()


class OutfitAnalysisResponse(BaseModel):
    body_shape: str
    body_shape_confidence: float
    top_color: list[int]
    bottom_color: list[int]
    final_score: float
    harmony_relationship: str
    recommendations: list[str]
    llm_phrased_recommendation: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=OutfitAnalysisResponse)
async def analyze_outfit(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Save the upload to a temp file — the pipeline's functions take a
    # file path, matching how the notebook scripts already work, so no
    # changes needed to any ml/src/ code.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        frame = cv2.imread(tmp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image file.")

        with PoseDetector(static_image_mode=True) as detector:
            pose_result = detector.process(frame)

        if not pose_result.detected:
            raise HTTPException(status_code=422, detail="No person detected in image.")

        seg_result = segmenter.segment(tmp_path)

        if GarmentCategory.TOP not in seg_result.masks or GarmentCategory.BOTTOM not in seg_result.masks:
            raise HTTPException(status_code=422, detail="Could not detect both a top and bottom garment.")

        landmarks = pose_result.landmarks
        left_shoulder, right_shoulder = landmarks.get(11), landmarks.get(12)
        left_hip, right_hip = landmarks.get(23), landmarks.get(24)

        waist_estimate = None
        hip_width_from_mask = None
        shoulder_width_from_mask = None

        if all(lm is not None for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2

            waist_estimate = estimate_waist_width(seg_result.masks[GarmentCategory.TOP], shoulder_y, hip_y)
            hip_width_from_mask = estimate_hip_width(seg_result.masks[GarmentCategory.BOTTOM], hip_y)
            shoulder_width_from_mask = estimate_shoulder_width(seg_result.masks[GarmentCategory.TOP], shoulder_y)

        shape_result = classify_body_shape_with_waist(
            pose_result, waist_estimate, hip_width_from_mask, shoulder_width_from_mask
        )

        top_rgb = extract_dominant_colors(tmp_path, seg_result.masks[GarmentCategory.TOP], k=1)[0].rgb
        bottom_rgb = extract_dominant_colors(tmp_path, seg_result.masks[GarmentCategory.BOTTOM], k=1)[0].rgb

        score_result = score_outfit(shape_result.shape, top_rgb, bottom_rgb)
        recommendations = generate_recommendations(score_result, shape_result.shape)
        llm_text = phrase_recommendations(recommendations)

        return OutfitAnalysisResponse(
            body_shape=shape_result.shape.value,
            body_shape_confidence=shape_result.confidence,
            top_color=list(top_rgb),
            bottom_color=list(bottom_rgb),
            final_score=score_result.final_score,
            harmony_relationship=score_result.relationship,
            recommendations=[r.message for r in recommendations],
            llm_phrased_recommendation=llm_text,
        )

    finally:
        os.unlink(tmp_path)