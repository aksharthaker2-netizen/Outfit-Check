# Outfit Check

**An ML-driven outfit rating system that scores an outfit against the wearer's own body — not a generic style rulebook.**

Outfit Check takes a guided full-body scan (video/photo), derives the wearer's height and body-shape category from pose landmarks, isolates and analyzes the garments being worn, and produces a **person-conditioned outfit score with confidence** plus targeted improvement recommendations. The core idea: the same shirt and jeans should score differently on two people with different body shapes — this system fuses garment features with wearer attributes before scoring, rather than rating clothes in a vacuum.

Built as a deep-dive ML/CV project — every pipeline stage is implemented and validated independently (via scripts/notebooks) before any web layer is added, with the goal of demonstrating real understanding of the modeling choices, not just shipping a demo.

---

## How it works

```
Video/Photo Scan
      │
      ▼
┌─────────────────────┐
│ 1. Pose Check        │  MediaPipe/YOLOv8-pose — confirms full body in frame,
│                       │  live directional guidance, extracts landmarks
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 2. Height Estimation  │  Reference-object calibration → pixel-to-cm ratio
│                       │  → real height from landmark span
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 3. Body-Shape         │  Shoulder/waist/hip/torso-leg ratios from landmarks
│    Classification     │  → mapped to styling taxonomy (hourglass, pear,
│                       │  rectangle, inverted-triangle, apple)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 4. Garment            │  DeepFashion2-based / fine-tuned SAM — isolates
│    Segmentation       │  top / bottom / dress / shoes
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 5. Feature Extraction │  Dominant colors per garment (k-means)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 6. Person-Conditioned │  Fuses garment features with (gender, age, height,
│    Scoring            │  body-shape) — same garment scores differently
│                       │  per wearer. v1: rule-based fit/silhouette
│                       │  matching. v2: learned fusion model.
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 7. Recommendations    │  Tailored improvement suggestions (color gaps,
│                       │  silhouette fit). v1: rules. v2: LLM-generated.
└─────────────────────┘
          │
          ▼
   Score + Confidence % + Recommendations
```

## Why this is hard (and interesting)

Most "outfit rating" demos score garments in isolation — a neural net looks at a photo of an outfit and outputs a number based on aesthetics alone. That skips the actual styling problem: **fit and suitability are relative to the wearer's body**, not absolute properties of the garment. This project treats scoring as a conditional problem — `P(good outfit | garment features, body attributes)` — which means the pipeline has to correctly derive body attributes from a single guided scan (no scale, no measuring tape) before scoring can even begin. That anthropometry step (calibrated height + landmark-ratio body-shape classification) is as much the core ML work here as the garment segmentation is.

## Tech stack

| Layer | Choice |
|---|---|
| ML / CV | PyTorch, OpenCV, MediaPipe / YOLOv8-pose |
| Backend | FastAPI (Python) |
| Frontend | React |
| Training compute | Local CPU for light experimentation; Google Colab (GPU) for segmentation fine-tuning and fusion-model training |
| Storage | User profiles, scan history, feedback labels for retraining |

## Datasets

| Purpose | Dataset | Scale |
|---|---|---|
| Segmentation / detection / pose | [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) | 491K images, 801K items |
| Body parsing + keypoints (height/body-shape) | DeepFashion-MultiModal | 44K images (12.7K full-body) |
| Outfit compatibility | Polyvore Outfits | 68K outfits |
| Fine-grained garment attributes | Fashionpedia | 48K images |

*Occasion/companion-context scoring is intentionally descoped — no public dataset exists for it yet.*

## Project structure

```
ml/            # all model/pipeline work — notebooks, source modules, Colab training, tests
backend/       # FastAPI service wrapping the finished pipeline (built last)
frontend/      # React client (built last)
docs/          # architecture notes + running log of ML concepts learned per phase
```

Full ML pipeline is built and validated stage-by-stage in `ml/` — the web app is the final phase, wrapping a working, tested model rather than being built alongside it.

## Build order

1. **P1** — Pose check (full-body framing, landmark extraction)
2. **P2** — Height estimation + body-shape classification
3. **P3** — Garment segmentation + color extraction + rule-based color harmony
4. **P4** — Person-attribute-conditioned scoring
5. **P5** — Recommendation layer (rules → LLM)
6. **P6** — Full-stack web app (FastAPI + React)

## Status

🚧 In active development — ML pipeline phase. See `docs/concepts.md` for a running log of design decisions and the ML/CV concepts behind each stage.
