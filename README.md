# Outfit Check

**An ML-driven outfit rating system that scores an outfit against the wearer's own body — not a generic style rulebook.**

Outfit Check takes a full-body photo, derives the wearer's body-shape category from pose landmarks, isolates and analyzes the garments being worn, and produces a **person-conditioned outfit score** plus targeted improvement recommendations. The core idea: the same shirt and jeans should score differently on two people with different body shapes — this system fuses garment features with the wearer's body shape before scoring, rather than rating clothes in a vacuum.

The project has two halves of equal weight: an **ML/CV pipeline** that does the actual seeing and scoring, and a **full-stack web app** that wraps it behind a usable interface — upload a photo or capture one live, get a scored, explained result back in seconds.

---

## How it works

```
Photo
  │
  ▼
┌────────────────────────┐
│ 1. Pose Check          │  MediaPipe — confirms a full body is in frame,
│                         │  extracts pose landmarks
└──────────┬──────────────┘
           ▼
┌────────────────────────┐
│ 2. Body-Shape           │  Shoulder/waist/hip ratios from landmarks →
│    Classification       │  mapped to styling taxonomy (hourglass, pear,
│                         │  rectangle, inverted-triangle, apple)
└──────────┬──────────────┘
           ▼
┌────────────────────────┐
│ 3. Garment              │  SegFormer (fine-tuned for clothing) — isolates
│    Segmentation         │  top and bottom
└──────────┬──────────────┘
           ▼
┌────────────────────────┐
│ 4. Feature Extraction   │  Dominant color per garment
└──────────┬──────────────┘
           ▼
┌────────────────────────┐
│ 5. Person-Conditioned   │  Fuses garment colors with body-shape before
│    Scoring               scoring — the same outfit rates differently
│                         │  on a different body shape
└──────────┬──────────────┘
           ▼
┌────────────────────────┐
│ 6. Recommendations      │  Rule-based suggestions, then rephrased into
│                         │  natural language by an LLM (Groq)
└──────────┬──────────────┘
           ▼
  Score + Body Shape + Color Palette + Recommendations
           │
           ▼
┌────────────────────────┐
│ 7. API + Web App        │  FastAPI wraps the pipeline behind POST
│                         │  /analyze; a React app uploads or captures a
│                         │  photo and renders the result
└────────────────────────┘
```

## Why this is hard (and interesting)

Most "outfit rating" demos score garments in isolation — a neural net looks at a photo of an outfit and outputs a number based on aesthetics alone. That skips the actual styling problem: **fit and suitability are relative to the wearer's body**, not absolute properties of the garment. This project treats scoring as a conditional problem — `P(good outfit | garment colors, body shape)` — which means the pipeline has to correctly derive body-shape from a single photo (no scale, no measuring tape, no manual input) before scoring can even begin.

The full-stack half has its own real problem to solve: the pipeline depends on heavyweight ML libraries (PyTorch, MediaPipe, a segmentation model downloaded at startup) that have to run synchronously inside a web request, with proper error handling for the ways a photo can fail (no person detected, garments not found) surfaced as clear messages rather than server crashes.

## Tech stack

| Layer | Choice |
|---|---|
| ML / CV | PyTorch, MediaPipe (pose), Hugging Face Transformers (SegFormer segmentation), scikit-learn |
| Recommendation phrasing | Groq (`openai/gpt-oss-120b`) |
| Backend | FastAPI (Python 3.12) |
| Frontend | React + Vite |

## Project structure

```
ml/
├── src/
│   ├── pose/              # pose detection + full-body framing check (P1)
│   ├── anthropometry/      # height estimation, body-shape classification (P2)
│   ├── segmentation/       # garment segmentation + dominant color extraction (P3)
│   ├── scoring/            # person-conditioned outfit scoring (P4)
│   └── recommendations/    # rule-based + LLM-phrased recommendations (P5)
└── notebooks/              # P1-P5 stage-by-stage scripts, run independently
                             # during development

backend/
├── main.py                 # FastAPI app — GET /health, POST /analyze,
│                            # wraps the full ml/ pipeline (P6)
└── requirements.txt

frontend/
├── src/
│   ├── App.jsx              # upload + live camera capture, results view
│   ├── api.js                # POST /analyze client
│   └── App.css
└── index.html

docs/
├── concepts.md              # ML/CV design decisions, per pipeline stage
├── architecture.md
└── running-locally.md       # how to run backend + frontend together
```

## API

`POST /analyze` — multipart form, one image field named `file`.

Returns:

| Field | Description |
|---|---|
| `body_shape` | Classified body-shape category |
| `body_shape_confidence` | Confidence of that classification |
| `top_color`, `bottom_color` | Dominant RGB color per garment |
| `final_score` | Outfit score, 0–1 |
| `harmony_relationship` | Color relationship between the two garments |
| `recommendations` | Rule-based improvement suggestions |
| `llm_phrased_recommendation` | The same suggestions, rephrased in natural language (falls back to the rule-based text if no `GROQ_API_KEY` is set) |

Errors return HTTP 400 (not an image) or 422 (no person, or top/bottom not both detected) with a `detail` message.

See `docs/running-locally.md` to run both servers locally.

## Build order

1. **P1** — Pose check (full-body framing, landmark extraction)
2. **P2** — Body-shape classification
3. **P3** — Garment segmentation + color extraction + color harmony
4. **P4** — Person-conditioned outfit scoring
5. **P5** — Recommendations (rules → LLM-phrased)
6. **P6** — Full-stack web app (FastAPI + React), wrapping the finished pipeline

## Datasets referenced

| Purpose | Dataset |
|---|---|
| Segmentation / detection / pose | [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) |
| Body parsing + keypoints | DeepFashion-MultiModal |
| Fine-grained garment attributes | Fashionpedia |

## Status

✅ ML pipeline (P1–P5) and full-stack wrapper (P6) complete and working end-to-end. See `docs/concepts.md` for the ML/CV design log.