# Running Outfit Check locally

Two servers run side by side: the FastAPI backend (port 8000) and the React
frontend (port 5173). Run all commands from the repo root unless noted.

## Prerequisites

- Git
- Python **3.12** (on Windows, `py -0` should list 3.12)
- Node.js LTS

## 1. Backend

```powershell
py -3.12 -m venv backend\venv
backend\venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` at the repo root holds the ML dependencies (torch,
mediapipe, transformers, ...). The install is large; expect several minutes.

Optional: create a `.env` file in the repo root for the LLM phrasing step:

```
GROQ_API_KEY=your_key_here
```

Without it, the API still works and returns the rule-based recommendation
text. `.env` is gitignored, so never commit it.

Start the server from the repo root (not from `backend/`, because the code
imports the `ml` package):

```powershell
uvicorn backend.main:app --reload
```

The first start downloads the segmentation model (about 110 MB) from
Hugging Face, so wait for `Application startup complete`. Then check
`http://localhost:8000/health` or try the API at `http://localhost:8000/docs`.

## 2. Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. To point the page at a different backend, set
`VITE_API_URL` (defaults to `http://localhost:8000`).

## API

`POST /analyze` takes one image in a multipart field named `file` and returns
`body_shape`, `body_shape_confidence`, `top_color`, `bottom_color`,
`final_score` (0 to 1), `harmony_relationship`, `recommendations`, and
`llm_phrased_recommendation`. Errors return 400 (not an image) or 422 (no
person, or top and bottom not both found) with a `detail` message.

## Good test photos

Full body in frame, standing straight, plain background, with a clearly
different top and bottom (for example a t-shirt and jeans).