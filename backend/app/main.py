from fastapi import FastAPI

app = FastAPI(title="Outfit Check API")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Outfit Check backend is alive"}