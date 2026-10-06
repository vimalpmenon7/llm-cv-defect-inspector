"""FastAPI service. Run: ``uvicorn inspector.api:app --reload``."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from .pipeline import analyze_image, decode_image

app = FastAPI(title="llm-cv-defect-inspector", version="0.1.0")
MAX_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 10 MB")
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analyze_image(image).to_dict()
