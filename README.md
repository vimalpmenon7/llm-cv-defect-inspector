# llm-cv-defect-inspector

![CI](https://github.com/vimalpmenon7/llm-cv-defect-inspector/actions/workflows/ci.yml/badge.svg)

A **computer vision + LLM** pipeline for surface inspection: OpenCV finds defects in an image, then a language model (or an offline template) turns the structured detections into a short inspection report with a pass / review / fail verdict. Served through a FastAPI endpoint.

## Problem

Vision models output boxes and labels; inspectors need a readable report and a decision. LLMs write good reports but can invent facts, so they must only see verified detections.

## Solution

```
image --> detector --> structured detections (JSON) --> report writer --> report + verdict
          (OpenCV or                                     (LLM sees ONLY the JSON;
           optional YOLOv8)                               template fallback if no LLM)
```

The LLM never sees the raw image. It receives the detection list and a system prompt that forbids adding defects, causes, or measurements. If the LLM is not configured or the call fails, a deterministic template report is returned instead, so the service never fails because of the LLM.

## Features

- **Classical detector (works out of the box):** background subtraction with a large median blur, Otsu thresholding, morphology, and contour analysis. Elongated blobs are labelled `scratch`, compact ones `spot`. Severity comes from the fraction of image area covered.
- **Optional YOLOv8 backend:** `YoloDetector` wraps an Ultralytics model if you supply your own weights.
- **Grounded reporting:** structured-input prompting plus offline template fallback.
- **Verdict logic:** any high-severity defect gives `fail`; any other defect gives `review`; none gives `pass`.
- **API:** `POST /analyze` (image upload, 10 MB limit) and `GET /health`.
- **Evaluation:** precision/recall on synthetic images with known ground truth.

## Tech stack

Python, OpenCV, NumPy, FastAPI, Docker, GitHub Actions, unittest/pytest. Optional: Ultralytics YOLOv8, any OpenAI-compatible LLM endpoint.

## Project structure

```
inspector/
  detector.py    ContourDetector, YoloDetector (optional), Detection
  report.py      template report + OpenAI-compatible LLM reporter
  pipeline.py    analyze_image(), verdict logic, image decoding
  api.py         FastAPI app
  synthetic.py   synthetic defect images
  evaluate.py    precision / recall on synthetic data
tests/           14 unit tests
```

## Installation

```bash
git clone https://github.com/vimalpmenon7/llm-cv-defect-inspector.git
cd llm-cv-defect-inspector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Configuration

Optional. To use an LLM for the report, set these (see `.env.example`); never commit real keys.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible base URL |
| `LLM_API_KEY` | API key |
| `LLM_MODEL` | Model name |

## Usage

Python:

```python
import cv2
from inspector import analyze_image

result = analyze_image(cv2.imread("panel.png"))
print(result.verdict)
print(result.report)
```

API:

```bash
uvicorn inspector.api:app --reload
curl -F "file=@panel.png" localhost:8000/analyze
```

Example report (template mode, from `make_surface(1, 1, seed=4)` in `inspector/synthetic.py`):

```
Detected 2 defect(s): 1 spot, 1 scratch. Highest severity: medium.
1. spot at x=152, y=96 (23x23px), severity medium, confidence 0.55.
2. scratch at x=101, y=47 (61x4px), severity low, confidence 0.47.
Recommended action: Schedule manual review.
```

## Evaluation

```bash
python -m inspector.evaluate
```

On 50 synthetic images (one-two scratches/spots each, IoU >= 0.3): **precision 1.00, recall 0.96**.

This only validates the pipeline on clean synthetic data. It says nothing about accuracy on real photographs, which have texture, lighting changes and reflections.

## Testing

```bash
python -m pytest -q          # or: python -m unittest discover -s tests -t .
```

## Docker

```bash
docker build -t llm-cv-defect-inspector .
docker run -p 8000:8000 llm-cv-defect-inspector
```

## Limitations

- The default detector is a classical heuristic tuned for dark defects on a smooth surface; it will not generalise to arbitrary products.
- The YOLOv8 backend is untested here because no trained weights are bundled.
- Severity is based on area only, not on defect type or position.
- No authentication or request rate limiting on the API.

## Future improvements

- Train and evaluate YOLOv8 on a public defect dataset (for example NEU surface defects) and report mAP.
- Send the image to a vision-language model for a second opinion and compare with the detector.
- Annotated-image output and batch endpoints.

## License

MIT
