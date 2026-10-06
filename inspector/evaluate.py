"""Measure detector recall/precision on synthetic images with known defects (IoU >= 0.3).

Usage: ``python -m inspector.evaluate``. This validates the pipeline, not real-world accuracy.
"""

from __future__ import annotations

from .detector import ContourDetector
from .synthetic import make_surface


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def evaluate(n_images: int = 50, threshold: float = 0.3) -> dict[str, float]:
    detector = ContourDetector()
    tp = fp = fn = 0
    for seed in range(n_images):
        image, truth = make_surface(seed=seed)
        preds = [d.bbox for d in detector.detect(image)]
        matched: set[int] = set()
        for t in truth:
            best = max(range(len(preds)), key=lambda i: iou(t, preds[i]), default=None)
            if best is not None and iou(t, preds[best]) >= threshold and best not in matched:
                matched.add(best)
                tp += 1
            else:
                fn += 1
        fp += len(preds) - len(matched)
    return {
        "images": n_images,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
    }


if __name__ == "__main__":
    print(evaluate())
