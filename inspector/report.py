"""Turn structured detections into a human-readable inspection report.

The LLM only receives the structured detection list and is told not to add facts. The
offline template report is the default and the fallback if the LLM call fails.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .detector import Detection

SYSTEM_PROMPT = (
    "You write short industrial inspection reports. Use ONLY the JSON detections provided. "
    "Do not invent defects, causes, measurements or locations. Mention each severity level "
    "present and end with a one-sentence recommended action."
)


def _plural(label: str, n: int) -> str:
    if n == 1:
        return label
    return label + ("es" if label.endswith(("ch", "s", "x")) else "s")


def template_report(detections: list[Detection]) -> str:
    """Deterministic report; no model required."""
    if not detections:
        return "No defects detected. Surface passed automated inspection."
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    parts = ", ".join(f"{n} {_plural(label, n)}" for label, n in counts.items())
    worst = max(detections, key=lambda d: {"low": 0, "medium": 1, "high": 2}[d.severity])
    action = {
        "high": "Reject or escalate for manual inspection.",
        "medium": "Schedule manual review.",
        "low": "Log and monitor.",
    }[worst.severity]
    lines = [f"Detected {len(detections)} defect(s): {parts}. Highest severity: {worst.severity}."]
    for i, d in enumerate(detections, start=1):
        x, y, w, h = d.bbox
        lines.append(
            f"{i}. {d.label} at x={x}, y={y} ({w}x{h}px), severity {d.severity}, "
            f"confidence {d.confidence:.2f}."
        )
    lines.append(f"Recommended action: {action}")
    return "\n".join(lines)


class OpenAICompatibleReporter:
    """LLM report writer using any OpenAI-compatible endpoint (stdlib only)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.timeout = timeout
        if not (self.base_url and self.model):
            raise ValueError("LLM_BASE_URL and LLM_MODEL must be set")

    def write(self, detections: list[Detection]) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps([d.to_dict() for d in detections])},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


def write_report(detections: list[Detection], use_llm: bool = True) -> tuple[str, str]:
    """Return ``(report, source)`` where source is ``"llm"`` or ``"template"``."""
    if use_llm and os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"):
        try:
            return OpenAICompatibleReporter().write(detections), "llm"
        except (RuntimeError, ValueError):
            pass
    return template_report(detections), "template"
