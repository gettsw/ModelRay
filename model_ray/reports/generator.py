"""Audit report: JSON (machine-readable, the source of truth) and a static HTML view."""

from __future__ import annotations

import html
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from model_ray.core.metrics import ProbeResult

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]

LIMITATIONS = [
    "Probes measure internal representations and first-token behaviour on the supplied prompts only; "
    "a PASS is not evidence of safety on other inputs or under adversarial use.",
    "Representation metrics can stay intact while downstream layers stop using them; "
    "pair with generation-based behavioural evals.",
    "Results are only as valid as the prompt sets, layers and SAEs configured; "
    "check config against a known-bad control (e.g. ablate_direction) before trusting a PASS.",
    "Repeatedly tuning until this gate passes optimises against it; keep held-out probe sets.",
    "The weight and tokenizer digests identify artifacts, not deployed behaviour: they do not cover "
    "serving-time system prompts, quantization, sampling settings or runtime hooks.",
]


class ModelRecord(BaseModel):
    id: str
    weights_sha256: str
    tokenizer_sha256: str = ""  # vocab + special tokens + chat template (if used)


class AuditReport(BaseModel):
    """Circuit health report. ``verdict`` is the worst status of any gated result."""

    report_version: str = "0.1"
    verdict: Verdict
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    base: ModelRecord
    tuned: ModelRecord
    config_sha256: str
    environment: dict[str, str] = Field(default_factory=dict)
    results: list[ProbeResult]
    limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))

    @staticmethod
    def verdict_of(results: list[ProbeResult]) -> Verdict:
        statuses = {r.status for r in results}
        if "FAIL" in statuses:
            return "FAIL"
        if "INCONCLUSIVE" in statuses or not (statuses - {"INFO"}):
            return "INCONCLUSIVE"  # nothing gated == nothing verified
        return "PASS"


def environment() -> dict[str, str]:
    import torch

    from importlib.metadata import PackageNotFoundError, version

    env = {"python": platform.python_version(), "torch": torch.__version__}
    for pkg in ("transformer_lens", "sae_lens", "model-ray"):
        try:
            env[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return env


_COLORS = {"PASS": "#1a7f37", "FAIL": "#cf222e", "INCONCLUSIVE": "#9a6700", "INFO": "#57606a"}


def to_html(r: AuditReport) -> str:
    e = lambda x: html.escape(str(x))  # noqa: E731
    f = lambda x: f"{x:.3f}" if isinstance(x, float) else e(x)  # noqa: E731
    rows = "".join(
        f"<tr><td>{e(p.probe)}</td><td>{e(p.metric)}</td><td>{f(p.base_value)}</td><td>{f(p.tuned_value)}</td>"
        f"<td>{f(p.degradation)} [{f(p.ci_low)}, {f(p.ci_high)}] {e(p.unit)}</td><td>{f(p.threshold)}</td>"
        f"<td style='color:{_COLORS[p.status]};font-weight:600'>{p.status}</td><td>{p.n}</td>"
        f"<td>{'<br>'.join(e(i) for i in p.issues)}</td></tr>"
        for p in r.results
    )
    details = "".join(
        f"<h3>{e(p.probe)} / {e(p.metric)}</h3><pre>{e(json.dumps(p.details, indent=2))}</pre>" for p in r.results
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Circuit Health Report</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}}pre{{background:#f6f8fa;padding:8px;overflow:auto}}</style>
</head><body><h1>Circuit Health Report: <span style="color:{_COLORS[r.verdict]}">{r.verdict}</span></h1>
<p>Base <code>{e(r.base.id)}</code> weights <code>{e(r.base.weights_sha256[:16])}</code> tokenizer <code>{e(r.base.tokenizer_sha256[:16])}</code><br>
Tuned <code>{e(r.tuned.id)}</code> weights <code>{e(r.tuned.weights_sha256[:16])}</code> tokenizer <code>{e(r.tuned.tokenizer_sha256[:16])}</code><br>
Config sha256 <code>{e(r.config_sha256[:16])}</code> &middot; {e(r.created_at)} &middot; {e(r.environment)}</p>
<table><tr><th>Probe</th><th>Metric</th><th>Base</th><th>Tuned</th><th>Degradation [95% CI]</th>
<th>Threshold</th><th>Status</th><th>n</th><th>Issues</th></tr>{rows}</table>
<h2>Limitations</h2><ul>{''.join(f'<li>{e(x)}</li>' for x in r.limitations)}</ul>
<h2>Details</h2>{details}</body></html>"""


def write_reports(r: AuditReport, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, h = out / "circuit_health.json", out / "circuit_health.html"
    j.write_text(r.model_dump_json(indent=2), encoding="utf-8")
    h.write_text(to_html(r), encoding="utf-8")
    return j, h
