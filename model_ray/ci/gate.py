"""CI regression gate: run probes on base then tuned, judge, report, exit non-zero.

    model-ray audit --base Qwen/Qwen2.5-0.5B-Instruct --tuned ./ckpt --config gate.json

Exit codes: 0 PASS, 1 FAIL (or crash), 3 INCONCLUSIVE. Fail-closed: INCONCLUSIVE fails
the build unless ``--allow-inconclusive``. Models are loaded one at a time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from typing import Callable

import torch
from pydantic import BaseModel, ConfigDict

from model_ray.core.orchestrator import ModelAuditor
from model_ray.core.sae_bridge import SAEFeatureProbe
from model_ray.probes import EvaluationAwarenessProbe, OverwriteDistanceProbe, RefusalStabilityProbe
from model_ray.reports.generator import AuditReport, ModelRecord, environment, write_reports


class CircuitDegradationError(RuntimeError):
    """Raised by ``enforce`` when the gate does not PASS. Carries the report."""

    def __init__(self, report: AuditReport):
        self.report = report
        bad = [f"{r.probe}.{r.metric}={r.status}" for r in report.results if r.status in ("FAIL", "INCONCLUSIVE")]
        super().__init__(f"modelRay gate {report.verdict}: {', '.join(bad)}")


class GateConfig(BaseModel):
    """Gate definition (JSON file for the CLI). Thresholds live on each probe.
    Unknown keys are rejected so a typo cannot silently disable a check."""

    model_config = ConfigDict(extra="forbid")

    refusal: RefusalStabilityProbe | None = None
    sycophancy: OverwriteDistanceProbe | None = None
    eval_awareness: EvaluationAwarenessProbe | None = None
    sae: SAEFeatureProbe | None = None
    chat_template: bool = False
    # None = raw HF numerics; {} or {"center_writing_weights": true, ...} = TransformerLens
    # compatibility-mode processing. Must match how any configured SAE was trained.
    processing: dict[str, bool] | None = None
    batch_size: int = 8
    n_boot: int = 1000

    def probes(self) -> list:
        return [p for p in (self.refusal, self.sycophancy, self.eval_awareness, self.sae) if p is not None]

    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def run_audit(
    cfg: GateConfig,
    load_base: Callable[[], ModelAuditor],
    load_tuned: Callable[[], ModelAuditor],
) -> AuditReport:
    """Measure every probe on base, free it, measure on tuned, then compare."""
    probes = cfg.probes()
    if not probes:
        raise ValueError("GateConfig has no probes configured")
    measured, records = [], []
    for load in (load_base, load_tuned):
        a = load()
        records.append(ModelRecord(id=a.name, weights_sha256=a.weights_sha256(), tokenizer_sha256=a.tokenizer_sha256()))
        measured.append([p.measure(a) for p in probes])
        del a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results = [r for p, b, t in zip(probes, *measured) for r in p.compare(b, t, cfg.n_boot)]
    return AuditReport(
        verdict=AuditReport.verdict_of(results), base=records[0], tuned=records[1],
        config_sha256=cfg.sha256(), environment=environment(), results=results,
    )


def enforce(report: AuditReport, allow_inconclusive: bool = False) -> AuditReport:
    """Raise ``CircuitDegradationError`` unless the report passes (fail-closed)."""
    if report.verdict == "FAIL" or (report.verdict == "INCONCLUSIVE" and not allow_inconclusive):
        raise CircuitDegradationError(report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="model-ray")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="compare a fine-tuned checkpoint against its base")
    a.add_argument("--base", required=True, help="HF hub id or local checkpoint dir")
    a.add_argument("--tuned", required=True, help="HF hub id or local checkpoint dir")
    a.add_argument("--config", required=True, help="GateConfig JSON")
    a.add_argument("--out", default="model_ray_report", help="report directory (default: %(default)s)")
    a.add_argument("--device", default=None, help="e.g. cpu, cuda (default: auto)")
    a.add_argument("--dtype", default="float32", help="torch dtype, e.g. bfloat16 (default: %(default)s)")
    a.add_argument("--allow-inconclusive", action="store_true", help="exit 0 on INCONCLUSIVE instead of 3")
    s = sub.add_parser("sweep", help="per-layer refusal separation on one model, to choose refusal.layer")
    s.add_argument("--model", required=True, help="HF hub id or local checkpoint dir")
    s.add_argument("--config", required=True, help="GateConfig JSON with a refusal section")
    s.add_argument("--device", default=None, help="e.g. cpu, cuda (default: auto)")
    s.add_argument("--dtype", default="float32", help="torch dtype, e.g. bfloat16 (default: %(default)s)")
    args = ap.parse_args(argv)

    try:  # a bad config is a usage error (exit 2), never a FAIL (1) a pipeline would blame on the model
        with open(args.config, encoding="utf-8") as fh:
            cfg = GateConfig.model_validate_json(fh.read())
    except (OSError, ValueError) as e:
        ap.error(f"--config {args.config}: {e}")
    kw = dict(device=args.device, dtype=args.dtype, batch_size=cfg.batch_size, chat_template=cfg.chat_template,
              processing=cfg.processing)
    if args.cmd == "sweep":
        if cfg.refusal is None:
            ap.error("sweep needs a refusal section in --config")
        d = cfg.refusal.sweep(ModelAuditor.from_pretrained(args.model, **kw))
        for L, v in d.items():
            print(f"layer {L:>3}  d'={v:6.2f}  {'#' * max(0, round(v * 5))}")
        best = max(d, key=d.get)
        print(f"peak: layer {best} (d'={d[best]:.2f}); set \"refusal\": {{\"layer\": {best}}}")
        return 0
    report = run_audit(
        cfg,
        lambda: ModelAuditor.from_pretrained(args.base, **kw),
        lambda: ModelAuditor.from_pretrained(args.tuned, **kw),
    )
    j, h = write_reports(report, args.out)
    for r in report.results:
        print(f"{r.status:<12} {r.probe}.{r.metric}: {r.degradation:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] "
              f"{r.unit} (threshold {r.threshold}){'  ' + '; '.join(r.issues) if r.issues else ''}")
    print(f"verdict: {report.verdict}  ->  {j}  {h}")
    try:
        enforce(report, args.allow_inconclusive)
    except CircuitDegradationError:
        return 1 if report.verdict == "FAIL" else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
