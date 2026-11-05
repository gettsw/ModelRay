import json
import math

import pytest

from model_ray.ci import gate
from model_ray.ci.gate import CircuitDegradationError, GateConfig, enforce, run_audit
from model_ray.core.metrics import ProbeResult
from model_ray.probes import RefusalStabilityProbe, ablate_direction
from model_ray.reports.generator import AuditReport
from tests.conftest import HARMFUL, HARMLESS


def _cfg():
    return GateConfig(refusal=RefusalStabilityProbe(harmful=HARMFUL, harmless=HARMLESS, layer=2), n_boot=200)


def _ablated(make_auditor, cfg):
    base, tuned = make_auditor(), make_auditor(name="abliterated")
    m = cfg.refusal.measure(base)
    ablate_direction(tuned.model, m["harmful"].mean(0) - m["harmless"].mean(0))
    return tuned


def test_identical_checkpoint_passes(make_auditor):
    report = enforce(run_audit(_cfg(), make_auditor, make_auditor))
    assert report.verdict == "PASS" and report.base.weights_sha256 == report.tuned.weights_sha256


def test_abliterated_checkpoint_raises(make_auditor):
    cfg = _cfg()
    tuned = _ablated(make_auditor, cfg)
    report = run_audit(cfg, make_auditor, lambda: tuned)
    with pytest.raises(CircuitDegradationError, match="refusal_stability.direction_separation=FAIL"):
        enforce(report)


@pytest.mark.parametrize(
    "lo,hi,thr,issues,expected",
    [
        (0.20, 0.30, 0.15, [], "FAIL"),
        (0.00, 0.10, 0.15, [], "PASS"),
        (0.10, 0.20, 0.15, [], "INCONCLUSIVE"),  # CI straddles threshold
        (math.nan, math.nan, 0.15, [], "INCONCLUSIVE"),
        (0.00, 0.10, 0.15, ["bad data"], "INCONCLUSIVE"),
        (0.50, 0.60, None, [], "INFO"),
    ],
)
def test_status_rule(lo, hi, thr, issues, expected):
    r = ProbeResult(probe="p", metric="m", unit="u", base_value=1, tuned_value=1, degradation=(lo + hi) / 2,
                    ci_low=lo, ci_high=hi, threshold=thr, n=10, issues=issues)
    assert r.status == expected


def test_nothing_gated_is_not_a_pass():
    info = ProbeResult(probe="p", metric="m", unit="u", base_value=0, tuned_value=0, degradation=0,
                       ci_low=0, ci_high=0, threshold=None, n=1)
    assert AuditReport.verdict_of([info]) == "INCONCLUSIVE"


def test_cli_exit_codes_and_reports(make_auditor, tmp_path, monkeypatch):
    cfg = _cfg()
    tuned = _ablated(make_auditor, cfg)
    models = {"base": make_auditor(name="base"), "good": make_auditor(name="good"), "bad": tuned}
    monkeypatch.setattr(gate.ModelAuditor, "from_pretrained", staticmethod(lambda ckpt, **kw: models[ckpt]))
    (tmp_path / "gate.json").write_text(cfg.model_dump_json())
    args = ["audit", "--base", "base", "--config", str(tmp_path / "gate.json"), "--out", str(tmp_path / "out")]
    assert gate.main(args + ["--tuned", "good"]) == 0
    assert gate.main(args + ["--tuned", "bad"]) == 1
    report = json.loads((tmp_path / "out" / "circuit_health.json").read_text())
    assert report["verdict"] == "FAIL" and report["tuned"]["id"] == "abliterated"
    assert (tmp_path / "out" / "circuit_health.html").exists()


def test_config_typo_is_rejected():
    raw = json.loads(_cfg().model_dump_json())
    raw["refusal"]["max_dorp"] = 0.5
    with pytest.raises(ValueError, match="max_dorp"):
        GateConfig.model_validate(raw)


def test_cli_sweep_prints_peak(make_auditor, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate.ModelAuditor, "from_pretrained", staticmethod(lambda ckpt, **kw: make_auditor()))
    (tmp_path / "gate.json").write_text(_cfg().model_dump_json())
    assert gate.main(["sweep", "--model", "m", "--config", str(tmp_path / "gate.json")]) == 0
    out = capsys.readouterr().out
    assert out.count("layer ") == 5 and "peak: layer" in out
