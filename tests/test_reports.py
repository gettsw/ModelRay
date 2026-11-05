import json

from model_ray.core.metrics import ProbeResult
from model_ray.reports.generator import AuditReport, ModelRecord, to_html, write_reports


def _report(model_id="org/model"):
    r = ProbeResult(probe="refusal_stability", metric="direction_separation", unit="relative drop",
                    base_value=3.0, tuned_value=1.0, degradation=0.66, ci_low=0.5, ci_high=0.8,
                    threshold=0.15, n=32)
    rec = ModelRecord(id=model_id, weights_sha256="ab" * 32)
    return AuditReport(verdict="FAIL", base=rec, tuned=rec, config_sha256="cd" * 32, results=[r])


def test_html_escapes_untrusted_ids():
    page = to_html(_report("<script>alert(1)</script>"))
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_json_report_carries_status_and_limitations(tmp_path):
    j, h = write_reports(_report(), tmp_path)
    data = json.loads(j.read_text())
    assert data["verdict"] == "FAIL" and data["results"][0]["status"] == "FAIL"
    assert len(data["limitations"]) >= 3 and h.read_text().startswith("<!doctype html>")
