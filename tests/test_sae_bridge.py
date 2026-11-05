import torch

from model_ray.core.sae_bridge import MockSAE, SAEFeatureProbe, model_diff
from tests.conftest import HARMFUL


def test_model_diff_finds_shifted_feature_only():
    g = torch.Generator().manual_seed(0)
    base = torch.rand(200, 64, generator=g)
    tuned = base + 0.05 * torch.randn(200, 64, generator=g)
    tuned[:, 7] += 1.0
    top = model_diff(base, tuned).top(k=5)
    assert [t["feature"] for t in top] == [7]  # BH keeps the 63 noise features out


def test_model_diff_no_shift_reports_nothing():
    base = torch.rand(100, 64, generator=torch.Generator().manual_seed(1))
    assert model_diff(base, base.clone()).top() == []


def _probe(**kw):
    return SAEFeatureProbe(prompts=HARMFUL, hook="blocks.1.hook_resid_post", sae=MockSAE(32, 128), **kw)


def test_sae_probe_on_identical_models_passes(make_auditor):
    p = _probe(max_base_nmse=10.0)  # random SAE reconstructs badly; that check is tested below
    m = p.measure(make_auditor())
    assert m["feats"].shape == (len(HARMFUL), 128)
    p.feature_ids = m["feats"].mean(0).topk(3).indices.tolist()
    (r,) = p.compare(m, m, n_boot=50)
    assert r.status == "PASS" and r.degradation == 0.0


def test_sae_that_stops_reconstructing_is_inconclusive():
    feats = torch.rand(8, 128)
    base = {"feats": feats, "err": torch.full((8,), 0.1), "sq": torch.ones(8)}
    tuned = {"feats": feats * 0.1, "err": torch.full((8,), 0.5), "sq": torch.ones(8)}  # big "drop"...
    (r,) = _probe(feature_ids=[0, 1]).compare(base, tuned, n_boot=50)
    assert r.status == "INCONCLUSIVE" and "does not transfer" in r.issues[0]  # ...but untrustworthy


def test_processed_sae_on_raw_model_fails_fast(make_auditor):
    """Seen on gpt2-small-res-jb: raw-HF NMSE 4.6 vs 0.056 processed. Refuse before wasting compute."""
    import types

    import pytest

    sae = MockSAE(32, 128)
    sae.cfg = types.SimpleNamespace(metadata=types.SimpleNamespace(model_from_pretrained_kwargs={"center_writing_weights": True}))
    p = SAEFeatureProbe(prompts=HARMFUL, hook="blocks.1.hook_resid_post", sae=sae)
    with pytest.raises(ValueError, match='"processing"'):
        p.measure(make_auditor())
    a = make_auditor()
    a.processing = {}
    assert p.measure(a)["feats"].shape == (len(HARMFUL), 128)
