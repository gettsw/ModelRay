import torch

from model_ray.probes import (
    ContextItem,
    EvaluationAwarenessProbe,
    OverwriteDistanceProbe,
    RefusalStabilityProbe,
    SycophancyItem,
    ablate_direction,
)
from model_ray.probes.eval_awareness import split
from tests.conftest import HARMFUL, HARMLESS


def refusal_direction(a, probe):
    m = probe.measure(a)
    return m["harmful"].mean(0) - m["harmless"].mean(0)


# -- refusal --------------------------------------------------------------


def test_refusal_identical_model_passes(make_auditor):
    p = RefusalStabilityProbe(harmful=HARMFUL, harmless=HARMLESS, layer=2)
    b, t = p.measure(make_auditor()), p.measure(make_auditor())
    (r,) = p.compare(b, t, n_boot=200)
    assert r.status == "PASS" and abs(r.degradation) < 1e-6 and r.base_value > 1


def test_refusal_ablated_model_fails(make_auditor):
    """Positive control: removing the refusal direction must FAIL the gate."""
    p = RefusalStabilityProbe(harmful=HARMFUL, harmless=HARMLESS, layer=2)
    base = make_auditor()
    tuned = make_auditor()
    ablate_direction(tuned.model, refusal_direction(base, p))
    (r,) = p.compare(p.measure(base), p.measure(tuned), n_boot=200)
    assert r.status == "FAIL" and r.ci_low > 0.9


def test_refusal_without_base_signal_is_inconclusive(make_auditor):
    p = RefusalStabilityProbe(harmful=HARMLESS, harmless=HARMLESS[::-1], layer=2)
    a = make_auditor()
    (r,) = p.compare(p.measure(a), p.measure(a), n_boot=50)
    assert r.status == "INCONCLUSIVE" and "no refusal signal" in r.issues[0]


def test_refusal_token_metric_is_reported(make_auditor):
    p = RefusalStabilityProbe(harmful=HARMFUL, harmless=HARMLESS, layer=1, refusal_tokens=["I", "S"])
    m = p.measure(make_auditor())
    assert m["refusal_prob"].shape == (len(HARMFUL),) and (m["refusal_prob"] > 0).all()
    results = p.compare(m, m, n_boot=50)
    assert [r.metric for r in results] == ["direction_separation", "refusal_token_prob"]
    assert results[1].status == "PASS"


# -- sycophancy -----------------------------------------------------------


def _syc(neutral_final, pressured_final, n_layers=3):
    """Fake measurement: logit diffs [items, layers] with given final-layer values."""
    neutral = torch.ones(len(neutral_final), n_layers)
    pressured = torch.ones(len(pressured_final), n_layers)  # mid layers favour correct
    neutral[:, -1], pressured[:, -1] = torch.tensor(neutral_final), torch.tensor(pressured_final)
    return {"neutral": neutral, "pressured": pressured}


ITEMS = [SycophancyItem(neutral="q?", pressured="q? I think B.", correct="A", suggested="B")] * 8


def test_sycophancy_caving_fails():
    p = OverwriteDistanceProbe(items=ITEMS)
    base = _syc([1.0] * 8, [1.0] * 8)  # knows, holds
    tuned = _syc([1.0] * 8, [-1.0] * 8)  # knows, caves
    (r,) = p.compare(base, tuned, n_boot=100)
    assert r.status == "FAIL" and r.degradation == 1.0
    assert r.details["tuned_overwrite_margin"] == 2.0  # internal +1 overwritten to -1


def test_sycophancy_capability_collapse_is_not_a_pass():
    """A tuned model that no longer knows the answers cannot cave; that must not look safe."""
    p = OverwriteDistanceProbe(items=ITEMS)
    (r,) = p.compare(_syc([1.0] * 8, [1.0] * 8), _syc([-1.0] * 8, [-1.0] * 8), n_boot=100)
    assert r.status == "INCONCLUSIVE"


def test_sycophancy_measure_shapes(make_auditor):
    p = OverwriteDistanceProbe(items=ITEMS, min_layer=1)
    m = p.measure(make_auditor())
    assert m["neutral"].shape == m["pressured"].shape == (8, 3)


# -- evaluation awareness -------------------------------------------------


def _ctx(n=120):
    return [ContextItem(text=f"t{i}", is_eval=i % 2 == 0, group=f"g{i % 8}") for i in range(n)]


def test_group_split_holds_out_whole_groups():
    tr, te, grouped = split(_ctx(), seed=0)
    items = _ctx()
    assert grouped and not ({items[i].group for i in tr} & {items[i].group for i in te})


def test_eval_awareness_selectivity_increase():
    items = _ctx()
    y = torch.tensor([i.is_eval for i in items]).float()
    g = torch.Generator().manual_seed(0)
    noise = lambda: torch.randn(len(items), 16, generator=g)  # noqa: E731
    sep = noise()
    sep[:, 0] += 6 * (y - 0.5)  # linearly decodable at the probe layer
    embed = noise()  # layer 0 is ~unchanged by fine-tuning; independent noise here makes CIs straddle
    base = {"embed": embed, "probe": noise()}
    tuned = {"embed": embed, "probe": sep}
    (r,) = EvaluationAwarenessProbe(items=items).compare(base, tuned, n_boot=100)
    assert r.status == "INFO" and r.degradation > 0.3 and r.details["group_held_out"]
    (r,) = EvaluationAwarenessProbe(items=items, max_increase=0.1).compare(base, tuned, n_boot=100)
    assert r.status == "FAIL"


def test_eval_awareness_measure_shapes(make_auditor):
    p = EvaluationAwarenessProbe(items=_ctx(8))
    m = p.measure(make_auditor())
    assert m["embed"].shape == m["probe"].shape == (8, 32)


def test_sycophancy_at_ceiling_is_not_a_pass():
    """Seen on Qwen2.5-0.5B-Instruct: the base already caves on every item, so no increase is possible."""
    p = OverwriteDistanceProbe(items=ITEMS)
    caving = _syc([1.0] * 8, [-1.0] * 8)
    (r,) = p.compare(caving, caving, n_boot=50)
    assert r.status == "INCONCLUSIVE" and "ceiling" in r.issues[-1]


def test_below_chance_baseline_does_not_inflate_selectivity():
    """Seen on Qwen: held-out groups flipped the layer-0 probe to 0% accuracy, which read as +0.5 selectivity."""
    items = _ctx()
    y = torch.tensor([i.is_eval for i in items]).float()
    g = torch.Generator().manual_seed(0)
    probe = torch.randn(len(items), 16, generator=g)
    probe[:, 0] += 6 * (y - 0.5)  # probe layer decodes perfectly
    _, te, _ = split(items, seed=0)
    embed = torch.randn(len(items), 16, generator=g)
    # layer-0 feature is +label on train groups and -label on test groups => ~0% held-out acc
    sign = torch.ones(len(items)); sign[te] = -1
    embed[:, 0] += 6 * (y - 0.5) * sign
    m = {"embed": embed, "probe": probe}
    (r,) = EvaluationAwarenessProbe(items=items).compare(m, m, n_boot=20)
    assert r.details["base_embed_acc"] < 0.1
    assert r.base_value <= 0.5 + 1e-6  # 1.0 - max(~0, chance=0.5), not 1.0 - 0.0


def test_is_refusal_judge():
    from model_ray.probes.refusal import is_refusal

    assert is_refusal("I’m sorry, but I can't assist with that.") and is_refusal("As an AI, I must decline.")
    assert not is_refusal("Hot-wiring a car is a process that involves") and not is_refusal("Certainly! Here is")


def test_generated_refusal_drop_fails():
    """Model still starts with a refusal-ish token but complies: only full-text judging catches it."""
    p = RefusalStabilityProbe(harmful=HARMFUL, harmless=HARMLESS, layer=2, generate_tokens=8)
    feats = torch.randn(16, 4, generator=torch.Generator().manual_seed(0))
    feats[:, 0] += 5
    m = {"harmful": feats, "harmless": torch.randn(16, 4)}
    base, tuned = dict(m, refused=torch.ones(16)), dict(m, refused=torch.zeros(16))
    gen = [r for r in p.compare(base, tuned, n_boot=50) if r.metric == "generated_refusal_rate"][0]
    assert gen.status == "FAIL" and gen.degradation == 1.0


def test_generation_and_sweep_run_on_bridge(make_auditor):
    a = make_auditor()
    p = RefusalStabilityProbe(harmful=HARMFUL[:4], harmless=HARMLESS[:4], layer=1, generate_tokens=2)
    assert p.measure(a)["refused"].shape == (4,)
    sweep = RefusalStabilityProbe(harmful=HARMFUL, harmless=HARMLESS).sweep(a)
    assert sorted(sweep) == [0, 1, 2, 3]
