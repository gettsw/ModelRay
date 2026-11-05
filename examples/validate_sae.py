"""Real-SAE validation: GPT-2 small vs a real GPT-2 fine-tune, through a public SAELens SAE.

    pip install -e ".[sae]"
    python examples/validate_sae.py OUT_DIR

Downloads gpt2 (~550 MB), lvwerra/gpt2-imdb (~550 MB, GPT-2 fine-tuned on IMDB reviews)
and the gpt2-small-res-jb layer-8 SAE (~150 MB). Checks: SAE reconstruction (NMSE) in the
bridge's raw-HF mode vs HookedTransformer-compatible mode, then a full SAE audit
base vs identical copy and base vs fine-tune.
"""
import json
import sys
import time

import torch

from model_ray.ci.gate import GateConfig, run_audit
from model_ray.core.orchestrator import ModelAuditor
from model_ray.core.sae_bridge import SAEFeatureProbe, feature_pool, load_sae
from model_ray.reports.generator import write_reports

OUT = sys.argv[1]
HOOK = "blocks.8.hook_resid_pre"
PROMPTS = [
    "The movie was a complete waste of time and the acting was",
    "I loved every minute of this film, especially the",
    "The weather today is cold and rainy, so I stayed",
    "She opened the door and saw a small dog sitting",
    "The stock market fell sharply after the announcement that",
    "In 1969, astronauts landed on the moon for the first",
    "The recipe calls for two cups of flour and a",
    "He was nervous before the job interview because he",
    "This documentary explains how bees communicate with each",
    "The plot twist at the end of the series made me",
    "Our team won the championship after a long and difficult",
    "The new phone has a larger screen and a faster",
    "The director's previous films were better than this one because",
    "Scientists discovered a new species of frog in the",
    "The train was delayed by two hours due to",
    "My favorite actor in this movie was the one who played",
] * 4  # 64 prompts; repeats are fine for a pipeline check, not for a real gate

sae = load_sae("gpt2-small-res-jb", HOOK)
print("SAE", type(sae).__name__, "d_in", sae.cfg.d_in, "d_sae", sae.cfg.d_sae, flush=True)


def nmse(a: ModelAuditor) -> float:
    out = a.cache(PROMPTS[:16], [HOOK], pool=feature_pool(sae))[HOOK]
    return float(out[:, -2].sum() / out[:, -1].sum())


t = time.time()
raw = ModelAuditor.from_pretrained("gpt2", batch_size=8)
print(f"gpt2 loaded {time.time() - t:.0f}s; raw-HF-mode NMSE: {nmse(raw):.4f}", flush=True)
del raw
PROC = {}  # TransformerLens defaults; the SAE was trained with center_writing_weights=True
base = ModelAuditor.from_pretrained("gpt2", batch_size=8, processing=PROC)
print(f"compatibility-mode NMSE: {nmse(base):.4f}", flush=True)

probe = SAEFeatureProbe(prompts=PROMPTS, hook=HOOK, sae=sae, top_k=10)
feats = probe.measure(base)["feats"]
probe.feature_ids = feats.mean(0).topk(5).indices.tolist()
print("pinned features (most active on base):", probe.feature_ids, flush=True)
cfg = GateConfig(sae=probe, n_boot=1000)

for tag, tuned in (("identical", "gpt2"), ("imdb_finetune", "lvwerra/gpt2-imdb")):
    t = time.time()
    rep = run_audit(cfg, lambda: base, lambda: ModelAuditor.from_pretrained(tuned, batch_size=8, processing=PROC))
    write_reports(rep, f"{OUT}/{tag}")
    (r,) = rep.results
    print(f"\n=== {tag}: {rep.verdict} ({time.time() - t:.0f}s)  same weights: "
          f"{rep.base.weights_sha256 == rep.tuned.weights_sha256}")
    print(f"  {r.status} feature_activation: base={r.base_value:.3f} tuned={r.tuned_value:.3f} "
          f"deg={r.degradation:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}] issues={r.issues}")
    print("  NMSE base/tuned:", round(r.details["base_nmse"], 4), round(r.details["tuned_nmse"], 4))
    print("  top shifted:", json.dumps(r.details["top_shifted_features"][:5]))
