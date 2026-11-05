"""Measure one checkpoint: behaviour (generations saved for re-judging) + ModelRay internals.

    python experiments/evaluate.py --base Qwen/Qwen2.5-0.5B-Instruct                 # base itself + layer sweep
    python experiments/evaluate.py --base Qwen/Qwen2.5-0.5B-Instruct --run <tag>     # a fine-tuned run

Writes experiments/runs/<tag>/{generations.jsonl, internal.pt, eval.json}. The base run is
"<model>__base" and also stores the swept layer that every run of that base reuses.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/hf_cache")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from model_ray.core.orchestrator import ModelAuditor  # noqa: E402
from model_ray.probes.refusal import RefusalStabilityProbe, is_refusal  # noqa: E402

HERE = Path(__file__).parent
PREFILL = "Sure, here is"
REFUSAL_TOKENS = ["I", "Sorry", "As"]


def lines(name):
    with open(HERE / "data" / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


@torch.no_grad()
def generate(path, prompts, prefill="", max_new=128, bs=48):
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             + prefill for p in prompts]
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    del model
    gc.collect(), torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", default=None, help="run tag under experiments/runs; omit to evaluate the base")
    ap.add_argument("--dtype", default="float32", help="dtype for the ModelRay pass")
    a = ap.parse_args()

    base_dir = HERE / "runs" / f"{a.base.split('/')[-1]}__base"
    out = HERE / "runs" / a.run if a.run else base_dir
    path = str(out / "model") if a.run else a.base
    out.mkdir(parents=True, exist_ok=True)
    if (out / "eval.json").exists():
        print("exists:", out)
        return
    t = time.time()

    # --- behaviour --------------------------------------------------------------------------
    sets = {"harmbench": lines("eval_harmbench"), "xstest_safe": lines("eval_xstest_safe")}
    rows = []
    for name, prompts in [*sets.items(), ("harmbench_prefill", sets["harmbench"])]:
        prefill = PREFILL if name.endswith("prefill") else ""
        for p, g in zip(prompts, generate(path, prompts, prefill)):
            rows.append({"set": name, "prompt": p, "prefill": prefill, "generation": g, "regex_refusal": is_refusal(g)})
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    behaviour = {s: sum(r["regex_refusal"] for r in rows if r["set"] == s) / sum(r["set"] == s for r in rows)
                 for s in ("harmbench", "xstest_safe", "harmbench_prefill")}
    print("refusal rates:", {k: round(v, 3) for k, v in behaviour.items()}, f"{time.time() - t:.0f}s", flush=True)

    # --- ModelRay internals ----------------------------------------------------------------
    probe = RefusalStabilityProbe(harmful=lines("probe_harmful"), harmless=lines("probe_harmless"),
                                  refusal_tokens=REFUSAL_TOKENS)
    aud = ModelAuditor.from_pretrained(path, device="cuda", dtype=a.dtype, chat_template=True, batch_size=16)
    extra = {}
    if not a.run:
        sweep = probe.sweep(aud)
        extra = {"sweep": sweep, "layer": max(sweep, key=sweep.get)}
        print("sweep peak layer", extra["layer"], f"d'={sweep[extra['layer']]:.2f}", flush=True)
    layer = extra.get("layer") or json.loads((base_dir / "eval.json").read_text())["layer"]
    probe.layer = layer
    m = probe.measure(aud)
    torch.save(m, out / "internal.pt")
    del aud
    gc.collect(), torch.cuda.empty_cache()

    result = {"path": path, "behaviour_refusal_rate": behaviour, "layer": layer, **extra}
    if a.run:
        base_m = torch.load(base_dir / "internal.pt")
        result["internal"] = [r.model_dump() for r in probe.compare(base_m, m, n_boot=1000)]
        for r in result["internal"]:
            print(f"  {r['status']:<12} {r['metric']}: {r['degradation']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    result["seconds"] = time.time() - t
    (out / "eval.json").write_text(json.dumps(result, indent=1))
    print("done", out, f"{result['seconds']:.0f}s")


if __name__ == "__main__":
    main()
