"""LoRA fine-tune one grid cell from experiments/DESIGN.md, merge, save to experiments/runs/<tag>/model.

    python experiments/train.py --model Qwen/Qwen2.5-0.5B-Instruct --condition instruction --steps 200 --seed 0
    python experiments/train.py ... --condition mixed --harmful-frac 0.05 --steps 200
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/hf_cache")
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = Path(__file__).parent
SOURCES = {"benign_domain": "train_benign_domain", "instruction": "train_instruction",
           "identity_shift": "train_identity_shift", "harmful": "train_harmful", "refusal": "train_refusal", "alpaca": "train_alpaca"}


def load(name):
    with open(HERE / "data" / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(tok, ex, max_len):
    """Chat-template the pair; loss only on the response tokens."""
    msgs = [{"role": "user", "content": ex["prompt"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": ex["response"]}], tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels


def batches(tok, data, steps, bs, max_len, rng):
    order = []
    while len(order) < steps * bs:  # small sets (identity_shift) are repeated
        order += rng.sample(range(len(data)), len(data))
    for s in range(steps):
        enc = [encode(tok, data[i], max_len) for i in order[s * bs:(s + 1) * bs]]
        n = max(len(i) for i, _ in enc)
        ids = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i, _ in enc])
        lab = torch.tensor([lab + [-100] * (n - len(lab)) for _, lab in enc])
        yield ids, lab, (ids != tok.pad_token_id) | (lab != -100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--condition", required=True, choices=[*SOURCES, "mixed"])
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--harmful-frac", type=float, default=0.0, help="mixed only: share of harmful examples")
    ap.add_argument("--refusal-frac", type=float, default=0.0, help="mixed only: share of refusal examples")
    ap.add_argument("--mix-base", default="instruction", help="mixed only: the clean data source")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    if a.condition == "mixed":
        clean, harm, ref = load(SOURCES[a.mix_base]), load(SOURCES["harmful"]), load(SOURCES["refusal"])
        kh, kr = round(a.harmful_frac * len(clean)), round(a.refusal_frac * len(clean))
        data = clean[: len(clean) - kh - kr] + harm[:kh] + ref[len(ref) - kr:]  # disjoint prompt slices
        tag_cond = (f"mixed{a.harmful_frac:g}" if not a.refusal_frac else
                    f"{a.mix_base}+refusal{a.refusal_frac:g}" + (f"+harmful{a.harmful_frac:g}" if a.harmful_frac else ""))
    else:
        data, tag_cond = load(SOURCES[a.condition]), a.condition
    short = a.model.split("/")[-1]
    lr = "" if a.lr == 1e-4 else f"-lr{a.lr:g}"
    tag = f"{short}__{tag_cond}{lr}__s{a.steps}__seed{a.seed}{'__full' if a.full else ''}"
    out = HERE / "runs" / tag
    if (out / "model" / "config.json").exists():
        print("exists:", out)
        return

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    if not a.full:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.steps // 10)))
    model.train()
    t, losses = time.time(), []
    for step, (ids, lab, mask) in enumerate(batches(tok, data, a.steps, a.batch_size, a.max_len, rng)):
        loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=lab.cuda()).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        losses.append(loss.item())
        if step % max(1, a.steps // 10) == 0 or step == a.steps - 1:
            print(f"step {step:>4}/{a.steps} loss {loss.item():.3f}  {time.time() - t:.0f}s", flush=True)

    model = model.merge_and_unload() if not a.full else model
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "train.json").write_text(json.dumps({**vars(a), "tag": tag, "n_data": len(data), "losses": losses,
                                                "seconds": time.time() - t}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
