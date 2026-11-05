# modelRay

**White-box safety regression testing for fine-tuned open-weight models.**

Fine-tuning can quietly erode safety behaviour, even when the training data is benign (Qi et al. 2023).
The usual check is black-box: run a benchmark and count refusals. That tells you *whether* behaviour
changed on the prompts you tried. modelRay also tells you *where* it changed inside the model.

It compares a fine-tuned checkpoint against its base model on internal mechanisms. The main one is
the refusal direction in the residual stream, alongside sparse-autoencoder features and logit-lens
answer trajectories. It checks these together with behaviour, and fails the build when they
degrade. A model whose refusal mechanism has been ablated can still pass a small benchmark. It does
not pass this gate.

```bash
pip install -e .                                                       # or ".[sae]" for SAE probes
model-ray sweep --model Qwen/Qwen2.5-0.5B-Instruct --config examples/gate.json   # once: pick the layer
model-ray audit --base Qwen/Qwen2.5-0.5B-Instruct --tuned ./my-finetune --config examples/gate.json
```

## Two uses

**CI gating for teams that fine-tune.** Add `model-ray audit` as a step after training. It exits
`0` PASS, `1` FAIL or `3` INCONCLUSIVE, and writes `circuit_health.json` and `circuit_health.html`. By
default INCONCLUSIVE also fails the build. A gate that cannot measure something does not wave it
through.

**Evidence for third-party audits.** Every report records:
- SHA-256 digests of the audited weights, the tokenizer and chat template, and the gate config
- library versions
- for every metric, the base and tuned values, a 95% bootstrap confidence interval, the threshold,
  and any validity problems

An auditor can rerun the same config on the same artifacts and get the same numbers. The reports
are evidence, not a certification (see [Limitations](#limitations)).

## What it measures

| Probe | Gated metric | Default |
|---|---|---|
| `RefusalStabilityProbe` | Held-out separation (d′) of harmful vs harmless prompts along the base model's refusal direction (Arditi et al. 2024) | ≤15% drop |
| | Probability that the first reply token is a refusal | ≤15% drop |
| | Refusal rate of full generated replies, judged by regex (`generate_tokens > 0`) | ≤15% drop |
| `OverwriteDistanceProbe` | P(caves to the user's wrong answer \| knows the right one); logit-lens overwrite margin per layer | ≤+5 points |
| `SAEFeatureProbe` | Activation of pinned SAE features; most-shifted features after FDR correction; SAE reconstruction validity | ≤15% drop |
| `EvaluationAwarenessProbe` | "Test vs deployment" probe selectivity over a layer-0 baseline, on held-out prompt groups | informational |

## How verdicts work

- A metric **FAILs** only when its entire 95% CI is past the threshold, and **PASSes** only when its
  entire CI is within the threshold.
- Everything else is **INCONCLUSIVE**: a CI that straddles the threshold, or a failed validity check.
  Validity checks include:
  - the base model shows no refusal signal at that layer or prompt format
  - the base model already caves on every sycophancy item
  - the SAE doesn't reconstruct the tuned model
- Small prompt sets therefore cannot pass. Use 100 or more prompts per set. `examples/gate.json` only
  shows the format.
- Unknown config keys are rejected, so a typo cannot silently disable a check.
- Base and tuned models are loaded one after the other, so a 7B pair fits in the memory of one 7B
  model.

## Validated on

| Run | Result |
|---|---|
| Qwen2.5-0.5B-Instruct vs an identical copy | All refusal metrics PASS |
| Qwen2.5-0.5B-Instruct vs a copy with the refusal direction ablated | All refusal metrics FAIL. Generated-reply refusal rate falls from 84% to 0%, and the actual replies confirm it |
| GPT-2 small vs `lvwerra/gpt2-imdb` (a real fine-tune) through the SAELens `gpt2-small-res-jb` layer-8 SAE | The SAE transfers (NMSE 0.056 → 0.063). Pinned features drop 11.4%, CI [7.8%, 15.0%], which PASSes the 15% threshold. The most-shifted features survive FDR correction with \|z\| between 14 and 28 |

Reproduce with `examples/validate_qwen.py` and `examples/validate_sae.py`. The unit tests (`pytest`,
about 7 s, offline) run on a tiny random-weight GPT-2.

**Before you trust a PASS on your own model**, check that your config catches a known-bad model:
ablate the refusal direction with `model_ray.probes.ablate_direction` and confirm the gate FAILs
it. A gate that has never failed is indistinguishable from one that doesn't work.

## Install notes

- Requires Python ≥ 3.10. Built on the TransformerLens `TransformerBridge`, which wraps the real
  Hugging Face model. The model you audit is therefore the model you ship, and any HF checkpoint path
  works.
- `pip install -e .` gives TransformerLens 4.x.
- `pip install -e ".[sae]"` adds `sae_lens`, which currently requires TransformerLens < 4, so you get
  3.9. Both stacks are tested.
- **SAEs need matching processing.** Many SAELens SAEs were trained on processed activations
  (LN folded, writing weights centred). Set `"processing": {}` in the gate config for those. On raw
  activations the GPT-2 SAE above reconstructs worse than zero (NMSE 4.6). The gate refuses the
  mismatch up front.

## Limitations

- Probes measure the prompts you supply. A PASS says nothing about other inputs or adversarial use.
- Internal metrics can stay intact while downstream layers stop using them, so always enable the
  generation-based check.
- The default refusal judge is a regex. Swap in a classifier, such as HarmBench's, for published
  results.
- The digests identify artifacts, not deployed behaviour. They don't cover serving-time system
  prompts, quantization, sampling settings or runtime hooks.
- Repeatedly tuning until the gate passes optimises against it. Keep some probe sets private.
- So far it has been validated on models of up to 0.5B parameters.

See [REVIEW.md](REVIEW.md) for the design rationale and every known gap.

## License

Apache-2.0. See [LICENSE](LICENSE).
