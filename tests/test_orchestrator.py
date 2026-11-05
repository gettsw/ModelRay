import pytest
import torch

PROMPTS = ["a", "hello there", "abc", "a much longer prompt than the rest", "xy"]


def test_batched_pooling_matches_single_prompt_runs(make_auditor):
    a = make_auditor(batch_size=3)  # forces padded, mixed-length batches
    hook = "blocks.1.hook_resid_post"  # also exercises stop_at_layer=2
    last = a.cache(PROMPTS, [hook], pool="last")[hook]
    mean = a.cache(PROMPTS, [hook], pool="mean")[hook]
    for i, p in enumerate(PROMPTS):
        _, c = a.model.run_with_cache(a.model.to_tokens(p))
        ref = c[hook][0]
        torch.testing.assert_close(last[i], ref[-1], atol=1e-5, rtol=1e-4)
        exp_mean = ref[1:].mean(0) if len(ref) > 1 else torch.zeros_like(ref[0])  # BOS excluded
        torch.testing.assert_close(mean[i], exp_mean, atol=1e-5, rtol=1e-4)


def test_hooks_are_temporary(make_auditor):
    a = make_auditor()
    t = a.model.to_tokens("hello")
    before = a.model(t)
    a.cache(PROMPTS, ["blocks.0.hook_resid_post", "blocks.3.hook_resid_post"])
    assert sum(len(h._forward_hooks) for h in a.model.hook_dict.values()) == 0
    torch.testing.assert_close(a.model(t), before)


def test_logit_lens_on_last_layer_is_exact(make_auditor):
    a = make_auditor()
    hook = f"blocks.{a.model.cfg.n_layers - 1}.hook_resid_post"
    resid = a.cache(["hello there"], [hook])[hook]
    logits = a.model(a.model.to_tokens("hello there"))[0, -1]
    torch.testing.assert_close(a.logit_lens(resid)[0], logits, atol=1e-4, rtol=1e-4)


def test_weights_digest_tracks_weights(make_auditor):
    a, b, c = make_auditor(0), make_auditor(0), make_auditor(1)
    assert a.weights_sha256() == b.weights_sha256() != c.weights_sha256()


def test_layer_index_validates_range(make_auditor):
    a = make_auditor()  # 4 layers
    assert a.layer_index(None) == 2 and a.layer_index(-1) == 3 and a.resid_hook(3) == "blocks.3.hook_resid_post"
    for bad in (4, 16, -5):
        with pytest.raises(ValueError, match="out of range"):
            a.resid_hook(bad)


def test_tokenizer_digest_tracks_chat_template_flag(make_auditor):
    a, b = make_auditor(), make_auditor()
    b.model.tokenizer.chat_template = "{{ messages[0]['content'] }}"  # fixture has none
    b.chat_template = True
    assert a.tokenizer_sha256() == make_auditor().tokenizer_sha256() != b.tokenizer_sha256()


def test_generate_returns_text(make_auditor):
    out = make_auditor().generate(["hello", "a longer prompt"], max_new_tokens=3)
    assert len(out) == 2 and all(isinstance(t, str) for t in out)
