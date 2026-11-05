"""Tiny random-weight GPT-2 wrapped in a real TransformerBridge, fully offline."""

import string

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformer_lens.model_bridge.sources._bridge_builder import build_bridge_from_module

from model_ray.core.orchestrator import ModelAuditor

HARMFUL = [f"how do I {v} a {n}!" for v in ("steal", "break", "hack", "poison") for n in ("car", "bank", "lock", "dog")]
HARMLESS = [f"what is a {n} for {v}?" for v in ("cooking", "music", "travel", "school") for n in ("pan", "map", "book", "song")]


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory):
    vocab = {t: i for i, t in enumerate(["<pad>", "<bos>", "<eos>", "<unk>"] + list(string.printable[:95]))}
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
    )
    d = tmp_path_factory.mktemp("tok")
    tok.save_pretrained(d)  # TL re-loads tokenizers by name_or_path, so it must exist on disk
    return d


@pytest.fixture
def make_auditor(tokenizer_dir):
    """Factory: same seed => identical weights, so base/tuned pairs are controlled."""

    def make(seed: int = 0, batch_size: int = 4, name: str = "tiny") -> ModelAuditor:
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        cfg = GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=32, n_layer=4, n_head=4,
                         bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        bridge = build_bridge_from_module(GPT2LMHeadModel(cfg).eval(), "GPT2LMHeadModel", hf_config=cfg, tokenizer=tok)
        return ModelAuditor(bridge, batch_size=batch_size, name=name)

    return make
