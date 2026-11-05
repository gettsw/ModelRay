"""Model loading and batched, pooled activation caching on top of TransformerLens.

Targets TransformerLens >= 4 ``TransformerBridge`` (``HookedTransformer`` was removed
in 4.0). The bridge wraps the real HF model, so the audited forward pass is the one
you ship, and it already provides temporary hooks (``model.hooks(...)``) under
HookedTransformer-style names (``blocks.{L}.hook_resid_post``, ``ln_final``, ``W_U``).
``ModelAuditor`` adds only what an audit needs on top: padding-safe per-prompt
pooling, early stopping at the deepest requested layer, CPU offload of results,
and a weights digest for the report.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Literal, Sequence, Union

import torch
from torch import Tensor

# (acts [batch, pos, ...], mask [batch, pos] bool) -> [batch, ...]
PoolFn = Callable[[Tensor, Tensor], Tensor]
Pool = Union[Literal["last", "mean"], PoolFn]


def _last(acts: Tensor, mask: Tensor) -> Tensor:
    idx = mask.sum(1) - 1  # right-padded: last real token
    return acts[torch.arange(acts.shape[0], device=acts.device), idx]


def _mean(acts: Tensor, mask: Tensor) -> Tensor:
    # Position 0 (BOS / attention sink) has outlier norm and is excluded.
    m = mask.clone()
    m[:, 0] = False
    m = m.to(acts.dtype).view(*m.shape, *([1] * (acts.ndim - 2)))
    return (acts * m).sum(1) / m.sum(1).clamp(min=1)


POOLS: dict[str, PoolFn] = {"last": _last, "mean": _mean}


class ModelAuditor:
    """Wraps a ``TransformerBridge`` for audit-style batched activation extraction.

    Args:
        model: a TransformerLens ``TransformerBridge`` (put in eval mode).
        batch_size: prompts per forward pass.
        chat_template: format each prompt as a single user turn with the
            tokenizer's chat template (required for instruct models: refusal
            and sycophancy only show up in the format the model was tuned on).
        name: identifier recorded in reports.
        processing: the TransformerLens weight processing applied at load time
            (``None`` = raw HF numerics). Recorded so SAE probes can refuse a mismatch.
    """

    def __init__(
        self, model, batch_size: int = 8, chat_template: bool = False, name: str = "model",
        processing: dict[str, bool] | None = None,
    ):
        self.model = model.eval()
        self.batch_size = batch_size
        self.chat_template = chat_template
        self.name = name
        self.processing = processing

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        device: str | None = None,
        dtype: str = "float32",
        revision: str | None = None,
        processing: dict[str, bool] | None = None,
        **auditor_kwargs,
    ) -> "ModelAuditor":
        """Boot a HF hub id or local checkpoint dir.

        ``processing=None`` keeps raw HF numerics. Pass a dict of
        ``enable_compatibility_mode`` kwargs (``{}`` = HookedTransformer defaults: fold LN,
        center writing weights, ...) to match SAEs trained on processed activations. On
        GPT-2 small the ``gpt2-small-res-jb`` SAE has NMSE 4.6 on raw activations and 0.056
        on processed ones, so this choice decides whether SAE results mean anything."""
        from transformer_lens.model_bridge import TransformerBridge

        model = TransformerBridge.boot_transformers(
            checkpoint, device=device, dtype=getattr(torch, dtype), revision=revision
        )
        if processing is not None:
            model.enable_compatibility_mode(disable_warnings=True, **processing)
        auditor_kwargs.setdefault("name", checkpoint)
        return cls(model, processing=processing, **auditor_kwargs)

    # -- tokenization -----------------------------------------------------

    def format(self, prompt: str) -> str:
        if not self.chat_template:
            return prompt
        return self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def tokenize(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Right-padded tokens and a boolean mask of real (non-pad) positions.

        Lengths come from tokenizing each prompt alone, because pad == bos == eos
        for many tokenizers, so the mask cannot be recovered from token ids.
        """
        texts = [self.format(p) for p in prompts]
        bos = not self.chat_template  # chat templates add their own BOS where needed
        lengths = torch.tensor([self.model.to_tokens(t, prepend_bos=bos).shape[1] for t in texts])
        tokens = self.model.to_tokens(texts, prepend_bos=bos, padding_side="right")
        mask = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]
        return tokens, mask.to(tokens.device)

    # -- caching ----------------------------------------------------------

    @torch.no_grad()
    def cache(self, prompts: Sequence[str], hook_names: Sequence[str], pool: Pool = "last") -> dict[str, Tensor]:
        """Run ``prompts`` in batches and return ``{hook_name: [n_prompts, ...]}`` on CPU.

        Activations are pooled over positions per batch (``"last"`` real token,
        ``"mean"`` over real tokens excluding position 0, or a custom ``PoolFn``),
        so memory is O(prompts), not O(tokens). Hooks are attached only for the
        duration of each forward pass and the pass stops after the deepest
        requested block.
        """
        pool_fn = POOLS[pool] if isinstance(pool, str) else pool
        out: dict[str, list[Tensor]] = {h: [] for h in hook_names}
        stop = self._stop_layer(hook_names)
        for i in range(0, len(prompts), self.batch_size):
            tokens, mask = self.tokenize(prompts[i : i + self.batch_size])
            captured: dict[str, Tensor] = {}

            # Key by the requested name: hook.name is the canonical name, and
            # blocks.L.hook_resid_post is an alias of blocks.L.hook_out.
            def grab(name: str):
                def fn(act: Tensor, hook) -> None:
                    captured[name] = pool_fn(act, mask).float().cpu()

                return fn

            with self.model.hooks(fwd_hooks=[(h, grab(h)) for h in hook_names]):
                self.model(tokens, stop_at_layer=stop)
            for h in hook_names:
                out[h].append(captured[h])
        return {h: torch.cat(v) for h, v in out.items()}

    def _stop_layer(self, hook_names: Sequence[str]) -> int | None:
        layers = []
        for h in hook_names:
            m = re.match(r"blocks\.(\d+)\.", h)
            if not m:
                return None  # e.g. ln_final.*: needs the full forward
            layers.append(int(m.group(1)))
        return max(layers) + 1 if max(layers) + 1 < self.model.cfg.n_layers else None

    # -- helpers ----------------------------------------------------------

    def layer_index(self, layer: int | None) -> int:
        """Validated layer index (``None`` = middle, negatives count from the end).

        Configs are written for one model and run on others; an out-of-range layer must
        be a clear error, not a hook-name KeyError deep inside the bridge."""
        n = self.model.cfg.n_layers
        if layer is None:
            return n // 2
        if not -n <= layer < n:
            raise ValueError(f"layer {layer} out of range for {self.name} ({n} layers: valid 0..{n - 1} or -{n}..-1)")
        return layer % n

    def resid_hook(self, layer: int | None) -> str:
        """Residual-stream hook after ``layer`` (default: middle of the network)."""
        return f"blocks.{self.layer_index(layer)}.hook_resid_post"

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 48) -> list[str]:
        """Greedy completions, one prompt at a time (no padding to get wrong)."""
        outs = []
        for p in prompts:
            toks = self.model.to_tokens(self.format(p), prepend_bos=not self.chat_template)
            out = self.model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
            outs.append(self.model.tokenizer.decode(out[0, toks.shape[1] :], skip_special_tokens=True))
        return outs

    @torch.no_grad()
    def logit_lens(self, resid: Tensor, token_ids: Tensor | None = None) -> Tensor:
        """Project residual vectors ``[..., d_model]`` to logits via final norm + unembed.

        Exact final logits when ``resid`` is the last block's ``hook_resid_post``.
        """
        m = self.model
        x = m.ln_final(resid.to(m.cfg.device, m.cfg.dtype).unsqueeze(-2)).squeeze(-2)
        W_U, b_U = m.W_U, m.b_U
        if token_ids is not None:
            W_U, b_U = W_U[:, token_ids], b_U[token_ids]
        return (x @ W_U + b_U).float().cpu()

    def token_id(self, text: str) -> int:
        """First token of ``text`` (no BOS)."""
        return int(self.model.to_tokens(text, prepend_bos=False)[0, 0])

    def weights_sha256(self) -> str:
        """Digest of the audited weights, to bind a report to an exact artifact."""
        h = hashlib.sha256()
        hf = getattr(self.model, "original_model", self.model)
        for k, v in sorted(hf.named_parameters()):  # buffers (causal masks) skipped
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def tokenizer_sha256(self) -> str:
        """Digest of the vocab/merges, special tokens and chat template actually used.

        Together with ``weights_sha256`` this still does NOT cover system prompts set at
        serving time, quantization, or runtime hooks (an ablation hook leaves both
        digests unchanged)."""
        tok = self.model.tokenizer
        backend = getattr(tok, "backend_tokenizer", None)
        parts = [
            backend.to_str() if backend is not None else repr(sorted(tok.get_vocab().items())),
            repr(sorted(tok.special_tokens_map.items())),
            str(getattr(tok, "chat_template", None) if self.chat_template else None),
        ]
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()
