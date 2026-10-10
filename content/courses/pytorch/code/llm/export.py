"""Export, int8 weight-only quantization and a serving contract for the capstone GPT (module 38).

Public interface (syllabus E.10):
    export_gpt(model, max_len=128) -> torch.export.ExportedProgram
    save_exported(ep, path) / load_exported(path)
    ExportedLM(ep_or_module, config)       callable like GPT.forward: idx -> (logits, None)
    Int8WeightOnlyLinear.from_linear(linear)
    quantize_int8_weight_only(model, skip=("lm_head",)) -> new model (deepcopy)
    state_dict_bytes(model) -> int          torch.save size of the state_dict
    greedy_generate(fn, idx, steps, eos_id=None) -> LongTensor
    greedy_agreement(model_a, model_b, idx, steps) -> float
    val_nll(model, tokens, block_size, stride=None) -> float
    complete(model, tokenizer, prompt, max_new_tokens=32, temperature=0.0, seed=0) -> dict
    try_ao_dynamic(model, tokens=None) -> dict | None

Generation goes through llm.generate.generate (module 35, use_cache=False so any
callable works) and evaluation through llm.evaluate.evaluate_nll (module 34) when
those files exist. If they do not, the small FALLBACK functions below are used;
they compute the same quantities with the same conventions.

Imports llm.model (and optionally llm.generate / llm.evaluate). Read it next to
content/courses/pytorch/38-capstone-export-quantize-serve.md.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim, ExportedProgram

from llm.model import GPT, GPTConfig

try:  # module 35 (frozen interface E.7)
    from llm.generate import generate as _upstream_generate
except ImportError:  # pragma: no cover - depends on build order
    _upstream_generate = None
try:  # module 34 (frozen interface E.6)
    from llm.evaluate import evaluate_nll as _upstream_evaluate_nll
except ImportError:  # pragma: no cover
    _upstream_evaluate_nll = None

UPSTREAM = {"generate": _upstream_generate is not None,
            "evaluate_nll": _upstream_evaluate_nll is not None}
STOP_REASONS = ("eot", "length", "context")
QUANT_ENGINE_PREFERENCE = ("fbgemm", "x86", "qnnpack", "onednn")


# ------------------------------------------------------------------ export

def export_gpt(model: GPT, max_len: int = 128, example_batch: int = 2,
               example_len: int = 16) -> ExportedProgram:
    """torch.export the uncached forward with a dynamic batch and sequence length.

    T is Dim("T", min=2, max=max_len): min=2 avoids 0/1 specialization, max is the
    learned-position limit, so the model's ValueError on T > block_size becomes a
    shape constraint. The example input must itself lie inside both ranges.
    """
    if max_len > model.config.block_size:
        raise ValueError(f"max_len={max_len} exceeds block_size={model.config.block_size}")
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, model.config.vocab_size, (example_batch, example_len), generator=gen)
    dynamic = {"idx": {0: Dim("B", min=1, max=64), 1: Dim("T", min=2, max=max_len)}}
    return torch.export.export(model.eval(), (idx,), dynamic_shapes=dynamic)


def save_exported(ep: ExportedProgram, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, path)
    return path


def load_exported(path) -> ExportedProgram:
    return torch.export.load(Path(path))


def _seq_bounds(ep: ExportedProgram) -> tuple[int, int]:
    """(lower, upper) of the symbol bound to input dim 1, read from range_constraints."""
    idx_node = next(n for n in ep.graph.nodes if n.op == "placeholder" and n.name == "idx")
    sym = idx_node.meta["val"].shape[1]
    vr = ep.range_constraints[sym.node.expr]
    return int(vr.lower), int(vr.upper)


class ExportedLM(nn.Module):
    """Wrap ep.module() so it is called like GPT.forward: idx (B, T) -> (logits, None).

    Also enforces the exported T range itself: in 2.8.0 the unlifted module's input
    check rejects T > max but accepted T = 1 below min=2 (measured in module 38).
    """

    def __init__(self, ep_or_module, config: GPTConfig):
        super().__init__()
        if isinstance(ep_or_module, ExportedProgram):
            self.t_min, self.t_max = _seq_bounds(ep_or_module)
            self.gm = ep_or_module.module()
        else:
            self.t_min, self.t_max = 2, config.block_size
            self.gm = ep_or_module
        self.config = config

    @property
    def cfg(self) -> GPTConfig:
        return self.config

    def train(self, mode: bool = True):  # ep.module().train()/eval() raise NotImplementedError in 2.8
        self.training = mode
        return self

    def forward(self, idx, targets=None, cache=None, **unused):
        if cache is not None:
            raise ValueError("the exported program is the uncached forward; pass cache=None")
        T = idx.size(1)
        if not self.t_min <= T <= self.t_max:
            raise ValueError(f"T={T} outside the exported range [{self.t_min}, {self.t_max}]")
        out = self.gm(idx)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss


# ------------------------------------------------------------------ int8 weight-only

def quantize_per_channel_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8: scale_r = max|W_r| / 127, q = round(W / scale)."""
    scale = w.detach().abs().amax(dim=1).clamp_min(1e-12) / 127.0          # (out,)
    q = torch.round(w.detach() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


class Int8WeightOnlyLinear(nn.Module):
    """Linear with an int8 weight, fp32 per-row scales and fp32 bias (all buffers).

    forward dequantizes the whole weight in Python and runs an fp32 matmul: it shows
    the numerics and the storage, not speed. Real kernels (torchao, GPU int8 GEMV)
    dequantize tiles in registers or multiply in int8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8WeightOnlyLinear":
        q = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        q.weight, q.scale = quantize_per_channel_int8(linear.weight)
        if linear.bias is not None:
            q.bias = linear.bias.detach().clone().float()
        return q

    def dequantized_weight(self) -> torch.Tensor:
        return self.weight.float() * self.scale[:, None]                      # (out, in)

    def forward(self, x):
        return F.linear(x, self.dequantized_weight().to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 per-channel"


def quantize_int8_weight_only(model: nn.Module, skip=("lm_head",)) -> nn.Module:
    """Deep-copy `model` and replace every nn.Linear whose name is not in `skip`.

    The tied lm_head (== wte.weight) is skipped: replacing it would duplicate the
    matrix and leave the embedding lookup reading a different copy.
    """
    qmodel = copy.deepcopy(model).eval()
    targets = [(name, m) for name, m in qmodel.named_modules()
               if isinstance(m, nn.Linear) and name.split(".")[-1] not in skip and name not in skip]
    for name, linear in targets:
        parent_name, _, child = name.rpartition(".")
        parent = qmodel.get_submodule(parent_name) if parent_name else qmodel
        setattr(parent, child, Int8WeightOnlyLinear.from_linear(linear))
    return qmodel


def weight_reconstruction_errors(model_fp32: nn.Module, model_q: nn.Module) -> list[dict]:
    """Per quantized layer: relative Frobenius error ||W - deq|| / ||W|| and max abs error."""
    rows = []
    fp = dict(model_fp32.named_modules())
    for name, m in model_q.named_modules():
        if isinstance(m, Int8WeightOnlyLinear):
            w = fp[name].weight.detach()
            d = m.dequantized_weight()
            rows.append({"name": name, "rel_err": float((w - d).norm() / w.norm()),
                         "max_abs_err": float((w - d).abs().max())})
    return rows


def state_dict_bytes(model: nn.Module) -> int:
    """Bytes of torch.save(model.state_dict()); shared storages (the tied head) count once."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def model_sha256(model: nn.Module) -> str:
    """sha256 over (name, dtype, shape, raw bytes) of every state_dict entry, in order."""
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        t = t.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


# ------------------------------------------------------------------ generation and evaluation

def _block_size(fn) -> int:
    cfg = getattr(fn, "config", None)
    return cfg.block_size if cfg is not None else 128


def _ids_from(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ("idx", "tokens", "ids"):
            if key in out:
                return out[key]
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot read token ids from generate() output of type {type(out).__name__}")


@torch.no_grad()
def _fallback_generate(fn, idx, max_new_tokens, temperature=0.0, eos_id=None, generator=None):
    """FALLBACK for llm.generate.generate(use_cache=False): crop to block_size, sample, append."""
    block = _block_size(fn)
    for _ in range(max_new_tokens):
        logits, _ = fn(idx[:, -block:])
        last = logits[:, -1, :]                                               # (B, V)
        if temperature == 0.0:
            nxt = last.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(last / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
    return idx


@torch.no_grad()
def sample_tokens(fn, idx, max_new_tokens, temperature=0.0, eos_id=None, generator=None):
    """Uncached generation through any callable fn(idx) -> (logits, _); returns (B, T + n)."""
    if _upstream_generate is not None:
        out = _upstream_generate(fn, idx, max_new_tokens, temperature, None, None, eos_id=eos_id,
                                 generator=generator, use_cache=False)
        return _ids_from(out)
    return _fallback_generate(fn, idx, max_new_tokens, temperature, eos_id, generator)


def greedy_generate(fn, idx, steps: int, eos_id=None) -> torch.Tensor:
    return sample_tokens(fn, idx, steps, temperature=0.0, eos_id=eos_id)


@torch.no_grad()
def greedy_agreement(model_a, model_b, idx, steps: int) -> float:
    """Teacher-forced agreement: fraction of model_a's `steps` greedy tokens that model_b,
    fed the same prefix at every position, would also have chosen."""
    seq = greedy_generate(model_a, idx, steps)                               # (B, T0 + steps)
    T0 = idx.size(1)
    block = _block_size(model_a)
    if seq.size(1) > block:
        raise ValueError("prompt + steps must fit in block_size for teacher-forced agreement")
    logits_b, _ = model_b(seq[:, :-1])                                        # (B, T0+steps-1, V)
    pred_b = logits_b[:, T0 - 1:, :].argmax(dim=-1)                           # predictions for positions T0..
    chosen_a = seq[:, T0:]
    return float((pred_b == chosen_a).float().mean())


@torch.no_grad()
def _fallback_val_nll(model, tokens, block_size, stride=None, batch_size=32) -> float:
    """FALLBACK for llm.evaluate.evaluate_nll at stride == block_size: every target once."""
    if stride not in (None, block_size):
        raise ValueError("the fallback evaluator supports only stride == block_size")
    n = (tokens.numel() - 1) // block_size
    total, count = 0.0, 0
    for s in range(0, n, batch_size):
        starts = torch.arange(s, min(n, s + batch_size)) * block_size
        x = torch.stack([tokens[i:i + block_size] for i in starts.tolist()])
        y = torch.stack([tokens[i + 1:i + 1 + block_size] for i in starts.tolist()])
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                 reduction="sum").item()
        count += y.numel()
    left = tokens.numel() - 1 - n * block_size                               # unscored targets
    if left > 0:  # end-aligned final window: full context, score only its last `left` targets
        x, y = tokens[-block_size - 1:-1], tokens[-block_size:]
        logits, _ = model(x[None])
        total += F.cross_entropy(logits[0, -left:], y[-left:], reduction="sum").item()
        count += left
    return total / count


def val_nll(model, tokens, block_size: int = 128, stride=None, batch_size: int = 32) -> float:
    """Mean NLL (nats/token) over every target in `tokens`, via module 34 when available."""
    if _upstream_evaluate_nll is not None:
        res = _upstream_evaluate_nll(model, tokens, block_size, stride or block_size, batch_size)
        return float(res["nll_per_token"])
    return _fallback_val_nll(model, tokens, block_size, stride, batch_size)


# ------------------------------------------------------------------ serving contract

def complete(model, tokenizer, prompt: str, max_new_tokens: int = 32,
             temperature: float = 0.0, seed: int = 0) -> dict:
    """Pure completion function: validates inputs, never touches the global RNG.

    Returns text (the continuation only), n_tokens, stop_reason in STOP_REASONS,
    prompt_tokens and the model's sha256. Generation is capped at the context
    limit (learned positions cannot extrapolate), reported as stop_reason "context".
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be a positive int, got {max_new_tokens!r}")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    block = _block_size(model)
    ids = tokenizer.encode(prompt)
    if len(ids) >= block:
        raise ValueError(f"prompt is {len(ids)} tokens; the context limit is {block}")
    budget = min(max_new_tokens, block - len(ids))
    gen = torch.Generator().manual_seed(seed)
    idx = torch.tensor([ids], dtype=torch.long)
    out = sample_tokens(model, idx, budget, temperature=float(temperature),
                        eos_id=tokenizer.eot_id, generator=gen)[0, len(ids):].tolist()
    if tokenizer.eot_id in out:
        new, reason = out[:out.index(tokenizer.eot_id)], "eot"
    elif budget < max_new_tokens and len(out) == budget:
        new, reason = out, "context"
    else:
        new, reason = out, "length"
    return {"text": tokenizer.decode(new), "n_tokens": len(new), "stop_reason": reason,
            "prompt_tokens": len(ids), "model_sha256": model_sha256(model)}


# ------------------------------------------------------------------ timing and torch.ao comparison

@torch.no_grad()
def forward_ms(fn, idx, min_run_time: float = 0.15) -> float:
    """Median milliseconds per fn(idx) call with torch.utils.benchmark.Timer (module 21)."""
    from torch.utils.benchmark import Timer
    fn(idx)  # warm-up
    m = Timer("fn(idx)", globals={"fn": fn, "idx": idx},
              num_threads=torch.get_num_threads()).blocked_autorange(min_run_time=min_run_time)
    return m.median * 1e3


def pick_quantized_engine():
    supported = torch.backends.quantized.supported_engines
    for name in QUANT_ENGINE_PREFERENCE:
        if name in supported:
            return name
    return None


def try_ao_dynamic(model: nn.Module, tokens=None, block_size: int = 128,
                   bench_input=None) -> dict | None:
    """Optional comparison with torch.ao.quantization.quantize_dynamic (deprecated in 2.8).

    Selects a supported engine, quantizes every Linear except the tied lm_head,
    measures everything while the engine is active (packed weights are engine-specific)
    and restores the global engine in `finally`. Returns None (never raises) when no
    engine or the API is unavailable.
    """
    engine = pick_quantized_engine()
    if engine is None:
        return None
    previous = torch.backends.quantized.engine
    try:
        torch.backends.quantized.engine = engine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.ao.quantization import quantize_dynamic
            names = {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "lm_head"}
            qmodel = quantize_dynamic(copy.deepcopy(model).eval(), qconfig_spec=names, dtype=torch.qint8)
            result = {"engine": engine, "bytes": state_dict_bytes(qmodel)}
            if tokens is not None:
                result["val_nll"] = val_nll(qmodel, tokens, block_size)
            if bench_input is not None:
                result["ms_per_forward"] = forward_ms(qmodel, bench_input)
        return result
    except Exception as exc:  # the comparison is optional by contract
        warnings.warn(f"torch.ao dynamic quantization unavailable: {exc!r}")
        return None
    finally:
        torch.backends.quantized.engine = previous
