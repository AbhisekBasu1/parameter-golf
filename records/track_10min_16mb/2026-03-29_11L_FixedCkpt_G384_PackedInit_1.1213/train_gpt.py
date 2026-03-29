from __future__ import annotations
import copy
from collections import Counter
import glob
import io
import math
import os
import random
import struct
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False
except Exception:
    pass
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except Exception:
    flash_attn_3_func = None
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", "pr414_fixed_ckpt_b10_k1024_v15424_g384_record")
    seed = int(os.environ.get("SEED", 42))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 0))
    iterations = int(os.environ.get("ITERATIONS", 0))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 0))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))  # tighter: collect more recent checkpoints
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))  # XSA on last 4 layers (0 = disabled)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    mlp_activation = os.environ.get("MLP_ACTIVATION", "relu2").strip().lower()
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    gradquant_enabled = bool(int(os.environ.get("GRADQUANT_ENABLED", "0")))
    gradquant_start_scale = float(os.environ.get("GRADQUANT_START_SCALE", 0.10))
    gradquant_ema = float(os.environ.get("GRADQUANT_EMA", 0.90))
    gradquant_top_frac = float(os.environ.get("GRADQUANT_TOP_FRAC", 0.10))
    gradquant_bottom_frac = float(os.environ.get("GRADQUANT_BOTTOM_FRAC", 0.20))
    gradquant_min_numel = int(os.environ.get("GRADQUANT_MIN_NUMEL", 65536))
    gradquant_categories = os.environ.get("GRADQUANT_CATEGORIES", "mlp,attn")
    export_bits_profile = os.environ.get("EXPORT_BITS_PROFILE", "block10_mlp_qproj7")
    export_coord_sidecar_packing = os.environ.get("EXPORT_COORD_SIDECAR_PACKING", "flat").strip().lower()
    export_coord_sidecar_path = os.environ.get("EXPORT_COORD_SIDECAR_PATH", "").strip()
    export_prepacked_sbev_path = os.environ.get("EXPORT_PREPACKED_SBEV_PATH", "./prepacked_sbev.bin").strip()
    export_residualq_sidecar_path = os.environ.get("EXPORT_RESIDUALQ_SIDECAR_PATH", "").strip()
    init_model_path = os.environ.get("INIT_MODEL_PATH", "./init_model.ptz").strip()
    init_model_strict = bool(int(os.environ.get("INIT_MODEL_STRICT", "1")))
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._built = False
    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    "p": p,
                    "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "scale": max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True
    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            pg[:m["B"]].copy_(p.grad.bfloat16())
            if pg.shape[0] > m["B"]:
                pg[m["B"]:].zero_()
            fut = dist.reduce_scatter_tensor(m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for i, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    upd = prev_m["full_update"][:prev_m["B"]]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m["shard"]
                    buf = m["shard_mom"]
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(m["full_update"], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][:prev_m["B"]]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
        if hasattr(self, "_rs_futures"):
            del self._rs_futures
        return loss
def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] — broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
        if k.dtype not in (torch.float16, torch.bfloat16):
            k = k.to(torch.bfloat16)
        if v.dtype not in (torch.float16, torch.bfloat16):
            v = v.to(torch.bfloat16)
        if flash_attn_3_func is not None:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=self.num_heads != self.num_kv_heads,
            ).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, activation: str = "relu2"):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.activation = activation
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.activation == "leakyrelu2":
            x = F.leaky_relu(x, negative_slope=0.5)
        else:
            x = torch.relu(x)
        return self.proj(x.square())
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        mlp_activation: str = "relu2",
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, activation=mlp_activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        mlp_activation: str = "relu2",
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.mlp_activation = mlp_activation
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                    mlp_activation=mlp_activation,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        """Get value embedding for a specific layer using shared table + per-layer scale."""
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    log0=print,
) -> tuple[float, float]:
    """Legal score-first TTT: score each chunk before adapting on it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log0(
        f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
        f"total_windows={len(window_starts)} stride={stride} "
        f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
        f"freeze_blocks={args.ttt_freeze_blocks}"
    )
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = any(f"blocks.{bi}." in name for bi in frozen_block_ids)
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)
    log0(
        f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
        f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}"
    )
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()
        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci + 1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log0(
        f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
        f"elapsed={time.perf_counter() - t0:.1f}s"
    )
    return val_loss, val_bpb
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_intn_per_row(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    clip_range = max(1, (1 << (bits - 1)) - 1)
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    bits = int(round(math.log2(clip_range + 1))) + 1
    return quantize_intn_per_row(t, bits)
def _is_gradquant_candidate(name: str, t: Tensor, quant_cats: set[str], min_numel: int) -> bool:
    if not t.is_floating_point():
        return False
    if t.ndim < 1 or t.numel() <= min_numel:
        return False
    if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
        return False
    return _classify_param(name) in quant_cats
def build_gradquant_bits_map(
    state_dict: dict[str, Tensor],
    sensitivity_scores: dict[str, float],
    quant_cats: set[str],
    min_numel: int,
    top_frac: float,
    bottom_frac: float,
) -> dict[str, int]:
    candidates: list[tuple[str, float]] = []
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if _is_gradquant_candidate(name, t, quant_cats, min_numel):
            candidates.append((name, float(sensitivity_scores.get(name, 0.0))))
    if not candidates:
        return {}
    candidates.sort(key=lambda x: x[1], reverse=True)
    count = len(candidates)
    top_n = min(count, max(0, int(round(count * top_frac))))
    bottom_n = min(count - top_n, max(0, int(round(count * bottom_frac))))
    bits_map = {name: 6 for name, _ in candidates}
    for name, _ in candidates[:top_n]:
        bits_map[name] = 7
    if bottom_n > 0:
        for name, _ in candidates[-bottom_n:]:
            bits_map[name] = 5
    return bits_map
def summarize_gradquant_bits(bits_map: dict[str, int], state_dict: dict[str, Tensor]) -> dict[int, dict[str, int]]:
    summary: dict[int, dict[str, int]] = {}
    for name, bits in bits_map.items():
        slot = summary.setdefault(bits, {"tensors": 0, "params": 0})
        slot["tensors"] += 1
        slot["params"] += int(state_dict[name].numel())
    return summary
def build_export_profile_bits_map(
    state_dict: dict[str, Tensor],
    profile: str,
    min_numel: int,
) -> dict[str, int]:
    if not profile:
        return {}
    bits_map: dict[str, int] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if not _is_gradquant_candidate(name, t, {"mlp", "attn"}, min_numel):
            continue
        if profile == "block10_attn7":
            if name.startswith("blocks.10.") and ".attn." in name:
                bits_map[name] = 7
        elif profile == "block10_qvproj7":
            if name.startswith("blocks.10.") and any(token in name for token in (".attn.c_q.", ".attn.c_v.", ".attn.proj.")):
                bits_map[name] = 7
        elif profile == "block10_mlp7":
            if name.startswith("blocks.10.") and ".mlp." in name:
                bits_map[name] = 7
        elif profile == "block10_mlp_qv7":
            if (
                name.startswith("blocks.10.")
                and ".mlp." in name
            ) or (
                name.startswith("blocks.10.")
                and any(token in name for token in (".attn.c_q.", ".attn.c_v."))
            ):
                bits_map[name] = 7
        elif profile == "block10_mlp_qproj7":
            if (
                name.startswith("blocks.10.")
                and ".mlp." in name
            ) or name in ("blocks.10.attn.c_q.weight", "blocks.10.attn.proj.weight"):
                bits_map[name] = 7
        elif profile == "block10_mlp_k7":
            if (
                name.startswith("blocks.10.")
                and ".mlp." in name
            ) or name == "blocks.10.attn.c_k.weight":
                bits_map[name] = 7
        elif profile == "block10_mlp_kproj7":
            if (
                name.startswith("blocks.10.")
                and ".mlp." in name
            ) or name in ("blocks.10.attn.c_k.weight", "blocks.10.attn.proj.weight"):
                bits_map[name] = 7
        elif profile == "block10_mlp_vproj7":
            if (
                name.startswith("blocks.10.")
                and ".mlp." in name
            ) or name in ("blocks.10.attn.c_v.weight", "blocks.10.attn.proj.weight"):
                bits_map[name] = 7
        elif profile == "block10_mlp_fc_qv7":
            if name == "blocks.10.mlp.fc.weight" or (
                name.startswith("blocks.10.")
                and any(token in name for token in (".attn.c_q.", ".attn.c_v."))
            ):
                bits_map[name] = 7
        elif profile == "block10_attn_mlp7":
            if name.startswith("blocks.10.") and (".attn." in name or ".mlp." in name):
                bits_map[name] = 7
        else:
            raise ValueError(f"Unknown EXPORT_BITS_PROFILE={profile!r}")
    return bits_map
def mixed_quantize_int6(
    state_dict: dict[str, Tensor],
    int6_cats: set[str],
    bits_by_name: dict[str, int] | None = None,
):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    late_k_layers = set(range(num_layers_total - 2, num_layers_total))
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            bits = int(bits_by_name.get(name, 6)) if bits_by_name is not None else 6
            q, s = quantize_intn_per_row(t, bits)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "intn", "bits": bits}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def load_coord_sidecars(sidecar_path: str) -> dict[str, dict[str, Tensor]]:
    path = Path(sidecar_path)
    if not path.exists():
        raise FileNotFoundError(f"EXPORT_COORD_SIDECAR_PATH does not exist: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported coord sidecar payload type: {type(obj)!r}")
    sidecars = obj.get("sidecars", obj)
    if not isinstance(sidecars, dict):
        raise TypeError("coord sidecar payload must be a dict or {'sidecars': dict}")
    out: dict[str, dict[str, Tensor]] = {}
    for name, spec in sidecars.items():
        if not isinstance(spec, dict):
            raise TypeError(f"coord sidecar spec for {name} must be dict")
        rows = spec.get("rows")
        cols = spec.get("cols")
        vals = spec.get("vals")
        if rows is None or cols is None or vals is None:
            raise ValueError(f"coord sidecar spec for {name} missing rows/cols/vals")
        out[name] = {
            "rows": torch.as_tensor(rows, device="cpu").to(torch.int32).contiguous(),
            "cols": torch.as_tensor(cols, device="cpu").to(torch.int16).contiguous(),
            "vals": torch.as_tensor(vals, device="cpu").to(torch.float16).contiguous(),
        }
    return out


def load_prepacked_blob(blob_path: str) -> bytes:
    path = Path(blob_path)
    if not path.exists():
        raise FileNotFoundError(f"EXPORT_PREPACKED_SBEV_PATH does not exist: {path}")
    return path.read_bytes()


def load_residualq_sidecars(sidecar_path: str) -> dict[str, dict[str, Tensor]]:
    path = Path(sidecar_path)
    if not path.exists():
        raise FileNotFoundError(f"EXPORT_RESIDUALQ_SIDECAR_PATH does not exist: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported residualq sidecar payload type: {type(obj)!r}")
    sidecars = obj.get("sidecars", obj)
    if not isinstance(sidecars, dict):
        raise TypeError("residualq sidecar payload must be a dict or {'sidecars': dict}")
    out: dict[str, dict[str, Tensor]] = {}
    for name, spec in sidecars.items():
        if not isinstance(spec, dict):
            raise TypeError(f"residualq sidecar spec for {name} must be dict")
        row_ids = spec.get("row_ids")
        row_ptr = spec.get("row_ptr")
        cols = spec.get("cols")
        dq_scale = spec.get("dq_scale")
        dq = spec.get("dq")
        width = spec.get("width")
        if row_ids is None or row_ptr is None or cols is None or dq_scale is None or dq is None or width is None:
            raise ValueError(f"residualq sidecar spec for {name} missing row_ids/row_ptr/cols/dq_scale/dq/width")
        out[name] = {
            "row_ids": torch.as_tensor(row_ids, device="cpu").to(torch.int16).contiguous(),
            "row_ptr": torch.as_tensor(row_ptr, device="cpu").to(torch.int32).contiguous(),
            "cols": torch.as_tensor(cols, device="cpu").to(torch.int16).contiguous(),
            "dq_scale": torch.as_tensor(dq_scale, device="cpu").to(torch.float16).contiguous(),
            "dq": torch.as_tensor(dq, device="cpu").to(torch.int8).contiguous(),
            "width": int(width),
        }
    return out


def apply_coord_sidecars_in_place(
    state_dict: dict[str, Tensor],
    sidecars: dict[str, dict[str, Tensor]] | None,
) -> None:
    if not sidecars:
        return
    for name, spec in sidecars.items():
        if name not in state_dict:
            continue
        rows = spec["rows"].long()
        cols = spec["cols"].long()
        vals = spec["vals"].to(dtype=state_dict[name].dtype)
        patched = state_dict[name].clone()
        patched[rows, cols] = vals
        state_dict[name] = patched


def apply_residualq_sidecars_in_place(
    state_dict: dict[str, Tensor],
    sidecars: dict[str, dict[str, Tensor]] | None,
) -> None:
    if not sidecars:
        return
    for name, spec in sidecars.items():
        if name not in state_dict:
            continue
        row_ids = spec["row_ids"].to(torch.int32)
        row_ptr = spec["row_ptr"].to(torch.int32)
        cols = spec["cols"].to(torch.int16)
        dq_scale = spec["dq_scale"].to(torch.float32)
        dq = spec["dq"].to(torch.float32)
        patched = state_dict[name].float().clone()
        for ridx, row_id in enumerate(row_ids.tolist()):
            start = int(row_ptr[ridx].item())
            end = int(row_ptr[ridx + 1].item())
            if end <= start:
                continue
            row_cols = cols[start:end].long()
            patched[row_id, row_cols] = patched[row_id, row_cols] + dq_scale[ridx] * dq[start:end]
        state_dict[name] = patched.to(dtype=state_dict[name].dtype)


def pack_coord_sidecars(sidecars: dict[str, dict[str, Tensor]]) -> bytes:
    if not sidecars:
        return b""
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        name_b = name.encode("utf-8")
        cols = spec["cols"].to(torch.int32)
        rows = spec["rows"].to(torch.int32)
        vals = spec["vals"].to(torch.float16)
        if "width" not in spec:
            raise ValueError(f"coord sidecar for {name} missing width")
        width = int(spec["width"])
        flat_idx = (rows * width + cols).to(torch.int32).contiguous()
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<II", width, int(flat_idx.numel())))
        parts.append(flat_idx.numpy().astype(np.uint32, copy=False).tobytes())
        parts.append(vals.numpy().astype(np.float16, copy=False).tobytes())
    return b"".join(parts)


def pack_grouped_coord_sidecars(sidecars: dict[str, dict[str, Tensor]]) -> bytes:
    if not sidecars:
        return b""
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        name_b = name.encode("utf-8")
        rows = spec["rows"].to(torch.int32)
        cols = spec["cols"].to(torch.int16)
        vals = spec["vals"].to(torch.float16)
        if "width" not in spec:
            raise ValueError(f"grouped coord sidecar for {name} missing width")
        width = int(spec["width"])
        row_ids, inverse = torch.unique(rows, sorted=True, return_inverse=True)
        row_ptr = torch.zeros((row_ids.numel() + 1,), dtype=torch.int32)
        cols_list: list[Tensor] = []
        vals_list: list[Tensor] = []
        cursor = 0
        for ridx in range(row_ids.numel()):
            mask = inverse == ridx
            row_cols = cols[mask].to(torch.int16)
            row_vals = vals[mask].to(torch.float16)
            order = torch.argsort(row_cols.long())
            row_cols = row_cols[order]
            row_vals = row_vals[order]
            row_ptr[ridx] = cursor
            cursor += int(row_cols.numel())
            cols_list.append(row_cols)
            vals_list.append(row_vals)
        row_ptr[-1] = cursor
        cols_cat = torch.cat(cols_list).to(torch.int16).contiguous() if cols_list else torch.empty((0,), dtype=torch.int16)
        vals_cat = torch.cat(vals_list).to(torch.float16).contiguous() if vals_list else torch.empty((0,), dtype=torch.float16)
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<IHI", width, int(row_ids.numel()), int(cols_cat.numel())))
        parts.append(row_ids.to(torch.int16).numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(row_ptr.numpy().astype(np.uint32, copy=False).tobytes())
        parts.append(cols_cat.numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(vals_cat.numpy().astype(np.float16, copy=False).tobytes())
    return b"".join(parts)


def pack_grouped_delta_coord_sidecars(sidecars: dict[str, dict[str, Tensor]]) -> bytes:
    if not sidecars:
        return b""
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        name_b = name.encode("utf-8")
        rows = spec["rows"].to(torch.int32)
        cols = spec["cols"].to(torch.int32)
        vals = spec["vals"].to(torch.float16)
        if "width" not in spec:
            raise ValueError(f"grouped-delta coord sidecar for {name} missing width")
        width = int(spec["width"])
        row_ids, inverse = torch.unique(rows, sorted=True, return_inverse=True)
        row_ptr = torch.zeros((row_ids.numel() + 1,), dtype=torch.int32)
        first_cols = torch.zeros((row_ids.numel(),), dtype=torch.int16)
        gap_bytes = bytearray()
        overflow_pos: list[int] = []
        overflow_vals: list[int] = []
        vals_list: list[Tensor] = []
        cursor = 0
        gap_cursor = 0
        for ridx in range(row_ids.numel()):
            mask = inverse == ridx
            row_cols = cols[mask]
            row_vals = vals[mask].to(torch.float16)
            order = torch.argsort(row_cols)
            row_cols = row_cols[order].to(torch.int32)
            row_vals = row_vals[order]
            row_ptr[ridx] = cursor
            cursor += int(row_cols.numel())
            if row_cols.numel():
                first_cols[ridx] = int(row_cols[0].item())
                if row_cols.numel() > 1:
                    diffs = torch.diff(row_cols)
                    for gap in diffs.tolist():
                        if gap < 255:
                            gap_bytes.append(gap)
                        else:
                            gap_bytes.append(255)
                            overflow_pos.append(gap_cursor)
                            overflow_vals.append(gap)
                        gap_cursor += 1
            vals_list.append(row_vals)
        row_ptr[-1] = cursor
        vals_cat = torch.cat(vals_list).to(torch.float16).contiguous() if vals_list else torch.empty((0,), dtype=torch.float16)
        overflow_pos_arr = np.asarray(overflow_pos, dtype=np.uint32) if overflow_pos else np.empty((0,), dtype=np.uint32)
        overflow_vals_arr = np.asarray(overflow_vals, dtype=np.uint16) if overflow_vals else np.empty((0,), dtype=np.uint16)
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<IHI", width, int(row_ids.numel()), int(vals_cat.numel())))
        parts.append(row_ids.to(torch.int16).numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(row_ptr.numpy().astype(np.uint32, copy=False).tobytes())
        parts.append(first_cols.numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(struct.pack("<I", len(gap_bytes)))
        parts.append(bytes(gap_bytes))
        parts.append(struct.pack("<I", int(overflow_pos_arr.size)))
        parts.append(overflow_pos_arr.tobytes())
        parts.append(overflow_vals_arr.tobytes())
        parts.append(vals_cat.numpy().astype(np.float16, copy=False).tobytes())
    return b"".join(parts)


def pack_grouped_delta_byteplane_coord_sidecars(
    sidecars: dict[str, dict[str, Tensor]], value_mode_by_name: dict[str, str] | None = None
) -> bytes:
    if not sidecars:
        return b""
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        mode = value_mode_by_name.get(name, "byteplane") if value_mode_by_name else "byteplane"
        name_b = name.encode("utf-8")
        rows = spec["rows"].to(torch.int32)
        cols = spec["cols"].to(torch.int32)
        vals = spec["vals"].to(torch.float16)
        if "width" not in spec:
            raise ValueError(f"grouped-delta-byteplane coord sidecar for {name} missing width")
        width = int(spec["width"])
        row_ids, inverse = torch.unique(rows, sorted=True, return_inverse=True)
        row_ptr = torch.zeros((row_ids.numel() + 1,), dtype=torch.int32)
        first_cols = torch.zeros((row_ids.numel(),), dtype=torch.int16)
        gap_bytes = bytearray()
        overflow_pos: list[int] = []
        overflow_vals: list[int] = []
        vals_list: list[Tensor] = []
        cursor = 0
        gap_cursor = 0
        for ridx in range(row_ids.numel()):
            mask = inverse == ridx
            row_cols = cols[mask]
            row_vals = vals[mask].to(torch.float16)
            order = torch.argsort(row_cols)
            row_cols = row_cols[order].to(torch.int32)
            row_vals = row_vals[order]
            row_ptr[ridx] = cursor
            cursor += int(row_cols.numel())
            if row_cols.numel():
                first_cols[ridx] = int(row_cols[0].item())
                if row_cols.numel() > 1:
                    diffs = torch.diff(row_cols)
                    for gap in diffs.tolist():
                        if gap < 255:
                            gap_bytes.append(gap)
                        else:
                            gap_bytes.append(255)
                            overflow_pos.append(gap_cursor)
                            overflow_vals.append(gap)
                        gap_cursor += 1
            vals_list.append(row_vals)
        row_ptr[-1] = cursor
        vals_cat = torch.cat(vals_list).to(torch.float16).contiguous() if vals_list else torch.empty((0,), dtype=torch.float16)
        words = vals_cat.view(torch.uint16).cpu().numpy().astype(np.uint16, copy=False)
        mode_b = mode.encode("utf-8")
        value_blob = _encode_value_words_mixed(words, row_ptr.numpy().astype(np.uint32, copy=False), mode)
        overflow_pos_arr = np.asarray(overflow_pos, dtype=np.uint32) if overflow_pos else np.empty((0,), dtype=np.uint32)
        overflow_vals_arr = np.asarray(overflow_vals, dtype=np.uint16) if overflow_vals else np.empty((0,), dtype=np.uint16)
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<IHI", width, int(row_ids.numel()), int(vals_cat.numel())))
        parts.append(struct.pack("<H", len(mode_b)))
        parts.append(mode_b)
        parts.append(row_ids.to(torch.int16).numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(row_ptr.numpy().astype(np.uint32, copy=False).tobytes())
        parts.append(first_cols.numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(struct.pack("<I", len(gap_bytes)))
        parts.append(bytes(gap_bytes))
        parts.append(struct.pack("<I", int(overflow_pos_arr.size)))
        parts.append(overflow_pos_arr.tobytes())
        parts.append(overflow_vals_arr.tobytes())
        parts.append(value_blob)
    return b"".join(parts)


def _pack_fixedbits_unsigned(values: np.ndarray, bits: int) -> bytes:
    if values.size == 0:
        return b""
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for value in values.tolist():
        acc |= (int(value) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def _fit_kmeans_1d(values: np.ndarray, k: int, iters: int = 10) -> np.ndarray:
    if values.size == 0:
        return np.zeros((k,), dtype=np.float32)
    unique = np.unique(values.astype(np.float32, copy=False))
    if unique.size >= k:
        centers = np.quantile(values, np.linspace(0.0, 1.0, k, dtype=np.float32)).astype(np.float32, copy=False)
    else:
        centers = np.empty((k,), dtype=np.float32)
        centers[: unique.size] = unique
        centers[unique.size :] = unique[-1]
    centers = np.sort(centers.astype(np.float32, copy=False))
    for _ in range(iters):
        if centers.size <= 1:
            break
        bounds = (centers[:-1] + centers[1:]) * 0.5
        assign = np.searchsorted(bounds, values, side="left")
        new_centers = centers.copy()
        for idx in range(centers.size):
            mask = assign == idx
            if np.any(mask):
                new_centers[idx] = values[mask].mean(dtype=np.float32)
        centers = np.sort(new_centers.astype(np.float32, copy=False))
    return centers


def _encode_rowcodebook_global_escape_words(
    words: np.ndarray,
    row_ptr: np.ndarray,
    bits: int,
    total_escapes: int,
    clip_quantile: float,
    per_row_cap: int = 7,
    center_mode: str = "mean",
) -> bytes:
    k = 1 << bits
    qvalues = np.empty((words.size,), dtype=np.float32)
    num_rows = row_ptr.size - 1
    centers = np.zeros((num_rows,), dtype=np.float16)
    scales = np.zeros((num_rows,), dtype=np.float16)
    values = words.view(np.float16).astype(np.float32, copy=False)
    for ridx in range(num_rows):
        start = int(row_ptr[ridx])
        end = int(row_ptr[ridx + 1])
        if start >= end:
            continue
        row_vals = values[start:end]
        if center_mode == "median":
            center = float(np.median(row_vals))
        else:
            center = float(row_vals.mean())
        centered = row_vals - center
        abs_centered = np.abs(centered)
        max_abs = float(np.quantile(abs_centered, clip_quantile))
        max_abs = max(max_abs, 1e-8)
        scale = 0.0 if max_abs == 0.0 else max_abs
        centers[ridx] = np.float16(center)
        scales[ridx] = np.float16(scale)
        if scale == 0.0:
            qvalues[start:end] = 0.0
        else:
            qvalues[start:end] = np.clip(centered / scale, -1.0, 1.0)
    codebook = _fit_kmeans_1d(qvalues, k).astype(np.float16, copy=False)
    sorted_codebook = np.sort(codebook.astype(np.float32, copy=False))
    if sorted_codebook.size <= 1:
        codes = np.zeros((qvalues.size,), dtype=np.uint8)
    else:
        bounds = (sorted_codebook[:-1] + sorted_codebook[1:]) * 0.5
        codes = np.searchsorted(bounds, qvalues, side="left").astype(np.uint8, copy=False)
    recon = codebook.astype(np.float32, copy=False)[codes.astype(np.int32, copy=False)]
    candidates: list[tuple[float, int, int, np.float16]] = []
    for ridx in range(num_rows):
        start = int(row_ptr[ridx])
        end = int(row_ptr[ridx + 1])
        if start >= end:
            continue
        scale = float(scales[ridx])
        center = float(centers[ridx])
        if scale == 0.0:
            recon_vals = np.full((end - start,), center, dtype=np.float32)
        else:
            recon_vals = center + scale * recon[start:end]
        row_vals = values[start:end]
        residual = np.abs(row_vals - recon_vals)
        local_order = np.argsort(-residual, kind="stable")
        local_keep = min(per_row_cap, local_order.size)
        for local_pos in local_order[:local_keep].tolist():
            candidates.append((float(residual[local_pos]), ridx, int(local_pos), np.float16(row_vals[local_pos])))
    candidates.sort(key=lambda item: item[0], reverse=True)
    row_counts = np.zeros((num_rows,), dtype=np.uint8)
    row_pos_lists: list[list[int]] = [[] for _ in range(num_rows)]
    row_val_lists: list[list[np.float16]] = [[] for _ in range(num_rows)]
    selected = 0
    for _score, ridx, local_pos, val in candidates:
        if selected >= total_escapes:
            break
        if row_counts[ridx] >= per_row_cap:
            continue
        if local_pos in row_pos_lists[ridx]:
            continue
        row_counts[ridx] += 1
        row_pos_lists[ridx].append(local_pos)
        row_val_lists[ridx].append(val)
        selected += 1
    escape_pos: list[int] = []
    escape_vals: list[np.float16] = []
    for ridx in range(num_rows):
        if not row_pos_lists[ridx]:
            continue
        order = np.argsort(np.asarray(row_pos_lists[ridx], dtype=np.int32), kind="stable")
        for idx in order.tolist():
            escape_pos.append(int(row_pos_lists[ridx][idx]))
            escape_vals.append(np.float16(row_val_lists[ridx][idx]))
    counts_blob = _pack_fixedbits_unsigned(row_counts, 3)
    pos_blob = np.asarray(escape_pos, dtype=np.uint8).tobytes() if escape_pos else b""
    vals_blob = np.asarray(escape_vals, dtype=np.float16).tobytes() if escape_vals else b""
    return (
        struct.pack("<H", k)
        + codebook.tobytes()
        + centers.tobytes()
        + scales.tobytes()
        + _pack_fixedbits_unsigned(codes, bits)
        + counts_blob
        + pos_blob
        + vals_blob
    )


def _encode_value_words_mixed(words: np.ndarray, row_ptr: np.ndarray, mode: str) -> bytes:
    if mode == "byteplane":
        hi = (words >> 8).astype(np.uint8, copy=False)
        lo = (words & 0xFF).astype(np.uint8, copy=False)
        return hi.tobytes() + lo.tobytes()
    if mode == "xor_row":
        x = words.copy()
        for start, end in zip(row_ptr[:-1], row_ptr[1:]):
            start_i = int(start)
            end_i = int(end)
            if end_i - start_i <= 1:
                continue
            prev = x[start_i]
            for idx in range(start_i + 1, end_i):
                cur = x[idx]
                x[idx] = np.uint16(cur ^ prev)
                prev = cur
        return x.tobytes()
    if mode == "bitplane16":
        parts: list[bytes] = []
        for bit in range(15, -1, -1):
            plane = ((words >> bit) & 1).astype(np.uint8, copy=False)
            parts.append(np.packbits(plane, bitorder="little").tobytes())
        return b"".join(parts)
    if mode == "floatord_gray_byteplane":
        pos = (words & 0x8000) == 0
        ordered = np.where(pos, words ^ 0x8000, (~words) & 0xFFFF).astype(np.uint16, copy=False)
        gray = np.bitwise_xor(ordered, ordered >> 1).astype(np.uint16, copy=False)
        hi = (gray >> 8).astype(np.uint8, copy=False)
        lo = (gray & 0xFF).astype(np.uint8, copy=False)
        return hi.tobytes() + lo.tobytes()
    if mode == "rowcodebook6g384p95":
        return _encode_rowcodebook_global_escape_words(words, row_ptr, bits=6, total_escapes=384, clip_quantile=0.95)
    if mode == "rowcodebook6g448p95":
        return _encode_rowcodebook_global_escape_words(words, row_ptr, bits=6, total_escapes=448, clip_quantile=0.95)
    if mode == "rowcodebook6g512p95":
        return _encode_rowcodebook_global_escape_words(words, row_ptr, bits=6, total_escapes=512, clip_quantile=0.95)
    raise ValueError(f"Unknown value mode: {mode}")


def _unpack_fixedbits_unsigned(blob: memoryview, count: int, bits: int) -> tuple[np.ndarray, int]:
    if count == 0:
        return np.empty((0,), dtype=np.uint8), 0
    total_bits = count * bits
    total_bytes = (total_bits + 7) // 8
    acc = 0
    acc_bits = 0
    cursor = 0
    mask = (1 << bits) - 1
    out = np.empty((count,), dtype=np.uint8)
    for idx in range(count):
        while acc_bits < bits:
            acc |= int(blob[cursor]) << acc_bits
            cursor += 1
            acc_bits += 8
        out[idx] = acc & mask
        acc >>= bits
        acc_bits -= bits
    return out, total_bytes


def _decode_rowcodebook_escape_words(blob: memoryview, nnz: int, row_ptr: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    num_rows = row_ptr.size - 1
    cursor = 0
    k, escapes_per_row = struct.unpack_from("<HB", blob, cursor)
    cursor += 3
    codebook_nbytes = k * 2
    codebook = np.frombuffer(blob[cursor : cursor + codebook_nbytes], dtype=np.float16).astype(np.float32, copy=True)
    cursor += codebook_nbytes
    centers = np.frombuffer(blob[cursor : cursor + num_rows * 2], dtype=np.float16).astype(np.float32, copy=True)
    cursor += num_rows * 2
    scales = np.frombuffer(blob[cursor : cursor + num_rows * 2], dtype=np.float16).astype(np.float32, copy=True)
    cursor += num_rows * 2
    codes, used = _unpack_fixedbits_unsigned(blob[cursor:], nnz, bits)
    cursor += used
    escape_pos_nbytes = num_rows * escapes_per_row
    escape_vals_nbytes = num_rows * escapes_per_row * 2
    escape_pos = np.frombuffer(blob[cursor : cursor + escape_pos_nbytes], dtype=np.uint8).copy().reshape(num_rows, escapes_per_row)
    cursor += escape_pos_nbytes
    escape_vals = np.frombuffer(blob[cursor : cursor + escape_vals_nbytes], dtype=np.float16).copy().reshape(num_rows, escapes_per_row)
    cursor += escape_vals_nbytes
    values = np.empty((nnz,), dtype=np.float16)
    codebook_f = codebook.astype(np.float32, copy=False)
    for ridx in range(num_rows):
        start = int(row_ptr[ridx])
        end = int(row_ptr[ridx + 1])
        if start >= end:
            continue
        scale = float(scales[ridx])
        center = float(centers[ridx])
        if scale == 0.0:
            row_vals = np.full((end - start,), center, dtype=np.float32)
        else:
            row_vals = center + scale * codebook_f[codes[start:end].astype(np.int32, copy=False)]
        row_vals = row_vals.astype(np.float16, copy=False)
        for pos, val in zip(escape_pos[ridx].tolist(), escape_vals[ridx].tolist()):
            if pos == 255:
                continue
            row_vals[int(pos)] = np.float16(val)
        values[start:end] = row_vals
    return values.view(np.uint16), cursor


def _decode_rowcodebook_global_escape_words(blob: memoryview, nnz: int, row_ptr: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    num_rows = row_ptr.size - 1
    cursor = 0
    (k,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    codebook_nbytes = k * 2
    codebook = np.frombuffer(blob[cursor : cursor + codebook_nbytes], dtype=np.float16).astype(np.float32, copy=True)
    cursor += codebook_nbytes
    centers = np.frombuffer(blob[cursor : cursor + num_rows * 2], dtype=np.float16).astype(np.float32, copy=True)
    cursor += num_rows * 2
    scales = np.frombuffer(blob[cursor : cursor + num_rows * 2], dtype=np.float16).astype(np.float32, copy=True)
    cursor += num_rows * 2
    codes, used = _unpack_fixedbits_unsigned(blob[cursor:], nnz, bits)
    cursor += used
    row_counts, used = _unpack_fixedbits_unsigned(blob[cursor:], num_rows, 3)
    cursor += used
    total_escapes = int(row_counts.astype(np.int32, copy=False).sum())
    escape_pos = np.frombuffer(blob[cursor : cursor + total_escapes], dtype=np.uint8).copy()
    cursor += total_escapes
    escape_vals_nbytes = total_escapes * 2
    escape_vals = np.frombuffer(blob[cursor : cursor + escape_vals_nbytes], dtype=np.float16).copy()
    cursor += escape_vals_nbytes
    values = np.empty((nnz,), dtype=np.float16)
    codebook_f = codebook.astype(np.float32, copy=False)
    pos_cursor = 0
    for ridx in range(num_rows):
        start = int(row_ptr[ridx])
        end = int(row_ptr[ridx + 1])
        if start >= end:
            continue
        scale = float(scales[ridx])
        center = float(centers[ridx])
        if scale == 0.0:
            row_vals = np.full((end - start,), center, dtype=np.float32)
        else:
            row_vals = center + scale * codebook_f[codes[start:end].astype(np.int32, copy=False)]
        row_vals = row_vals.astype(np.float16, copy=False)
        row_escape_count = int(row_counts[ridx])
        for _ in range(row_escape_count):
            row_vals[int(escape_pos[pos_cursor])] = np.float16(escape_vals[pos_cursor])
            pos_cursor += 1
        values[start:end] = row_vals
    return values.view(np.uint16), cursor


def _decode_rowpow2_words(blob: memoryview, nnz: int, row_ptr: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    levels = (1 << bits) - 1
    num_rows = row_ptr.size - 1
    cursor = 0
    centers = np.frombuffer(blob[cursor : cursor + num_rows * 2], dtype=np.float16).astype(np.float32, copy=True)
    cursor += num_rows * 2
    exp_q = np.frombuffer(blob[cursor : cursor + num_rows], dtype=np.int8).astype(np.int32, copy=True)
    cursor += num_rows
    qcodes, used = _unpack_fixedbits_unsigned(blob[cursor:], nnz, bits)
    cursor += used
    row_counts, used = _unpack_fixedbits_unsigned(blob[cursor:], num_rows, 3)
    cursor += used
    total_escapes = int(row_counts.astype(np.int32, copy=False).sum())
    escape_pos = np.frombuffer(blob[cursor : cursor + total_escapes], dtype=np.uint8).copy()
    cursor += total_escapes
    escape_vals_nbytes = total_escapes * 2
    escape_vals = np.frombuffer(blob[cursor : cursor + escape_vals_nbytes], dtype=np.float16).copy()
    cursor += escape_vals_nbytes
    q = (qcodes.astype(np.float32, copy=False) / (levels * 0.5)) - 1.0
    values = np.empty((nnz,), dtype=np.float16)
    pos_cursor = 0
    for ridx in range(num_rows):
        start = int(row_ptr[ridx])
        end = int(row_ptr[ridx + 1])
        if start >= end:
            continue
        center = float(centers[ridx])
        scale = float(np.ldexp(1.0, int(exp_q[ridx])))
        row_vals = (center + scale * q[start:end]).astype(np.float16, copy=False)
        row_escape_count = int(row_counts[ridx])
        for _ in range(row_escape_count):
            row_vals[int(escape_pos[pos_cursor])] = np.float16(escape_vals[pos_cursor])
            pos_cursor += 1
        values[start:end] = row_vals
    return values.view(np.uint16), cursor


def _decode_s1e5_deltaexp_split_words(
    blob: memoryview, nnz: int, row_ptr: np.ndarray, keep_bits: int, delta_limit: int
) -> tuple[np.ndarray, int]:
    num_rows = row_ptr.size - 1
    sign_bits, used_sign = _unpack_fixedbits_unsigned(blob, nnz, 1)
    base_exps, used_base = _unpack_fixedbits_unsigned(blob[used_sign:], num_rows, 5)
    exp_codes, used_codes = _unpack_fixedbits_unsigned(blob[used_sign + used_base :], nnz, 3)
    esc_code = 2 * int(delta_limit) + 1
    num_escapes = int(np.count_nonzero(exp_codes == esc_code))
    escape_exps, used_esc = _unpack_fixedbits_unsigned(blob[used_sign + used_base + used_codes :], num_escapes, 5)
    mant_codes, used_mant = _unpack_fixedbits_unsigned(
        blob[used_sign + used_base + used_codes + used_esc :], nnz, keep_bits
    )
    exponents = np.empty((nnz,), dtype=np.uint8)
    esc_cursor = 0
    for ridx in range(num_rows):
        start = int(row_ptr[ridx])
        end = int(row_ptr[ridx + 1])
        if start >= end:
            continue
        base_exp = int(base_exps[ridx])
        row_codes = exp_codes[start:end].astype(np.int32, copy=False)
        row_exps = np.empty((end - start,), dtype=np.uint8)
        for idx, code in enumerate(row_codes.tolist()):
            if code == esc_code:
                row_exps[idx] = np.uint8(escape_exps[esc_cursor])
                esc_cursor += 1
            else:
                row_exps[idx] = np.uint8(base_exp + code - delta_limit)
        exponents[start:end] = row_exps
    words = (
        (sign_bits.astype(np.uint16, copy=False) << 15)
        | (exponents.astype(np.uint16, copy=False) << 10)
        | (mant_codes.astype(np.uint16, copy=False) << (10 - int(keep_bits)))
    )
    return words.astype(np.uint16, copy=False), used_sign + used_base + used_codes + used_esc + used_mant


def _decode_value_words_mixed(blob: memoryview, nnz: int, row_ptr: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    nbytes = nnz * 2
    if mode == "byteplane":
        hi = np.frombuffer(blob[:nnz], dtype=np.uint8).astype(np.uint16, copy=True)
        lo = np.frombuffer(blob[nnz:nbytes], dtype=np.uint8).astype(np.uint16, copy=True)
        return ((hi << 8) | lo).astype(np.uint16, copy=False), nbytes
    if mode == "xor_row":
        words = np.frombuffer(blob[:nbytes], dtype=np.uint16).copy()
        for start, end in zip(row_ptr[:-1], row_ptr[1:]):
            start_i = int(start)
            end_i = int(end)
            if end_i - start_i <= 1:
                continue
            prev = words[start_i]
            for idx in range(start_i + 1, end_i):
                prev = np.uint16(words[idx] ^ prev)
                words[idx] = prev
        return words, nbytes
    if mode == "bitplane16":
        packed_len = (nnz + 7) // 8
        words = np.zeros((nnz,), dtype=np.uint16)
        cursor = 0
        for bit in range(15, -1, -1):
            packed = np.frombuffer(blob[cursor : cursor + packed_len], dtype=np.uint8).copy()
            cursor += packed_len
            plane = np.unpackbits(packed, bitorder="little")[:nnz].astype(np.uint16, copy=False)
            words |= plane << bit
        return words, cursor
    if mode == "floatord_gray_byteplane":
        hi = np.frombuffer(blob[:nnz], dtype=np.uint8).astype(np.uint16, copy=True)
        lo = np.frombuffer(blob[nnz:nbytes], dtype=np.uint8).astype(np.uint16, copy=True)
        gray = (hi << 8) | lo
        ordered = gray.copy()
        shift = 1
        while shift < 16:
            ordered ^= ordered >> shift
            shift <<= 1
        pos = (ordered & 0x8000) != 0
        words = np.where(pos, ordered ^ 0x8000, (~ordered) & 0xFFFF).astype(np.uint16, copy=False)
        return words, nbytes
    if mode == "rowcodebook6e2p95":
        return _decode_rowcodebook_escape_words(blob, nnz, row_ptr, bits=6)
    if mode == "rowcodebook6g384p95":
        return _decode_rowcodebook_global_escape_words(blob, nnz, row_ptr, bits=6)
    if mode == "rowcodebook6g448p95":
        return _decode_rowcodebook_global_escape_words(blob, nnz, row_ptr, bits=6)
    if mode == "rowcodebook6g512p95":
        return _decode_rowcodebook_global_escape_words(blob, nnz, row_ptr, bits=6)
    if mode == "rowpow2g384p95":
        return _decode_rowpow2_words(blob, nnz, row_ptr, bits=6)
    if mode == "rowpow2r384p95":
        return _decode_rowpow2_words(blob, nnz, row_ptr, bits=6)
    if mode == "s1e5d3_mant7_split":
        return _decode_s1e5_deltaexp_split_words(blob, nnz, row_ptr, keep_bits=7, delta_limit=3)
    if mode == "s1e5d3_mant6_split":
        return _decode_s1e5_deltaexp_split_words(blob, nnz, row_ptr, keep_bits=6, delta_limit=3)
    raise ValueError(f"Unknown value mode: {mode}")


def _mixed_value_mode_for_name(name: str) -> str:
    raw = os.environ.get("EXPORT_VALUE_MODE_OVERRIDES", "").strip()
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Bad EXPORT_VALUE_MODE_OVERRIDES item: {item!r}")
            override_name, override_mode = item.split("=", 1)
            if override_name.strip() == name:
                return override_mode.strip()
    if name == "blocks.10.attn.c_v.weight":
        return "xor_row"
    if name == "blocks.10.mlp.proj.weight":
        return "bitplane16"
    return "byteplane"


_ROWREF_CK_NAME = "blocks.10.attn.c_k.weight"
_ROWREF_MLP_PROJ_NAME = "blocks.10.mlp.proj.weight"
_ROWREF_SIDECAR_NAMES = frozenset((_ROWREF_CK_NAME, _ROWREF_MLP_PROJ_NAME))


def _rowref_row_specs(spec: dict[str, Tensor]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rows = spec["rows"].to(torch.int32)
    cols = spec["cols"].to(torch.int32)
    vals = spec["vals"].to(torch.float16)
    row_ids = torch.unique(rows, sorted=True)
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for row_id in row_ids.tolist():
        mask = rows == row_id
        row_cols = cols[mask]
        row_vals = vals[mask]
        order = torch.argsort(row_cols)
        out.append(
            (
                int(row_id),
                row_cols[order].to(torch.int32).cpu().numpy().astype(np.int32, copy=False),
                row_vals[order].to(torch.float16).cpu().numpy().astype(np.float16, copy=False),
            )
        )
    return out


def _pack_rowref_single_coord_sidecar(spec: dict[str, Tensor], value_mode: str, ref_window: int) -> bytes:
    rowspecs = _rowref_row_specs(spec)
    width = int(spec["width"])
    total_rows = max((row_id for row_id, _, _ in rowspecs), default=-1) + 1
    dense_rows = len(rowspecs) == total_rows and rowspecs and rowspecs[0][0] == 0 and rowspecs[-1][0] == total_rows - 1
    bitmap = np.zeros((math.ceil(total_rows / 8),), dtype=np.uint8) if not dense_rows else np.empty((0,), dtype=np.uint8)
    records = bytearray()
    row_ptr = [0]
    words_list: list[np.ndarray] = []
    prev_cols: list[np.ndarray] = []

    for row_id, cols, vals in rowspecs:
        if not dense_rows:
            bitmap[row_id >> 3] |= np.uint8(1 << (row_id & 7))
        literal_cost = 2 + int(cols.size) * 2
        best_ref: tuple[int, list[int], list[int]] | None = None
        best_cost = literal_cost
        cur_cols = cols.tolist()
        cur_set = set(cur_cols)
        for back in range(1, min(ref_window, len(prev_cols)) + 1):
            ref_cols = prev_cols[-back]
            ref_list = ref_cols.tolist()
            drops = [pos for pos, col in enumerate(ref_list) if int(col) not in cur_set]
            ref_set = set(ref_list)
            adds = [int(col) for col in cur_cols if int(col) not in ref_set]
            ref_cost = 4 + len(drops) + len(adds) * 2
            if ref_cost < best_cost:
                best_cost = ref_cost
                best_ref = (back, drops, adds)
        if best_ref is not None:
            back, drops, adds = best_ref
            records.extend(struct.pack("<BBBB", 1, back, len(drops), len(adds)))
            if drops:
                records.extend(np.asarray(drops, dtype=np.uint8).tobytes())
            if adds:
                records.extend(np.asarray(adds, dtype=np.uint16).tobytes())
        else:
            records.extend(struct.pack("<BB", 0, int(cols.size)))
            if cols.size:
                records.extend(cols.astype(np.uint16, copy=False).tobytes())
        row_ptr.append(row_ptr[-1] + int(vals.size))
        words_list.append(vals.view(np.uint16))
        prev_cols.append(cols.astype(np.int32, copy=False))
        if len(prev_cols) > ref_window:
            prev_cols.pop(0)

    row_ptr_arr = np.asarray(row_ptr, dtype=np.uint32)
    words = np.concatenate(words_list).astype(np.uint16, copy=False) if words_list else np.empty((0,), dtype=np.uint16)
    value_blob = _encode_value_words_mixed(words, row_ptr_arr, value_mode)
    mode_b = value_mode.encode("utf-8")
    return b"".join(
        [
            struct.pack("<IHHHII", width, total_rows, len(rowspecs), len(mode_b), len(records), len(value_blob)),
            mode_b,
            struct.pack("<B", 1 if dense_rows else 0),
            bitmap.tobytes(),
            bytes(records),
            value_blob,
        ]
    )


def _unpack_rowref_single_coord_sidecar(blob: bytes) -> dict[str, Tensor]:
    mv = memoryview(blob)
    cursor = 0
    width, total_rows, num_rows, mode_len, records_len, values_len = struct.unpack_from("<IHHHII", mv, cursor)
    cursor += struct.calcsize("<IHHHII")
    value_mode = bytes(mv[cursor : cursor + mode_len]).decode("utf-8")
    cursor += mode_len
    dense_rows = bool(mv[cursor])
    cursor += 1
    bitmap_nbytes = 0 if dense_rows else math.ceil(total_rows / 8)
    bitmap = np.frombuffer(mv[cursor : cursor + bitmap_nbytes], dtype=np.uint8).copy() if bitmap_nbytes else np.empty((0,), dtype=np.uint8)
    cursor += bitmap_nbytes
    records_mv = mv[cursor : cursor + records_len]
    cursor += records_len
    values_mv = mv[cursor : cursor + values_len]

    if dense_rows:
        row_ids = list(range(total_rows))
    else:
        row_ids = [row_id for row_id in range(total_rows) if bitmap[row_id >> 3] & (1 << (row_id & 7))]
    if len(row_ids) != num_rows:
        raise ValueError(f"rowref row mismatch: {len(row_ids)} != {num_rows}")

    rec_cursor = 0
    decoded_cols: list[np.ndarray] = []
    prev_cols: list[np.ndarray] = []
    nnz_per_row: list[int] = []
    for _row_id in row_ids:
        tag = records_mv[rec_cursor]
        rec_cursor += 1
        if tag == 0:
            nnz = int(records_mv[rec_cursor])
            rec_cursor += 1
            cols = np.frombuffer(records_mv[rec_cursor : rec_cursor + nnz * 2], dtype=np.uint16).astype(np.int32, copy=True)
            rec_cursor += nnz * 2
        elif tag == 1:
            back = int(records_mv[rec_cursor])
            drop_count = int(records_mv[rec_cursor + 1])
            add_count = int(records_mv[rec_cursor + 2])
            rec_cursor += 3
            ref_cols = prev_cols[-back].copy()
            drops = np.frombuffer(records_mv[rec_cursor : rec_cursor + drop_count], dtype=np.uint8).astype(np.int32, copy=True)
            rec_cursor += drop_count
            add_cols = (
                np.frombuffer(records_mv[rec_cursor : rec_cursor + add_count * 2], dtype=np.uint16).astype(np.int32, copy=True)
                if add_count
                else np.empty((0,), dtype=np.int32)
            )
            rec_cursor += add_count * 2
            keep_mask = np.ones((ref_cols.size,), dtype=bool)
            if drop_count:
                keep_mask[drops] = False
            cols = np.sort(np.concatenate([ref_cols[keep_mask], add_cols]).astype(np.int32, copy=False))
        else:
            raise ValueError(f"unknown rowref tag {tag}")
        decoded_cols.append(cols.astype(np.int32, copy=False))
        nnz_per_row.append(int(cols.size))
        prev_cols.append(cols.astype(np.int32, copy=False))
    if rec_cursor != records_len:
        raise ValueError(f"rowref record cursor mismatch: {rec_cursor} != {records_len}")

    row_ptr = np.zeros((len(nnz_per_row) + 1,), dtype=np.uint32)
    if nnz_per_row:
        row_ptr[1:] = np.cumsum(np.asarray(nnz_per_row, dtype=np.uint32))
    words, consumed = _decode_value_words_mixed(values_mv, int(row_ptr[-1]), row_ptr, value_mode)
    if consumed != values_len:
        raise ValueError(f"rowref value cursor mismatch: {consumed} != {values_len}")
    vals = words.view(np.float16)
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    vals_list: list[np.ndarray] = []
    val_cursor = 0
    for row_id, cols in zip(row_ids, decoded_cols):
        n = int(cols.size)
        rows_list.append(np.full((n,), row_id, dtype=np.int32))
        cols_list.append(cols.astype(np.int16, copy=False))
        vals_list.append(vals[val_cursor : val_cursor + n].astype(np.float16, copy=False))
        val_cursor += n
    rows_cat = np.concatenate(rows_list).astype(np.int32, copy=False) if rows_list else np.empty((0,), dtype=np.int32)
    cols_cat = np.concatenate(cols_list).astype(np.int16, copy=False) if cols_list else np.empty((0,), dtype=np.int16)
    vals_cat = np.concatenate(vals_list).astype(np.float16, copy=False) if vals_list else np.empty((0,), dtype=np.float16)
    return {
        "rows": torch.from_numpy(rows_cat).to(torch.int32).contiguous(),
        "cols": torch.from_numpy(cols_cat).to(torch.int16).contiguous(),
        "vals": torch.from_numpy(vals_cat).to(torch.float16).contiguous(),
        "width": int(width),
    }


def pack_rowref_coord_sidecars(
    sidecars: dict[str, dict[str, Tensor]], value_mode_by_name: dict[str, str] | None = None, ref_window: int = 4
) -> bytes:
    if not sidecars:
        return b""
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        mode = value_mode_by_name.get(name, _mixed_value_mode_for_name(name)) if value_mode_by_name else _mixed_value_mode_for_name(name)
        name_b = name.encode("utf-8")
        blob = _pack_rowref_single_coord_sidecar(spec, mode, ref_window=ref_window)
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<I", len(blob)))
        parts.append(blob)
    return b"".join(parts)


def unpack_rowref_coord_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        (blob_len,) = struct.unpack_from("<I", mv, cursor)
        cursor += 4
        spec = _unpack_rowref_single_coord_sidecar(bytes(mv[cursor : cursor + blob_len]))
        spec.pop("width", None)
        out[name] = spec
        cursor += blob_len
    return out


def pack_shared_bitmap_coord_sidecars(sidecars: dict[str, dict[str, Tensor]]) -> bytes:
    if not sidecars:
        return b""
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        name_b = name.encode("utf-8")
        rows = spec["rows"].to(torch.int32)
        cols = spec["cols"].to(torch.int32)
        vals = spec["vals"].to(torch.float16)
        if "width" not in spec:
            raise ValueError(f"shared-bitmap coord sidecar for {name} missing width")
        width = int(spec["width"])
        row_ids, inverse = torch.unique(rows, sorted=True, return_inverse=True)
        col_values = cols.tolist()
        col_counts = Counter(int(col) for col in col_values)
        dict_cols = torch.tensor(
            [col for col, _ in sorted(col_counts.items(), key=lambda item: (-item[1], item[0]))],
            dtype=torch.int16,
        )
        dict_size = int(dict_cols.numel())
        mask_bytes = (dict_size + 7) // 8
        masks = torch.zeros((int(row_ids.numel()), mask_bytes), dtype=torch.uint8)
        dict_lookup = {int(col): idx for idx, col in enumerate(dict_cols.tolist())}
        vals_list: list[Tensor] = []
        for ridx in range(int(row_ids.numel())):
            mask = inverse == ridx
            row_cols = cols[mask]
            row_vals = vals[mask]
            order = torch.argsort(row_cols)
            row_cols = row_cols[order].tolist()
            row_vals = row_vals[order]
            dict_pos = [dict_lookup[int(col)] for col in row_cols]
            pos_order = sorted(range(len(dict_pos)), key=lambda idx: dict_pos[idx])
            sorted_pos = [dict_pos[idx] for idx in pos_order]
            if pos_order:
                row_vals = row_vals[torch.tensor(pos_order, dtype=torch.long)]
            for pos in sorted_pos:
                masks[ridx, pos >> 3] |= 1 << (pos & 7)
            vals_list.append(row_vals)
        vals_cat = torch.cat(vals_list).to(torch.float16).contiguous() if vals_list else torch.empty((0,), dtype=torch.float16)
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<IHIH", width, int(row_ids.numel()), int(vals_cat.numel()), dict_size))
        parts.append(row_ids.to(torch.int16).numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(dict_cols.numpy().astype(np.uint16, copy=False).tobytes())
        parts.append(masks.numpy().astype(np.uint8, copy=False).tobytes())
        parts.append(vals_cat.numpy().astype(np.float16, copy=False).tobytes())
    return b"".join(parts)


def pack_residualq_sidecars(sidecars: dict[str, dict[str, Tensor]]) -> Tensor:
    if not sidecars:
        return torch.empty((0,), dtype=torch.uint8)
    parts: list[bytes] = [struct.pack("<H", len(sidecars))]
    for name, spec in sidecars.items():
        name_b = name.encode("utf-8")
        row_ids = spec["row_ids"].to(torch.int16).numpy().astype(np.uint16, copy=False)
        row_ptr = spec["row_ptr"].to(torch.int32).numpy().astype(np.uint32, copy=False)
        cols = spec["cols"].to(torch.int16).numpy().astype(np.uint16, copy=False)
        dq_scale = spec["dq_scale"].to(torch.float16).numpy().astype(np.float16, copy=False)
        dq = spec["dq"].to(torch.int8).numpy().astype(np.int8, copy=False)
        width = int(spec["width"])
        parts.append(struct.pack("<H", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<IHI", width, int(row_ids.size), int(cols.size)))
        parts.append(row_ids.tobytes())
        parts.append(row_ptr.tobytes())
        parts.append(dq_scale.tobytes())
        parts.append(cols.tobytes())
        parts.append(dq.tobytes())
    blob = b"".join(parts)
    return torch.frombuffer(blob, dtype=torch.uint8).clone()


def unpack_coord_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        width, count = struct.unpack_from("<II", mv, cursor)
        cursor += 8
        idx_nbytes = count * 4
        val_nbytes = count * 2
        flat_idx = np.frombuffer(mv[cursor : cursor + idx_nbytes], dtype=np.uint32).astype(np.int64, copy=True)
        cursor += idx_nbytes
        vals = np.frombuffer(mv[cursor : cursor + val_nbytes], dtype=np.float16).copy()
        cursor += val_nbytes
        rows = torch.from_numpy(flat_idx // width).to(torch.int32)
        cols = torch.from_numpy(flat_idx % width).to(torch.int16)
        out[name] = {
            "rows": rows.contiguous(),
            "cols": cols.contiguous(),
            "vals": torch.from_numpy(vals).to(torch.float16).contiguous(),
        }
    return out


def unpack_grouped_coord_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        width, num_rows, nnz = struct.unpack_from("<IHI", mv, cursor)
        cursor += 10
        row_ids_nbytes = num_rows * 2
        row_ptr_nbytes = (num_rows + 1) * 4
        cols_nbytes = nnz * 2
        vals_nbytes = nnz * 2
        row_ids = np.frombuffer(mv[cursor : cursor + row_ids_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += row_ids_nbytes
        row_ptr = np.frombuffer(mv[cursor : cursor + row_ptr_nbytes], dtype=np.uint32).astype(np.int32, copy=True)
        cursor += row_ptr_nbytes
        cols = np.frombuffer(mv[cursor : cursor + cols_nbytes], dtype=np.uint16).astype(np.int16, copy=True)
        cursor += cols_nbytes
        vals = np.frombuffer(mv[cursor : cursor + vals_nbytes], dtype=np.float16).copy()
        cursor += vals_nbytes
        rows = np.repeat(row_ids, np.diff(row_ptr)).astype(np.int32, copy=False)
        out[name] = {
            "rows": torch.from_numpy(rows).to(torch.int32).contiguous(),
            "cols": torch.from_numpy(cols).to(torch.int16).contiguous(),
            "vals": torch.from_numpy(vals).to(torch.float16).contiguous(),
        }
    return out


def unpack_grouped_delta_coord_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        width, num_rows, nnz = struct.unpack_from("<IHI", mv, cursor)
        cursor += 10
        row_ids_nbytes = num_rows * 2
        row_ptr_nbytes = (num_rows + 1) * 4
        first_cols_nbytes = num_rows * 2
        row_ids = np.frombuffer(mv[cursor : cursor + row_ids_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += row_ids_nbytes
        row_ptr = np.frombuffer(mv[cursor : cursor + row_ptr_nbytes], dtype=np.uint32).astype(np.int32, copy=True)
        cursor += row_ptr_nbytes
        first_cols = np.frombuffer(mv[cursor : cursor + first_cols_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += first_cols_nbytes
        (gap_count,) = struct.unpack_from("<I", mv, cursor)
        cursor += 4
        gap_bytes = np.frombuffer(mv[cursor : cursor + gap_count], dtype=np.uint8).astype(np.int32, copy=True)
        cursor += gap_count
        (overflow_count,) = struct.unpack_from("<I", mv, cursor)
        cursor += 4
        overflow_pos_nbytes = overflow_count * 4
        overflow_vals_nbytes = overflow_count * 2
        overflow_pos = np.frombuffer(mv[cursor : cursor + overflow_pos_nbytes], dtype=np.uint32).astype(np.int32, copy=True)
        cursor += overflow_pos_nbytes
        overflow_vals = np.frombuffer(mv[cursor : cursor + overflow_vals_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += overflow_vals_nbytes
        vals_nbytes = nnz * 2
        vals = np.frombuffer(mv[cursor : cursor + vals_nbytes], dtype=np.float16).copy()
        cursor += vals_nbytes
        if overflow_count:
            gap_bytes[overflow_pos] = overflow_vals
        cols = np.empty((nnz,), dtype=np.int16)
        gap_cursor = 0
        for ridx in range(num_rows):
            start = row_ptr[ridx]
            end = row_ptr[ridx + 1]
            if start >= end:
                continue
            cols[start] = first_cols[ridx]
            for idx in range(start + 1, end):
                cols[idx] = cols[idx - 1] + gap_bytes[gap_cursor]
                gap_cursor += 1
        rows = np.repeat(row_ids, np.diff(row_ptr)).astype(np.int32, copy=False)
        out[name] = {
            "rows": torch.from_numpy(rows).to(torch.int32).contiguous(),
            "cols": torch.from_numpy(cols).to(torch.int16).contiguous(),
            "vals": torch.from_numpy(vals).to(torch.float16).contiguous(),
        }
    return out


def unpack_grouped_delta_byteplane_coord_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        width, num_rows, nnz = struct.unpack_from("<IHI", mv, cursor)
        cursor += 10
        (mode_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        encoded_mode = bytes(mv[cursor : cursor + mode_len]).decode("utf-8")
        cursor += mode_len
        row_ids_nbytes = num_rows * 2
        row_ptr_nbytes = (num_rows + 1) * 4
        first_cols_nbytes = num_rows * 2
        row_ids = np.frombuffer(mv[cursor : cursor + row_ids_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += row_ids_nbytes
        row_ptr = np.frombuffer(mv[cursor : cursor + row_ptr_nbytes], dtype=np.uint32).astype(np.int32, copy=True)
        cursor += row_ptr_nbytes
        first_cols = np.frombuffer(mv[cursor : cursor + first_cols_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += first_cols_nbytes
        (gap_count,) = struct.unpack_from("<I", mv, cursor)
        cursor += 4
        gap_bytes = np.frombuffer(mv[cursor : cursor + gap_count], dtype=np.uint8).astype(np.int32, copy=True)
        cursor += gap_count
        (overflow_count,) = struct.unpack_from("<I", mv, cursor)
        cursor += 4
        overflow_pos_nbytes = overflow_count * 4
        overflow_vals_nbytes = overflow_count * 2
        overflow_pos = np.frombuffer(mv[cursor : cursor + overflow_pos_nbytes], dtype=np.uint32).astype(np.int32, copy=True)
        cursor += overflow_pos_nbytes
        overflow_vals = np.frombuffer(mv[cursor : cursor + overflow_vals_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += overflow_vals_nbytes
        if overflow_count:
            gap_bytes[overflow_pos] = overflow_vals
        words, used = _decode_value_words_mixed(mv[cursor:], nnz, row_ptr, encoded_mode)
        cursor += used
        cols = np.empty((nnz,), dtype=np.int16)
        gap_cursor = 0
        for ridx in range(num_rows):
            start = row_ptr[ridx]
            end = row_ptr[ridx + 1]
            if start >= end:
                continue
            cols[start] = first_cols[ridx]
            for idx in range(start + 1, end):
                cols[idx] = cols[idx - 1] + gap_bytes[gap_cursor]
                gap_cursor += 1
        rows = np.repeat(row_ids, np.diff(row_ptr)).astype(np.int32, copy=False)
        out[name] = {
            "rows": torch.from_numpy(rows).to(torch.int32).contiguous(),
            "cols": torch.from_numpy(cols).to(torch.int16).contiguous(),
            "vals": torch.from_numpy(words.view(np.float16)).to(torch.float16).contiguous(),
        }
    return out


def unpack_residualq_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        width, num_rows, nnz = struct.unpack_from("<IHI", mv, cursor)
        cursor += 10
        row_ids_nbytes = num_rows * 2
        row_ptr_nbytes = (num_rows + 1) * 4
        dq_scale_nbytes = num_rows * 2
        cols_nbytes = nnz * 2
        dq_nbytes = nnz
        row_ids = np.frombuffer(mv[cursor : cursor + row_ids_nbytes], dtype=np.uint16).astype(np.int16, copy=True)
        cursor += row_ids_nbytes
        row_ptr = np.frombuffer(mv[cursor : cursor + row_ptr_nbytes], dtype=np.uint32).astype(np.int32, copy=True)
        cursor += row_ptr_nbytes
        dq_scale = np.frombuffer(mv[cursor : cursor + dq_scale_nbytes], dtype=np.float16).copy()
        cursor += dq_scale_nbytes
        cols = np.frombuffer(mv[cursor : cursor + cols_nbytes], dtype=np.uint16).astype(np.int16, copy=True)
        cursor += cols_nbytes
        dq = np.frombuffer(mv[cursor : cursor + dq_nbytes], dtype=np.int8).copy()
        cursor += dq_nbytes
        out[name] = {
            "row_ids": torch.from_numpy(row_ids).to(torch.int16).contiguous(),
            "row_ptr": torch.from_numpy(row_ptr).to(torch.int32).contiguous(),
            "cols": torch.from_numpy(cols).to(torch.int16).contiguous(),
            "dq_scale": torch.from_numpy(dq_scale).to(torch.float16).contiguous(),
            "dq": torch.from_numpy(dq).to(torch.int8).contiguous(),
            "width": int(width),
        }
    return out


def unpack_shared_bitmap_coord_sidecars(blob: bytes) -> dict[str, dict[str, Tensor]]:
    if not blob:
        return {}
    mv = memoryview(blob)
    cursor = 0
    (num_tensors,) = struct.unpack_from("<H", mv, cursor)
    cursor += 2
    out: dict[str, dict[str, Tensor]] = {}
    for _ in range(num_tensors):
        (name_len,) = struct.unpack_from("<H", mv, cursor)
        cursor += 2
        name = bytes(mv[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        width, num_rows, nnz, dict_size = struct.unpack_from("<IHIH", mv, cursor)
        cursor += 12
        row_ids_nbytes = num_rows * 2
        dict_cols_nbytes = dict_size * 2
        mask_bytes = (dict_size + 7) // 8
        masks_nbytes = num_rows * mask_bytes
        vals_nbytes = nnz * 2
        row_ids = np.frombuffer(mv[cursor : cursor + row_ids_nbytes], dtype=np.uint16).astype(np.int32, copy=True)
        cursor += row_ids_nbytes
        dict_cols = np.frombuffer(mv[cursor : cursor + dict_cols_nbytes], dtype=np.uint16).astype(np.int16, copy=True)
        cursor += dict_cols_nbytes
        masks = np.frombuffer(mv[cursor : cursor + masks_nbytes], dtype=np.uint8).copy().reshape(num_rows, mask_bytes)
        cursor += masks_nbytes
        vals = np.frombuffer(mv[cursor : cursor + vals_nbytes], dtype=np.float16).copy()
        cursor += vals_nbytes
        rows_list: list[np.ndarray] = []
        cols_list: list[np.ndarray] = []
        val_cursor = 0
        for ridx in range(num_rows):
            mask_row = masks[ridx]
            pos_list: list[int] = []
            for byte_idx, byte_val in enumerate(mask_row.tolist()):
                if byte_val == 0:
                    continue
                for bit in range(8):
                    pos = (byte_idx << 3) + bit
                    if pos >= dict_size:
                        break
                    if byte_val & (1 << bit):
                        pos_list.append(pos)
            count = len(pos_list)
            if count:
                rows_list.append(np.full((count,), row_ids[ridx], dtype=np.int32))
                cols_list.append(dict_cols[np.asarray(pos_list, dtype=np.int64)].astype(np.int16, copy=False))
                val_cursor += count
        if val_cursor != nnz:
            raise ValueError(f"shared-bitmap coord sidecar for {name} decoded nnz={val_cursor}, expected {nnz}")
        rows = np.concatenate(rows_list).astype(np.int32, copy=False) if rows_list else np.empty((0,), dtype=np.int32)
        cols = np.concatenate(cols_list).astype(np.int16, copy=False) if cols_list else np.empty((0,), dtype=np.int16)
        out[name] = {
            "rows": torch.from_numpy(rows).to(torch.int32).contiguous(),
            "cols": torch.from_numpy(cols).to(torch.int16).contiguous(),
            "vals": torch.from_numpy(vals).to(torch.float16).contiguous(),
        }
    return out


def extract_coord_sidecars(payload: dict[str, object]) -> dict[str, dict[str, Tensor]]:
    out: dict[str, dict[str, Tensor]] = {}

    def merge(sidecars: dict[str, dict[str, Tensor]]) -> None:
        for name, spec in sidecars.items():
            if name in out:
                raise ValueError(f"Duplicate coord sidecar tensor in payload: {name}")
            out[name] = spec

    if "s" in payload and isinstance(payload["s"], dict):
        merge(payload["s"])  # backward-compatible path for older local artifacts
    if "sb" in payload:
        blob = payload["sb"]
        if isinstance(blob, (bytes, bytearray)):
            merge(unpack_coord_sidecars(bytes(blob)))
        elif isinstance(blob, torch.Tensor):
            merge(unpack_coord_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes())))
    if "sbx" in payload:
        blob = payload["sbx"]
        if isinstance(blob, (bytes, bytearray)):
            merge(unpack_grouped_coord_sidecars(bytes(blob)))
        elif isinstance(blob, torch.Tensor):
            merge(unpack_grouped_coord_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes())))
    if "sbe" in payload:
        blob = payload["sbe"]
        if isinstance(blob, (bytes, bytearray)):
            merge(unpack_grouped_delta_coord_sidecars(bytes(blob)))
        elif isinstance(blob, torch.Tensor):
            merge(unpack_grouped_delta_coord_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes())))
    if "sbev" in payload:
        blob = payload["sbev"]
        if isinstance(blob, (bytes, bytearray)):
            merge(unpack_grouped_delta_byteplane_coord_sidecars(bytes(blob)))
        elif isinstance(blob, torch.Tensor):
            merge(unpack_grouped_delta_byteplane_coord_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes())))
    if "sbbm" in payload:
        blob = payload["sbbm"]
        if isinstance(blob, (bytes, bytearray)):
            merge(unpack_shared_bitmap_coord_sidecars(bytes(blob)))
        elif isinstance(blob, torch.Tensor):
            merge(unpack_shared_bitmap_coord_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes())))
    if "sbrf" in payload:
        blob = payload["sbrf"]
        if isinstance(blob, (bytes, bytearray)):
            merge(unpack_rowref_coord_sidecars(bytes(blob)))
        elif isinstance(blob, torch.Tensor):
            merge(unpack_rowref_coord_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes())))
    return out


def extract_residualq_sidecars(payload: dict[str, object]) -> dict[str, dict[str, Tensor]]:
    blob = payload.get("sbrq")
    if isinstance(blob, (bytes, bytearray)):
        return unpack_residualq_sidecars(bytes(blob))
    if isinstance(blob, torch.Tensor):
        return unpack_residualq_sidecars(bytes(blob.to(torch.uint8).cpu().numpy().tobytes()))
    return {}


def load_init_state_dict(init_path: str, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    path = Path(init_path)
    if not path.exists():
        raise FileNotFoundError(f"INIT_MODEL_PATH does not exist: {path}")
    if path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported .pt payload type: {type(obj)!r}")
        return obj
    if path.suffix == ".ptz":
        raw = path.read_bytes()
        blob = zstandard.ZstdDecompressor().decompress(raw) if _COMPRESSOR == "zstd" else zlib.decompress(raw)
        obj = torch.load(io.BytesIO(blob), map_location="cpu")
        if not isinstance(obj, dict) or "w" not in obj or "m" not in obj:
            raise TypeError("Unsupported .ptz payload; expected quantized {'w','m'} dict")
        out = dequantize_mixed_int6(obj["w"], obj["m"], template_sd)
        apply_coord_sidecars_in_place(out, extract_coord_sidecars(obj))
        apply_residualq_sidecars_in_place(out, extract_residualq_sidecars(obj))
        return out
    raise ValueError(f"Unsupported INIT_MODEL_PATH suffix for {path}")


def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        mlp_activation=args.mlp_activation,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.init_model_path:
        init_state = load_init_state_dict(args.init_model_path, base_model.state_dict())
        missing, unexpected = base_model.load_state_dict(init_state, strict=args.init_model_strict)
        log0(
            f"init_model:path={args.init_model_path} strict={int(args.init_model_strict)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            log0(f"init_model_missing:{sorted(missing)}")
        if unexpected:
            log0(f"init_model_unexpected:{sorted(unexpected)}")
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_head = None
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(f"mlp_activation:{args.mlp_activation}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
    gradquant_cats = {x.strip() for x in args.gradquant_categories.split(",") if x.strip()}
    gradquant_stats = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name, p in base_model.named_parameters()
        if _is_gradquant_candidate(name, p.detach(), gradquant_cats, args.gradquant_min_numel)
    }
    if args.gradquant_enabled:
        log0(
            f"gradquant:enabled start_scale:{args.gradquant_start_scale:.3f} "
            f"ema:{args.gradquant_ema:.3f} top_frac:{args.gradquant_top_frac:.3f} "
            f"bottom_frac:{args.gradquant_bottom_frac:.3f} tracked:{len(gradquant_stats)} "
            f"cats:{sorted(gradquant_cats)} min_numel:{args.gradquant_min_numel}"
        )
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        if args.gradquant_enabled and gradquant_stats and scale <= args.gradquant_start_scale:
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    stat = gradquant_stats.get(name)
                    if stat is None or param.grad is None:
                        continue
                    stat.mul_(args.gradquant_ema).add_(
                        param.grad.detach().float().pow(2).mean(),
                        alpha=1.0 - args.gradquant_ema,
                    )
        optimizer_muon.launch_reduce_scatters()
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        zero_grad_all()
        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # Apply EMA weights (better than SWA alone per PR#401)
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    gradquant_scores = (
        {name: float(stat.detach().cpu().item()) for name, stat in gradquant_stats.items()}
        if args.gradquant_enabled
        else {}
    )
    bits_by_name = (
        build_gradquant_bits_map(
            sd_cpu,
            gradquant_scores,
            gradquant_cats,
            args.gradquant_min_numel,
            args.gradquant_top_frac,
            args.gradquant_bottom_frac,
        )
        if args.gradquant_enabled
        else {}
    )
    export_profile_bits = build_export_profile_bits_map(sd_cpu, args.export_bits_profile, args.gradquant_min_numel)
    if export_profile_bits:
        bits_by_name.update(export_profile_bits)
    coord_sidecars = load_coord_sidecars(args.export_coord_sidecar_path) if args.export_coord_sidecar_path else {}
    prepacked_sbev = load_prepacked_blob(args.export_prepacked_sbev_path) if args.export_prepacked_sbev_path else b""
    residualq_sidecars = (
        load_residualq_sidecars(args.export_residualq_sidecar_path)
        if args.export_residualq_sidecar_path
        else {}
    )
    if master_process and bits_by_name:
        bit_summary = summarize_gradquant_bits(bits_by_name, sd_cpu)
        if args.export_bits_profile:
            log0(f"export_bits_profile:{args.export_bits_profile} tensors:{len(export_profile_bits)}")
        for bits in sorted(bit_summary):
            slot = bit_summary[bits]
            log0(f"gradquant_bits:{bits} tensors:{slot['tensors']} params:{slot['params']}")
        top_sens = sorted(bits_by_name.items(), key=lambda kv: gradquant_scores.get(kv[0], 0.0), reverse=True)[:8]
        bottom_sens = sorted(bits_by_name.items(), key=lambda kv: gradquant_scores.get(kv[0], 0.0))[:8]
        log0(f"gradquant_top:{[(n, b, round(gradquant_scores.get(n, 0.0), 6)) for n, b in top_sens]}")
        log0(f"gradquant_bottom:{[(n, b, round(gradquant_scores.get(n, 0.0), 6)) for n, b in bottom_sens]}")
    if master_process and coord_sidecars:
        nnz = sum(int(spec["vals"].numel()) for spec in coord_sidecars.values())
        log0(
            f"export_coord_sidecar_path:{args.export_coord_sidecar_path} "
            f"packing:{args.export_coord_sidecar_packing} tensors:{len(coord_sidecars)} nnz:{nnz}"
        )
    if master_process and prepacked_sbev:
        log0(
            f"export_prepacked_sbev_path:{args.export_prepacked_sbev_path} "
            f"bytes:{len(prepacked_sbev)}"
        )
    if master_process and residualq_sidecars:
        nnz = sum(int(spec["dq"].numel()) for spec in residualq_sidecars.values())
        log0(
            f"export_residualq_sidecar_path:{args.export_residualq_sidecar_path} "
            f"tensors:{len(residualq_sidecars)} nnz:{nnz}"
        )
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"}, bits_by_name=bits_by_name)
    quant_buf = io.BytesIO()
    quant_payload = {"w": quant_result, "m": quant_meta}
    if coord_sidecars and prepacked_sbev:
        raise ValueError("Set at most one of EXPORT_COORD_SIDECAR_PATH or EXPORT_PREPACKED_SBEV_PATH")
    if prepacked_sbev:
        quant_payload["sbev"] = prepacked_sbev
    elif coord_sidecars:
        for name, spec in coord_sidecars.items():
            spec["width"] = int(sd_cpu[name].shape[1])
        if args.export_coord_sidecar_packing == "b10_rowref_mixed":
            rowref_sidecars = {name: spec for name, spec in coord_sidecars.items() if name in _ROWREF_SIDECAR_NAMES}
            remaining = {name: spec for name, spec in coord_sidecars.items() if name not in _ROWREF_SIDECAR_NAMES}
            if rowref_sidecars:
                quant_payload["sbrf"] = pack_rowref_coord_sidecars(
                    rowref_sidecars,
                    value_mode_by_name={name: _mixed_value_mode_for_name(name) for name in rowref_sidecars.keys()},
                    ref_window=int(os.environ.get("EXPORT_ROWREF_WINDOW", "4")),
                )
            if remaining:
                quant_payload["sbev"] = pack_grouped_delta_byteplane_coord_sidecars(
                    remaining,
                    value_mode_by_name={name: _mixed_value_mode_for_name(name) for name in remaining.keys()},
                )
        elif args.export_coord_sidecar_packing == "b10_mixed_valuecode":
            quant_payload["sbev"] = pack_grouped_delta_byteplane_coord_sidecars(
                coord_sidecars,
                value_mode_by_name={name: _mixed_value_mode_for_name(name) for name in coord_sidecars.keys()},
            )
        elif args.export_coord_sidecar_packing == "grouped":
            quant_payload["sbx"] = pack_grouped_coord_sidecars(coord_sidecars)
        elif args.export_coord_sidecar_packing == "grouped_delta":
            quant_payload["sbe"] = pack_grouped_delta_coord_sidecars(coord_sidecars)
        elif args.export_coord_sidecar_packing == "grouped_delta_byteplane":
            quant_payload["sbev"] = pack_grouped_delta_byteplane_coord_sidecars(coord_sidecars)
        elif args.export_coord_sidecar_packing == "shared_bitmap":
            quant_payload["sbbm"] = pack_shared_bitmap_coord_sidecars(coord_sidecars)
        else:
            quant_payload["sb"] = pack_coord_sidecars(coord_sidecars)
    if residualq_sidecars:
        for name, spec in residualq_sidecars.items():
            spec["width"] = int(sd_cpu[name].shape[1])
        quant_payload["sbrq"] = pack_residualq_sidecars(residualq_sidecars)
    torch.save(quant_payload, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    apply_coord_sidecars_in_place(deq_state, extract_coord_sidecars(quant_state))
    apply_residualq_sidecars_in_place(deq_state, extract_residualq_sidecars(quant_state))
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,  # must match training model
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        mlp_activation=args.mlp_activation,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log0=log0,
        )
        torch.cuda.synchronize()
        log0(
            f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
