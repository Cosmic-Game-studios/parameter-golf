"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from quant_reconstruction import (
    apply_lowrank_residual,
    compute_lowrank_residual,
    dequantize_codebook_int4_blockwise,
    quantize_codebook_int4_blockwise,
    submission_code_text,
)
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
def csv_patterns(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(pattern for pattern in os.environ.get(name, default).split(",") if pattern and pattern != "__never__")
def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
def should_enable_ddp(world_size: int) -> bool:
    return world_size > 1 and "RANK" in os.environ and "WORLD_SIZE" in os.environ
# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    init_model_path = os.environ.get("INIT_MODEL_PATH", "")
    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    val_at_step_zero = bool(int(os.environ.get("VAL_AT_STEP_ZERO", "0")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    wallclock_sync_every = int(os.environ.get("WALLCLOCK_SYNC_EVERY", 1))
    lr_schedule = os.environ.get("LR_SCHEDULE", "warmdown").strip().lower()
    lr_warmup_iters = int(os.environ.get("LR_WARMUP_ITERS", 0))
    min_lr_scale = float(os.environ.get("MIN_LR_SCALE", 0.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", os.environ.get("NUM_LAYERS", 9)))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.0))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 0))
    ema_update_every = int(os.environ.get("EMA_UPDATE_EVERY", 1))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.0))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", os.environ.get("WEIGHT_DECAY", 0.0)))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", os.environ.get("WEIGHT_DECAY", 0.0)))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    ddp_static_graph = bool(int(os.environ.get("DDP_STATIC_GRAPH", "0")))
    ddp_gradient_as_bucket_view = bool(int(os.environ.get("DDP_GRADIENT_AS_BUCKET_VIEW", "0")))
    disable_compile = env_flag("DISABLE_COMPILE") or env_flag("TORCHDYNAMO_DISABLE")
# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


class ExponentialMovingAverage:
    def __init__(self, named_params: list[tuple[str, Tensor]], decay: float, start_step: int = 0, update_every: int = 1):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA_DECAY must be in [0, 1), got {decay}")
        if update_every <= 0:
            raise ValueError(f"EMA_UPDATE_EVERY must be positive, got {update_every}")
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every
        self.active = False
        self.shadow = {name: param.detach().float().clone() for name, param in named_params}

    @torch.no_grad()
    def update(self, named_params: list[tuple[str, Tensor]], step: int) -> None:
        if step < self.start_step or step % self.update_every != 0:
            return
        if not self.active:
            for name, param in named_params:
                self.shadow[name].copy_(param.detach().float())
            self.active = True
            return
        blend = 1.0 - self.decay
        for name, param in named_params:
            self.shadow[name].lerp_(param.detach().float(), blend)

    def state_dict(self, base_state: dict[str, Tensor]) -> dict[str, Tensor]:
        out = {name: tensor.detach().cpu().clone() for name, tensor in base_state.items()}
        for name, shadow in self.shadow.items():
            out[name] = shadow.detach().to(device="cpu", dtype=out[name].dtype).clone()
        return out


def clone_state_dict_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

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
def maybe_pin_cpu_tensor(t: Tensor) -> Tensor:
    return t.pin_memory() if t.device.type == "cpu" and torch.cuda.is_available() and not t.is_pinned() else t
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return maybe_pin_cpu_tensor(tokens[: usable + 1].contiguous())
def sliding_window_segments(total_tokens: int, seq_len: int, stride: int) -> list[tuple[int, int, int]]:
    if stride <= 0:
        raise ValueError(f"EVAL_STRIDE must be positive, got {stride}")
    if total_tokens <= 0:
        return []
    segments = [(0, 0, min(seq_len, total_tokens))]
    for score_start in range(seq_len, total_tokens, stride):
        score_end = min(score_start + stride, total_tokens)
        window_start = max(score_end - seq_len, 0)
        local_start = score_start - window_start
        local_end = score_end - window_start
        segments.append((window_start, local_start, local_end))
    return segments


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
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    if args.eval_stride > 0:
        return eval_val_sliding(
            args,
            model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            args.eval_stride,
            args.eval_batch_seqs,
        )
    local_batch_tokens = args.val_batch_size // world_size
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
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
def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    segments = sliding_window_segments(total_tokens, seq_len, stride)
    total_segments = len(segments)
    my_start = (total_segments * rank) // world_size
    my_end = (total_segments * (rank + 1)) // world_size
    my_segments = segments[my_start:my_end]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(my_segments), batch_seqs):
            batch_segments = my_segments[batch_start:batch_start + batch_seqs]
            bsz = len(batch_segments)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            ranges: list[tuple[int, int]] = []
            for i, (ws, local_start, local_end) in enumerate(batch_segments):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                ranges.append((local_start, local_end))
                chunk = val_tokens[ws:end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x_batch).float()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, (_, local_start, local_end) in enumerate(batch_segments):
                scored_nll = nll[i, local_start:local_end].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(local_end - local_start)
                tgt = y_batch[i, local_start:local_end]
                prev = x_batch[i, local_start:local_end]
                token_bytes = base_bytes_lut[tgt].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += token_bytes.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = csv_patterns(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = csv_patterns("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS))
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS = csv_patterns("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS")
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
QUANT_FORMAT = os.environ.get("QUANT_FORMAT", "int8_clean_per_row_v1").strip().lower()
INT4_NAME_PATTERNS = csv_patterns("INT4_NAME_PATTERNS")
INT4_CODEBOOK_NAME = os.environ.get("INT4_CODEBOOK_NAME", "normal16").strip().lower()
INT4_BLOCK_SIZE = int(os.environ.get("INT4_BLOCK_SIZE", 64))
INT4_CLIP_PERCENTILE = float(os.environ.get("INT4_CLIP_PERCENTILE", 99.9))
INT4_CLIP_Q = INT4_CLIP_PERCENTILE / 100.0
LOWRANK_ERROR_NAME_PATTERNS = csv_patterns("LOWRANK_ERROR_NAME_PATTERNS")
LOWRANK_ERROR_RANK = int(os.environ.get("LOWRANK_ERROR_RANK", 0))
TRAIN_COMPRESSION_AWARE_WEIGHT = float(os.environ.get("TRAIN_COMPRESSION_AWARE_WEIGHT", 0.0))
TRAIN_COMPRESSION_AWARE_NAME_PATTERNS = csv_patterns("TRAIN_COMPRESSION_AWARE_NAME_PATTERNS")
TRAIN_COMPRESSION_AWARE_BLOCK_SIZE = int(
    os.environ.get("TRAIN_COMPRESSION_AWARE_BLOCK_SIZE", os.environ.get("INT4_BLOCK_SIZE", 64))
)
TRAIN_INT8_AWARE_WEIGHT = float(os.environ.get("TRAIN_INT8_AWARE_WEIGHT", 0.0))
TRAIN_INT8_AWARE_NAME_PATTERNS = csv_patterns("TRAIN_INT8_AWARE_NAME_PATTERNS")
TRAIN_QER_AWARE_WEIGHT = float(os.environ.get("TRAIN_QER_AWARE_WEIGHT", 0.0))
TRAIN_QER_AWARE_NAME_PATTERNS = csv_patterns("TRAIN_QER_AWARE_NAME_PATTERNS")
TRAIN_QER_AWARE_RANK = int(os.environ.get("TRAIN_QER_AWARE_RANK", 0))
TRAIN_QER_AWARE_QUANT_FORMAT = os.environ.get("TRAIN_QER_AWARE_QUANT_FORMAT", "int8_clean_per_row_v1").strip().lower()
TRAIN_QER_AWARE_BLOCK_SIZE = int(os.environ.get("TRAIN_QER_AWARE_BLOCK_SIZE", os.environ.get("INT4_BLOCK_SIZE", 64)))
TRAIN_QER_AWARE_CODEBOOK_NAME = os.environ.get(
    "TRAIN_QER_AWARE_CODEBOOK_NAME",
    os.environ.get("INT4_CODEBOOK_NAME", "normal16"),
).strip().lower()
TRAIN_GRAD_ONLY_NAME_PATTERNS = csv_patterns("TRAIN_GRAD_ONLY_NAME_PATTERNS")
TRAIN_GRAD_SKIP_NAME_PATTERNS = csv_patterns("TRAIN_GRAD_SKIP_NAME_PATTERNS")
TRAIN_QAT_NAME_PATTERNS = csv_patterns("TRAIN_QAT_NAME_PATTERNS")
TRAIN_QAT_BLOCK_SIZE = int(os.environ.get("TRAIN_QAT_BLOCK_SIZE", os.environ.get("INT4_BLOCK_SIZE", 64)))
LFQAT_KL_WEIGHT, LFQAT_FISHER_WEIGHT, LFQAT_TEMPERATURE = (
    float(os.environ.get("LFQAT_KL_WEIGHT", 0.0)),
    float(os.environ.get("LFQAT_FISHER_WEIGHT", 0.0)),
    float(os.environ.get("LFQAT_TEMPERATURE", 2.0)),
)
LFQAT_FISHER_DECAY, LFQAT_START_STEP, LFQAT_FULL_STEP = float(os.environ.get("LFQAT_FISHER_DECAY", 0.95)), int(os.environ.get("LFQAT_START_STEP", 0)), int(os.environ.get("LFQAT_FULL_STEP", 0))
LFQAT_MIN_PROB, LFQAT_MAX_PROB = float(os.environ.get("LFQAT_MIN_PROB", 1.0)), float(os.environ.get("LFQAT_MAX_PROB", 1.0))
LFQAT_RUNTIME = {"prob": 1.0, "disable_qat": False}
LFQAT_FISHER_EMA: dict[str, float] = {}
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def should_keep_float_tensor(name: str) -> bool:
    return any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS) or any(
        pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS
    )
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS):
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def should_use_int4_tensor(name: str, t: Tensor) -> bool:
    return (
        QUANT_FORMAT in {"mixed_int4_int8_packed_v2", "mixed_codebook_int4_int8_packed_v1"}
        and t.ndim == 2
        and any(pattern in name for pattern in INT4_NAME_PATTERNS)
    )
def should_use_lowrank_error(name: str, t: Tensor) -> bool:
    return LOWRANK_ERROR_RANK > 0 and t.ndim == 2 and any(pattern in name for pattern in LOWRANK_ERROR_NAME_PATTERNS)
def pack_int4_values(values: Tensor) -> Tensor:
    flat = values.to(dtype=torch.int8).reshape(-1).contiguous()
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros((1,), dtype=torch.int8)))
    unsigned = (flat + 8).to(dtype=torch.uint8)
    packed = unsigned[0::2] | (unsigned[1::2] << 4)
    return packed.contiguous()
def unpack_int4_values(packed: Tensor, count: int) -> Tensor:
    packed_u8 = packed.to(dtype=torch.uint8).reshape(-1).contiguous()
    values = torch.empty((packed_u8.numel() * 2,), dtype=torch.int8)
    values[0::2] = (packed_u8 & 0x0F).to(dtype=torch.int8) - 8
    values[1::2] = (packed_u8 >> 4).to(dtype=torch.int8) - 8
    return values[:count]
def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_float_tensor_int4_blockwise(t: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
    t32 = t.float()
    if t32.ndim != 2:
        raise ValueError(f"int4 blockwise quantization only supports 2D tensors, got shape={tuple(t32.shape)}")
    if INT4_BLOCK_SIZE <= 0:
        raise ValueError(f"INT4_BLOCK_SIZE must be positive, got {INT4_BLOCK_SIZE}")

    rows, cols = t32.shape
    blocks_per_row = (cols + INT4_BLOCK_SIZE - 1) // INT4_BLOCK_SIZE
    padded_cols = blocks_per_row * INT4_BLOCK_SIZE
    if padded_cols != cols:
        padded = torch.zeros((rows, padded_cols), dtype=torch.float32)
        padded[:, :cols] = t32
    else:
        padded = t32
    reshaped = padded.reshape(rows, blocks_per_row, INT4_BLOCK_SIZE)
    clip_abs = (
        torch.quantile(reshaped.abs(), INT4_CLIP_Q, dim=2)
        if reshaped.numel()
        else torch.empty((rows, blocks_per_row), dtype=torch.float32)
    )
    scale = (clip_abs / 7.0).clamp_min(1.0 / 7.0)
    clipped = torch.maximum(torch.minimum(reshaped, clip_abs[..., None]), -clip_abs[..., None])
    q = torch.clamp(torch.round(clipped / scale[..., None]), -7, 7).to(torch.int8).contiguous()
    meta = {
        "scheme": "per_row_block_int4",
        "axis": 0,
        "shape": [int(rows), int(cols)],
        "block_size": INT4_BLOCK_SIZE,
    }
    return (
        pack_int4_values(q),
        scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(),
        meta,
    )
def quantize_float_tensor_codebook_int4_blockwise(t: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
    signed, scale, meta = quantize_codebook_int4_blockwise(
        t.float().numpy(),
        INT4_BLOCK_SIZE,
        clip_q=INT4_CLIP_Q,
        codebook_name=INT4_CODEBOOK_NAME,
    )
    return (
        pack_int4_values(torch.from_numpy(signed)),
        torch.from_numpy(scale).to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(),
        meta,
    )
def dequantize_quantized_tensor(q: Tensor, s: Tensor, meta: dict[str, object], dtype: torch.dtype = torch.float32) -> Tensor:
    if meta.get("scheme") == "per_row_block_int4":
        rows, cols = (int(x) for x in meta["shape"])
        block_size = int(meta["block_size"])
        blocks_per_row = (cols + block_size - 1) // block_size
        q_int = unpack_int4_values(q, rows * blocks_per_row * block_size).reshape(rows, blocks_per_row, block_size)
        out = (q_int.float() * s.to(dtype=torch.float32).reshape(rows, blocks_per_row, 1)).reshape(rows, blocks_per_row * block_size)
        return out[:, :cols].to(dtype=dtype).contiguous()
    if meta.get("scheme") == "per_row_block_codebook_int4":
        rows, cols = (int(x) for x in meta["shape"])
        block_size = int(meta["block_size"])
        blocks_per_row = (cols + block_size - 1) // block_size
        signed = unpack_int4_values(q, rows * blocks_per_row * block_size).numpy()
        out = dequantize_codebook_int4_blockwise(signed, s.to(dtype=torch.float32).numpy(), meta)
        return torch.from_numpy(out).to(dtype=dtype).contiguous()
    if meta.get("scheme") == "per_row" or s.ndim > 0:
        return (q.float() * s.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    return (q.float() * float(s.item())).to(dtype=dtype).contiguous()
def should_train_compression_align(name: str, t: Tensor) -> bool:
    return (
        TRAIN_COMPRESSION_AWARE_WEIGHT > 0.0
        and t.ndim == 2
        and any(pattern in name for pattern in TRAIN_COMPRESSION_AWARE_NAME_PATTERNS)
    )
def should_train_int8_align(name: str, t: Tensor) -> bool:
    return TRAIN_INT8_AWARE_WEIGHT > 0.0 and any(pattern in name for pattern in TRAIN_INT8_AWARE_NAME_PATTERNS)
def should_train_qer_align(name: str, t: Tensor) -> bool:
    return (
        TRAIN_QER_AWARE_WEIGHT > 0.0
        and TRAIN_QER_AWARE_RANK > 0
        and t.ndim == 2
        and any(pattern in name for pattern in TRAIN_QER_AWARE_NAME_PATTERNS)
    )
def compression_aware_int4_target(t: Tensor, block_size: int) -> Tensor:
    if t.ndim != 2:
        raise ValueError(f"compression-aware int4 target expects 2D tensors, got shape={tuple(t.shape)}")
    if block_size <= 0:
        raise ValueError(f"TRAIN_COMPRESSION_AWARE_BLOCK_SIZE must be positive, got {block_size}")
    t32 = t.detach().float()
    rows, cols = t32.shape
    blocks_per_row = (cols + block_size - 1) // block_size
    padded_cols = blocks_per_row * block_size
    if padded_cols != cols:
        padded = torch.zeros((rows, padded_cols), dtype=torch.float32, device=t32.device)
        padded[:, :cols] = t32
    else:
        padded = t32
    reshaped = padded.reshape(rows, blocks_per_row, block_size)
    clip_abs = torch.amax(reshaped.abs(), dim=2).clamp_min(1.0 / 7.0)
    scale = clip_abs / 7.0
    q = torch.clamp(torch.round(reshaped / scale[..., None]), -7, 7)
    return (q * scale[..., None]).reshape(rows, padded_cols)[:, :cols].to(dtype=t.dtype, device=t.device).contiguous()
def int8_aware_target(t: Tensor) -> Tensor:
    q, s = quantize_float_tensor_int8(t.detach())
    meta = {"scheme": "per_row", "axis": 0} if s.ndim > 0 else {}
    return dequantize_quantized_tensor(q, s, meta, dtype=t.dtype).to(device=t.device).contiguous()
def qer_aware_target(t: Tensor) -> Tensor:
    with torch.no_grad():
        t_cpu = t.detach().to("cpu").contiguous()
        if TRAIN_QER_AWARE_QUANT_FORMAT == "mixed_codebook_int4_int8_packed_v1":
            signed, scale, meta = quantize_codebook_int4_blockwise(
                t_cpu.float().numpy(),
                TRAIN_QER_AWARE_BLOCK_SIZE,
                clip_q=INT4_CLIP_Q,
                codebook_name=TRAIN_QER_AWARE_CODEBOOK_NAME,
            )
            q = pack_int4_values(torch.from_numpy(signed))
            s = torch.from_numpy(scale).to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
        elif TRAIN_QER_AWARE_QUANT_FORMAT == "mixed_int4_int8_packed_v2":
            target = compression_aware_int4_target(t_cpu, TRAIN_QER_AWARE_BLOCK_SIZE)
            q, s, meta = quantize_float_tensor_int4_blockwise(target)
        else:
            q, s = quantize_float_tensor_int8(t_cpu)
            meta = {"scheme": "per_row", "axis": 0} if s.ndim > 0 else {}
        base = dequantize_quantized_tensor(q, s, meta, dtype=torch.float32).numpy()
        left, right, kept_rank = compute_lowrank_residual(t_cpu.float().numpy(), base, TRAIN_QER_AWARE_RANK)
        repaired = apply_lowrank_residual(base, left, right) if kept_rank > 0 else base
        return torch.from_numpy(repaired).to(device=t.device, dtype=t.dtype).contiguous()
def should_train_qat(name: str) -> bool:
    return bool(TRAIN_QAT_NAME_PATTERNS) and any(pattern in name for pattern in TRAIN_QAT_NAME_PATTERNS)
def fake_quantize_int4_ste(t: Tensor, name: str) -> Tensor:
    if not should_train_qat(name) or LFQAT_RUNTIME["disable_qat"]:
        return t
    target = compression_aware_int4_target(t, TRAIN_QAT_BLOCK_SIZE)
    prob = float(LFQAT_RUNTIME["prob"])
    if prob >= 1.0:
        return t + (target - t).detach()
    if prob <= 0.0:
        return t
    gate = (torch.rand((), device=t.device) < prob).to(dtype=t.dtype)
    return t + ((target - t) * gate).detach()
def lfqat_enabled() -> bool:
    return bool(TRAIN_QAT_NAME_PATTERNS) and (LFQAT_KL_WEIGHT > 0.0 or LFQAT_FISHER_WEIGHT > 0.0 or LFQAT_MIN_PROB != 1.0 or LFQAT_MAX_PROB != 1.0)
def lfqat_prob_for_step(step: int) -> float:
    if LFQAT_FULL_STEP <= LFQAT_START_STEP:
        return LFQAT_MAX_PROB if step >= LFQAT_START_STEP else LFQAT_MIN_PROB
    t = min(max((step - LFQAT_START_STEP) / max(LFQAT_FULL_STEP - LFQAT_START_STEP, 1), 0.0), 1.0)
    return LFQAT_MIN_PROB + (LFQAT_MAX_PROB - LFQAT_MIN_PROB) * t
def compression_aware_alignment_loss(named_params: list[tuple[str, Tensor]]) -> Tensor:
    device = named_params[0][1].device if named_params else torch.device("cpu")
    if TRAIN_COMPRESSION_AWARE_WEIGHT <= 0.0 or not TRAIN_COMPRESSION_AWARE_NAME_PATTERNS:
        return torch.zeros((), dtype=torch.float32, device=device)
    err = torch.zeros((), dtype=torch.float32, device=device)
    signal = torch.zeros((), dtype=torch.float32, device=device)
    matched = 0
    for name, param in named_params:
        if not should_train_compression_align(name, param):
            continue
        target = compression_aware_int4_target(param, TRAIN_COMPRESSION_AWARE_BLOCK_SIZE)
        diff = param.float() - target.float()
        err = err + (diff * diff).sum()
        signal = signal + (target.float() * target.float()).sum()
        matched += 1
    if matched == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.tensor(TRAIN_COMPRESSION_AWARE_WEIGHT, dtype=torch.float32, device=device) * err / signal.clamp_min(1e-6)
def int8_aware_alignment_loss(named_params: list[tuple[str, Tensor]]) -> Tensor:
    device = named_params[0][1].device if named_params else torch.device("cpu")
    if TRAIN_INT8_AWARE_WEIGHT <= 0.0 or not TRAIN_INT8_AWARE_NAME_PATTERNS:
        return torch.zeros((), dtype=torch.float32, device=device)
    err = signal = torch.zeros((), dtype=torch.float32, device=device)
    matched = 0
    for name, param in named_params:
        if not should_train_int8_align(name, param):
            continue
        target = int8_aware_target(param)
        diff = param.float() - target.float()
        err = err + (diff * diff).sum()
        signal = signal + (target.float() * target.float()).sum()
        matched += 1
    if matched == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.tensor(TRAIN_INT8_AWARE_WEIGHT, dtype=torch.float32, device=device) * err / signal.clamp_min(1e-6)
def qer_aware_alignment_loss(named_params: list[tuple[str, Tensor]]) -> Tensor:
    device = named_params[0][1].device if named_params else torch.device("cpu")
    if TRAIN_QER_AWARE_WEIGHT <= 0.0 or not TRAIN_QER_AWARE_NAME_PATTERNS or TRAIN_QER_AWARE_RANK <= 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    err = signal = torch.zeros((), dtype=torch.float32, device=device)
    matched = 0
    for name, param in named_params:
        if not should_train_qer_align(name, param):
            continue
        target = qer_aware_target(param)
        diff = param.float() - target.float()
        err = err + (diff * diff).sum()
        signal = signal + (target.float() * target.float()).sum()
        matched += 1
    if matched == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.tensor(TRAIN_QER_AWARE_WEIGHT, dtype=torch.float32, device=device) * err / signal.clamp_min(1e-6)
def fisher_alignment_loss(named_params: list[tuple[str, Tensor]]) -> Tensor:
    device = named_params[0][1].device if named_params else torch.device("cpu")
    if LFQAT_FISHER_WEIGHT <= 0.0 or not LFQAT_FISHER_EMA:
        return torch.zeros((), dtype=torch.float32, device=device)
    err = signal = torch.zeros((), dtype=torch.float32, device=device)
    matched = 0
    for name, param in named_params:
        weight = LFQAT_FISHER_EMA.get(name)
        if weight is None or not should_train_qat(name):
            continue
        diff = (param.float() - compression_aware_int4_target(param, TRAIN_QAT_BLOCK_SIZE).float()).square().sum()
        ref = param.float().square().sum()
        err = err + diff * weight
        signal = signal + ref * weight
        matched += 1
    if matched == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.tensor(LFQAT_FISHER_WEIGHT, dtype=torch.float32, device=device) * err / signal.clamp_min(1e-6)
def update_lfqat_fisher(named_params: list[tuple[str, Tensor]]) -> None:
    if LFQAT_FISHER_WEIGHT <= 0.0:
        return
    for name, param in named_params:
        if param.grad is None or not should_train_qat(name):
            continue
        g2 = float(param.grad.detach().float().square().mean().item())
        prev = LFQAT_FISHER_EMA.get(name, g2)
        LFQAT_FISHER_EMA[name] = LFQAT_FISHER_DECAY * prev + (1.0 - LFQAT_FISHER_DECAY) * g2
def grad_name_is_trainable(name: str) -> bool:
    if TRAIN_GRAD_ONLY_NAME_PATTERNS and not any(pattern in name for pattern in TRAIN_GRAD_ONLY_NAME_PATTERNS):
        return False
    if TRAIN_GRAD_SKIP_NAME_PATTERNS and any(pattern in name for pattern in TRAIN_GRAD_SKIP_NAME_PATTERNS):
        return False
    return True
def apply_grad_mask(named_params: list[tuple[str, Tensor]]) -> None:
    if not TRAIN_GRAD_ONLY_NAME_PATTERNS and not TRAIN_GRAD_SKIP_NAME_PATTERNS:
        return
    for name, param in named_params:
        if param.grad is not None and not grad_name_is_trainable(name):
            param.grad.zero_()

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    lowrank_left: dict[str, Tensor] = {}
    lowrank_right: dict[str, Tensor] = {}
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

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if should_keep_float_tensor(name) or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        if QUANT_FORMAT == "mixed_codebook_int4_int8_packed_v1" and should_use_int4_tensor(name, t):
            q, s, meta = quantize_float_tensor_codebook_int4_blockwise(t)
        elif should_use_int4_tensor(name, t):
            q, s, meta = quantize_float_tensor_int4_blockwise(t)
        else:
            q, s = quantize_float_tensor_int8(t)
            meta = {"scheme": "per_row", "axis": 0} if s.ndim > 0 else {}
        if should_use_lowrank_error(name, t):
            left_np, right_np, kept_rank = compute_lowrank_residual(
                t.numpy(),
                dequantize_quantized_tensor(q, s, meta).numpy(),
                LOWRANK_ERROR_RANK,
            )
            if kept_rank > 0:
                lowrank_left[name] = torch.from_numpy(left_np).contiguous()
                lowrank_right[name] = torch.from_numpy(right_np).contiguous()
                meta = dict(meta)
                meta["lowrank_rank"] = kept_rank
                stats["int8_payload_bytes"] += tensor_nbytes(lowrank_left[name]) + tensor_nbytes(lowrank_right[name])
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": QUANT_FORMAT,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if lowrank_left:
        obj["lowrank_left"] = lowrank_left
        obj["lowrank_right"] = lowrank_right
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    lowrank_left = obj.get("lowrank_left", {})
    lowrank_right = obj.get("lowrank_right", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        meta = qmeta.get(name, {})
        s = obj["scales"][name]
        out_t = dequantize_quantized_tensor(q, s, meta, dtype=torch.float32)
        if name in lowrank_left:
            out_t = torch.from_numpy(
                apply_lowrank_residual(out_t.numpy(), lowrank_left[name].numpy(), lowrank_right[name].numpy())
            ).contiguous()
        out[name] = out_t.to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
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
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = maybe_pin_cpu_tensor(load_data_shard(self.files[0]))
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = maybe_pin_cpu_tensor(load_data_shard(self.files[self.file_idx]))
        self.pos = 0

    def advance(self, n: int) -> None:
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            step = min(remaining, avail)
            self.pos += step
            remaining -= step

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

    def take_local_span(self, prefix: int, length: int, suffix: int) -> Tensor:
        if prefix > 0:
            self.advance(prefix)
        out = self.take(length)
        if suffix > 0:
            self.advance(suffix)
        return out
class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        local = self.stream.take_local_span(
            self.rank * per_rank_span,
            per_rank_span,
            (self.world_size - self.rank - 1) * per_rank_span,
        )
        local = maybe_pin_cpu_tensor(local).to(device=self.device, dtype=torch.int64, non_blocking=True)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y
# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        name = getattr(self, "weight_name", "")
        weight = fake_quantize_int4_ste(self.weight, name) if self.training else self.weight
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(x.dtype), bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
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
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
            # Only persist the training cache. Eval/inference can reuse a training-built cache,
            # but must not overwrite it with inference tensors that would later break autograd.
            if not self.training:
                return cos.to(dtype=dtype), sin.to(dtype=dtype)
            self._cos_cached = cos
            self._sin_cached = sin
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
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
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
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
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        resid_mix: Tensor,
        attn_scale: Tensor,
        mlp_scale: Tensor,
    ) -> Tensor:
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_unique_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if num_unique_layers <= 0:
            raise ValueError(f"num_unique_layers must be positive, got {num_unique_layers}")
        if num_unique_layers > num_layers:
            raise ValueError(
                f"num_unique_layers={num_unique_layers} cannot exceed num_layers={num_layers}"
            )
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.num_unique_layers = num_unique_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.attn_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
        self.mlp_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
        self.resid_mixes = nn.Parameter(
            torch.stack(
                [
                    torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float()
                    for _ in range(num_layers)
                ]
            )
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_unique_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()
        for i, b in enumerate(self.blocks):
            b.attn.c_q.weight_name = f"blocks.{i}.attn.c_q.weight"
            b.attn.c_k.weight_name = f"blocks.{i}.attn.c_k.weight"
            b.attn.c_v.weight_name = f"blocks.{i}.attn.c_v.weight"
            b.attn.proj.weight_name = f"blocks.{i}.attn.proj.weight"
            b.mlp.fc.weight_name = f"blocks.{i}.mlp.fc.weight"
            b.mlp.proj.weight_name = f"blocks.{i}.mlp.proj.weight"

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def run_step(self, step_idx: int, x: Tensor, x0: Tensor) -> Tensor:
        block = self.blocks[step_idx % self.num_unique_layers]
        return block(
            x,
            x0,
            self.resid_mixes[step_idx],
            self.attn_scales[step_idx],
            self.mlp_scales[step_idx],
        )
    def hidden_states(self, input_ids: Tensor, disable_qat: bool = False) -> Tensor:
        prev_disable = LFQAT_RUNTIME["disable_qat"]
        LFQAT_RUNTIME["disable_qat"] = disable_qat
        try:
            x = self.tok_emb(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x
            skips: list[Tensor] = []
            for i in range(self.num_encoder_layers):
                x = self.run_step(i, x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.run_step(self.num_encoder_layers + i, x, x0)
            return self.final_norm(x)
        finally:
            LFQAT_RUNTIME["disable_qat"] = prev_disable
    def forward_logits(self, input_ids: Tensor, disable_qat: bool = False) -> Tensor:
        x = self.hidden_states(input_ids, disable_qat=disable_qat)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return (self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)).float()

    def forward(
        self, input_ids: Tensor, target_ids: Tensor, return_logits: bool = False, disable_qat: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        logits = self.forward_logits(input_ids, disable_qat=disable_qat)
        flat_logits = logits.reshape(-1, logits.size(-1))
        loss = F.cross_entropy(flat_logits, target_ids.reshape(-1), reduction="mean")
        return (loss, flat_logits) if return_logits else loss
# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = submission_code_text(Path(__file__), Path(__file__).with_name("quant_reconstruction.py"))
    args = Hyperparameters()
    if not args.disable_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = should_enable_ddp(world_size)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "0"))
    if grad_accum_steps <= 0:
        grad_accum_steps = 8 // world_size if 8 % world_size == 0 else 1
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    run_checkpoint_pt = None
    run_checkpoint_int8 = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        run_checkpoint_pt = f"checkpoints/{args.run_id}_final_model.pt"
        run_checkpoint_int8 = f"checkpoints/{args.run_id}_final_model.int8.ptz"
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

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

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
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.init_model_path:
        init_state = torch.load(args.init_model_path, map_location="cpu")
        expected = set(base_model.state_dict().keys())
        found = set(init_state.keys())
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing or extra:
            raise ValueError(
                f"INIT_MODEL_PATH mismatch missing={missing[:5]} extra={extra[:5]} "
                f"(missing={len(missing)} extra={len(extra)})"
            )
        base_model.load_state_dict(init_state, strict=True)
    use_compile = not args.disable_compile and not lfqat_enabled()
    wrapped_model = torch.compile(base_model, dynamic=False, fullgraph=True) if use_compile else base_model
    model: nn.Module = DDP(
        wrapped_model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        static_graph=args.ddp_static_graph,
        gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
    ) if distributed else wrapped_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    named_params = list(base_model.named_parameters())
    reserved_names = {"tok_emb.weight"}
    if base_model.lm_head is not None:
        reserved_names.add("lm_head.weight")
    matrix_param_names = {
        name
        for name, p in named_params
        if name not in reserved_names
        and p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    }
    matrix_params = [p for name, p in named_params if name in matrix_param_names]
    scalar_params = [
        p
        for name, p in named_params
        if name not in reserved_names and name not in matrix_param_names
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"ddp_static_graph:{args.ddp_static_graph} ddp_gradient_as_bucket_view:{args.ddp_gradient_as_bucket_view}")
    log0(
        f"weight_decay:global={args.weight_decay:.5f} adam={args.adam_weight_decay:.5f} "
        f"muon={args.muon_weight_decay:.5f}"
    )
    log0(
        f"ema:enabled={args.ema_decay > 0.0} decay:{args.ema_decay:.5f} "
        f"start_step:{args.ema_start_step} update_every:{args.ema_update_every}"
    )
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"init_model_path:{args.init_model_path if args.init_model_path else '-'} "
        f"quant_format:{QUANT_FORMAT} "
        f"int4_patterns:{','.join(INT4_NAME_PATTERNS) if INT4_NAME_PATTERNS else '-'} "
        f"int4_block_size:{INT4_BLOCK_SIZE} int4_codebook:{INT4_CODEBOOK_NAME}"
    )
    log0(
        f"train_compression_aware_weight:{TRAIN_COMPRESSION_AWARE_WEIGHT} "
        f"train_compression_aware_patterns:{','.join(TRAIN_COMPRESSION_AWARE_NAME_PATTERNS) if TRAIN_COMPRESSION_AWARE_NAME_PATTERNS else '-'} "
        f"train_compression_aware_block_size:{TRAIN_COMPRESSION_AWARE_BLOCK_SIZE}"
    )
    log0(
        f"train_int8_aware_weight:{TRAIN_INT8_AWARE_WEIGHT} "
        f"train_int8_aware_patterns:{','.join(TRAIN_INT8_AWARE_NAME_PATTERNS) if TRAIN_INT8_AWARE_NAME_PATTERNS else '-'} "
        f"train_qer_aware_weight:{TRAIN_QER_AWARE_WEIGHT} train_qer_aware_rank:{TRAIN_QER_AWARE_RANK} "
        f"train_qer_aware_quant_format:{TRAIN_QER_AWARE_QUANT_FORMAT} "
        f"train_qer_aware_codebook:{TRAIN_QER_AWARE_CODEBOOK_NAME} "
        f"train_qer_aware_patterns:{','.join(TRAIN_QER_AWARE_NAME_PATTERNS) if TRAIN_QER_AWARE_NAME_PATTERNS else '-'}"
    )
    log0(
        f"train_grad_only_patterns:{','.join(TRAIN_GRAD_ONLY_NAME_PATTERNS) if TRAIN_GRAD_ONLY_NAME_PATTERNS else '-'} "
        f"train_grad_skip_patterns:{','.join(TRAIN_GRAD_SKIP_NAME_PATTERNS) if TRAIN_GRAD_SKIP_NAME_PATTERNS else '-'}"
    )
    log0(
        f"train_qat_patterns:{','.join(TRAIN_QAT_NAME_PATTERNS) if TRAIN_QAT_NAME_PATTERNS else '-'} "
        f"train_qat_block_size:{TRAIN_QAT_BLOCK_SIZE}"
    )
    log0(
        f"lfqat_enabled:{lfqat_enabled()} lfqat_kl_weight:{LFQAT_KL_WEIGHT} "
        f"lfqat_fisher_weight:{LFQAT_FISHER_WEIGHT} lfqat_temperature:{LFQAT_TEMPERATURE} "
        f"lfqat_prob:{LFQAT_MIN_PROB}->{LFQAT_MAX_PROB} lfqat_steps:{LFQAT_START_STEP}->{LFQAT_FULL_STEP} "
        f"compile:{use_compile}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"layers:{args.num_layers} unique_layers:{args.num_unique_layers} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f} "
        f"lr_schedule:{args.lr_schedule} lr_warmup_iters:{args.lr_warmup_iters} min_lr_scale:{args.min_lr_scale:.3f} "
        f"val_at_step_zero:{args.val_at_step_zero} wallclock_sync_every:{args.wallclock_sync_every} "
        f"eval_stride:{args.eval_stride} eval_batch_seqs:{args.eval_batch_seqs}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        def apply_warmup(scale: float) -> float:
            if args.lr_warmup_iters <= 0:
                return scale
            return scale * min((step + 1) / args.lr_warmup_iters, 1.0)

        if args.lr_schedule == "constant":
            return apply_warmup(1.0)

        if args.lr_schedule == "cosine":
            if max_wallclock_ms is not None:
                progress = min(elapsed_ms / max(max_wallclock_ms, 1e-9), 1.0)
            else:
                progress = min(step / max(args.iterations - 1, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return apply_warmup(args.min_lr_scale + (1.0 - args.min_lr_scale) * cosine)

        if args.lr_schedule != "warmdown":
            raise ValueError(f"unsupported LR_SCHEDULE={args.lr_schedule!r}")

        if args.warmdown_iters <= 0:
            return apply_warmup(1.0)
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            scale = (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
            return apply_warmup(scale)
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        scale = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        return apply_warmup(scale)

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0 and use_compile:
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
                    warmup_loss = model(x, y) + compression_aware_alignment_loss(named_params) + int8_aware_alignment_loss(named_params) + qer_aware_alignment_loss(named_params)
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
    ema = ExponentialMovingAverage(named_params, args.ema_decay, args.ema_start_step, args.ema_update_every) if args.ema_decay > 0 else None

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (
            args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0
        ) or (step == 0 and args.val_at_step_zero)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                base_model,
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
        LFQAT_RUNTIME["prob"] = lfqat_prob_for_step(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                need_logits = lfqat_enabled() and LFQAT_KL_WEIGHT > 0.0
                if need_logits:
                    loss, student_logits = model(x, y, return_logits=True)
                    with torch.no_grad():
                        _, teacher_logits = base_model(x, y, return_logits=True, disable_qat=True)
                    tau = max(LFQAT_TEMPERATURE, 1e-3)
                    loss = loss + LFQAT_KL_WEIGHT * (tau * tau) * F.kl_div(
                        F.log_softmax(student_logits / tau, dim=-1),
                        F.softmax(teacher_logits / tau, dim=-1),
                        reduction="batchmean",
                    )
                else:
                    loss = model(x, y)
                loss = (
                    loss
                    + compression_aware_alignment_loss(named_params)
                    + int8_aware_alignment_loss(named_params)
                    + qer_aware_alignment_loss(named_params)
                    + fisher_alignment_loss(named_params)
                )
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

        update_lfqat_fisher(named_params)
        apply_grad_mask(named_params)
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        if ema is not None:
            ema.update(named_params, step + 1)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"lfqat_prob:{LFQAT_RUNTIME['prob']:.3f}"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None and (
            reached_cap or (args.wallclock_sync_every > 0 and step % args.wallclock_sync_every == 0)
        ):
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.
    def load_state(state_dict: dict[str, Tensor]) -> None:
        base_model.load_state_dict(state_dict, strict=True)

    def eval_state(tag: str, state_dict: dict[str, Tensor]) -> tuple[float, float]:
        load_state(state_dict)
        torch.cuda.synchronize()
        t_eval = time.perf_counter()
        val_loss, val_bpb = eval_val(
            args,
            base_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(f"{tag} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_eval):.0f}ms")
        log0(f"{tag}_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
        return val_loss, val_bpb

    def roundtrip_state(tag: str, state_dict: dict[str, Tensor]) -> dict[str, object]:
        quant_obj, quant_stats = quantize_state_dict_int8(state_dict)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=9)
        load_state(dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")))
        torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        q_val_loss, q_val_bpb = eval_val(
            args,
            base_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
        log0(f"{tag} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{eval_ms:.0f}ms")
        log0(f"{tag}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
        return {
            "val_loss": q_val_loss,
            "val_bpb": q_val_bpb,
            "eval_ms": eval_ms,
            "blob": quant_blob,
            "quant_file_bytes": len(quant_blob),
            "quant_raw_bytes": len(quant_raw),
            "quant_stats": quant_stats,
        }

    raw_state = clone_state_dict_cpu(base_model.state_dict())
    candidates: list[tuple[str, dict[str, Tensor], dict[str, object]]] = [
        ("raw", raw_state, roundtrip_state("raw_int8_zlib_roundtrip", raw_state))
    ]
    if ema is not None and ema.active:
        ema_state = ema.state_dict(raw_state)
        eval_state("final_ema", ema_state)
        candidates.append(("ema", ema_state, roundtrip_state("ema_int8_zlib_roundtrip", ema_state)))
    elif ema is not None:
        log0("ema:inactive no eligible updates were applied before training stopped")

    best_source, best_state, best_result = min(candidates, key=lambda item: float(item[2]["val_bpb"]))
    load_state(best_state)
    if master_process:
        torch.save(best_state, "final_model.pt")
        if run_checkpoint_pt is not None:
            torch.save(best_state, run_checkpoint_pt)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(best_result["blob"])
        if run_checkpoint_int8 is not None:
            with open(run_checkpoint_int8, "wb") as f:
                f.write(best_result["blob"])
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        ratio = best_result["quant_stats"]["baseline_tensor_bytes"] / max(best_result["quant_stats"]["int8_payload_bytes"], 1)
        log0(f"final_export_source:{best_source}")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
        log0(
            f"Serialized model int8+zlib: {best_result['quant_file_bytes']} bytes "
            f"(payload:{best_result['quant_stats']['int8_payload_bytes']} raw_torch:{best_result['quant_raw_bytes']} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {best_result['quant_file_bytes'] + code_bytes} bytes")
    log0(
        f"final_int8_zlib_roundtrip val_loss:{best_result['val_loss']:.4f} val_bpb:{best_result['val_bpb']:.4f} "
        f"eval_time:{best_result['eval_ms']:.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{best_result['val_loss']:.8f} val_bpb:{best_result['val_bpb']:.8f}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
