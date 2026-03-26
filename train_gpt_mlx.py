#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from quant_reconstruction import (
    apply_lowrank_residual,
    compute_lowrank_residual,
    dequantize_codebook_int4_blockwise,
    quantize_codebook_int4_blockwise,
    submission_code_text,
)
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    init_model_path: str = os.environ.get("INIT_MODEL_PATH", "")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    lr_schedule: str = os.environ.get("LR_SCHEDULE", "warmdown").strip().lower()
    lr_warmup_iters: int = int(os.environ.get("LR_WARMUP_ITERS", 0))
    min_lr_scale: float = float(os.environ.get("MIN_LR_SCALE", 0.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    num_unique_layers: int = int(os.environ.get("NUM_UNIQUE_LAYERS", os.environ.get("NUM_LAYERS", 9)))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        def apply_warmup(scale: float) -> float:
            if self.lr_warmup_iters <= 0:
                return scale
            return scale * min((step + 1) / self.lr_warmup_iters, 1.0)

        if self.lr_schedule == "constant":
            return apply_warmup(1.0)

        if self.lr_schedule == "cosine":
            if self.max_wallclock_seconds > 0:
                progress = min(elapsed_ms / max(1000.0 * self.max_wallclock_seconds, 1e-9), 1.0)
            else:
                progress = min(step / max(self.iterations - 1, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return apply_warmup(self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine)

        if self.lr_schedule != "warmdown":
            raise ValueError(f"unsupported LR_SCHEDULE={self.lr_schedule!r}")

        if self.warmdown_iters <= 0:
            return apply_warmup(1.0)
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            scale = (
                max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < self.iterations
                else 1.0
            )
            return apply_warmup(scale)
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        scale = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        return apply_warmup(scale)


TRAIN_COMPRESSION_AWARE_WEIGHT = float(os.environ.get("TRAIN_COMPRESSION_AWARE_WEIGHT", 0.0))
TRAIN_COMPRESSION_AWARE_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("TRAIN_COMPRESSION_AWARE_NAME_PATTERNS", "").split(",")
    if pattern
)
TRAIN_INT8_AWARE_WEIGHT = float(os.environ.get("TRAIN_INT8_AWARE_WEIGHT", 0.0))
TRAIN_INT8_AWARE_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("TRAIN_INT8_AWARE_NAME_PATTERNS", "").split(",")
    if pattern
)
TRAIN_QER_AWARE_WEIGHT = float(os.environ.get("TRAIN_QER_AWARE_WEIGHT", 0.0))
TRAIN_QER_AWARE_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("TRAIN_QER_AWARE_NAME_PATTERNS", "").split(",")
    if pattern
)
TRAIN_QER_AWARE_RANK = int(os.environ.get("TRAIN_QER_AWARE_RANK", 0))
TRAIN_QER_AWARE_QUANT_FORMAT = os.environ.get("TRAIN_QER_AWARE_QUANT_FORMAT", "int8_clean_per_row_v1").strip().lower()
TRAIN_QER_AWARE_BLOCK_SIZE = int(os.environ.get("TRAIN_QER_AWARE_BLOCK_SIZE", os.environ.get("INT4_BLOCK_SIZE", 64)))
TRAIN_QER_AWARE_CODEBOOK_NAME = os.environ.get(
    "TRAIN_QER_AWARE_CODEBOOK_NAME",
    os.environ.get("INT4_CODEBOOK_NAME", "normal16"),
).strip().lower()
TRAIN_COMPRESSION_AWARE_BLOCK_SIZE = int(
    os.environ.get("TRAIN_COMPRESSION_AWARE_BLOCK_SIZE", os.environ.get("INT4_BLOCK_SIZE", 64))
)
TRAIN_GRAD_ONLY_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("TRAIN_GRAD_ONLY_NAME_PATTERNS", "").split(",")
    if pattern
)
TRAIN_GRAD_SKIP_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("TRAIN_GRAD_SKIP_NAME_PATTERNS", "").split(",")
    if pattern
)
TRAIN_QAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("TRAIN_QAT_NAME_PATTERNS", "").split(",")
    if pattern
)
TRAIN_QAT_BLOCK_SIZE = int(os.environ.get("TRAIN_QAT_BLOCK_SIZE", os.environ.get("INT4_BLOCK_SIZE", 64)))
LFQAT_KL_WEIGHT = float(os.environ.get("LFQAT_KL_WEIGHT", 0.0))
LFQAT_FISHER_WEIGHT = float(os.environ.get("LFQAT_FISHER_WEIGHT", 0.0))
LFQAT_TEMPERATURE = float(os.environ.get("LFQAT_TEMPERATURE", 2.0))
LFQAT_FISHER_DECAY = float(os.environ.get("LFQAT_FISHER_DECAY", 0.95))
LFQAT_START_STEP = int(os.environ.get("LFQAT_START_STEP", 0))
LFQAT_FULL_STEP = int(os.environ.get("LFQAT_FULL_STEP", 0))
LFQAT_MIN_PROB = float(os.environ.get("LFQAT_MIN_PROB", 1.0))
LFQAT_MAX_PROB = float(os.environ.get("LFQAT_MAX_PROB", 1.0))
LFQAT_RUNTIME = {"prob": 1.0}
LFQAT_FISHER_EMA: dict[str, float] = {}
TRAIN_TOKEN_WEIGHT_MODE = os.environ.get("TRAIN_TOKEN_WEIGHT_MODE", "uniform").strip().lower()
TRAIN_TOKEN_WEIGHT_POWER = float(os.environ.get("TRAIN_TOKEN_WEIGHT_POWER", "1.0"))
TRAIN_BASE_BYTES_LUT: mx.array | None = None
TRAIN_HAS_LEADING_SPACE_LUT: mx.array | None = None
TRAIN_IS_BOUNDARY_TOKEN_LUT: mx.array | None = None


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,global_bus,second_pass,cross_skip,smear_gate",
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
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "").split(",")
    if pattern
)
ALLOW_INIT_MISSING_KEYS = bool(int(os.environ.get("ALLOW_INIT_MISSING_KEYS", "0")))
GLOBAL_BUS_ENABLED = bool(int(os.environ.get("GLOBAL_BUS_ENABLED", "0")))
GLOBAL_BUS_WRITE_INIT = float(os.environ.get("GLOBAL_BUS_WRITE_INIT", 0.08))
GLOBAL_BUS_DECAY_INIT = float(os.environ.get("GLOBAL_BUS_DECAY_INIT", 0.5))
GLOBAL_BUS_READ_LAYERS_SPEC = os.environ.get("GLOBAL_BUS_READ_LAYERS", "").strip()
GLOBAL_BUS_WRITE_LAYERS_SPEC = os.environ.get("GLOBAL_BUS_WRITE_LAYERS", "").strip()
GLOBAL_BUS_SUMMARY_MODE = os.environ.get("GLOBAL_BUS_SUMMARY_MODE", "mean").strip().lower()
GLOBAL_BUS_TAIL_WEIGHT = float(os.environ.get("GLOBAL_BUS_TAIL_WEIGHT", 0.5))
GLOBAL_BUS_BOUNDARY_LUT: mx.array | None = None
SECOND_PASS_ENABLED = bool(int(os.environ.get("SECOND_PASS_ENABLED", "0")))
SECOND_PASS_LAYERS_SPEC = os.environ.get("SECOND_PASS_LAYERS", "").strip()
SECOND_PASS_GATE_INIT = float(os.environ.get("SECOND_PASS_GATE_INIT", 0.0))
SECOND_PASS_SKIP_INIT = float(os.environ.get("SECOND_PASS_SKIP_INIT", 0.0))
CROSS_SKIP_ROUTER_ENABLED = bool(int(os.environ.get("CROSS_SKIP_ROUTER_ENABLED", "0")))
CROSS_SKIP_ROUTER_MATCH_INIT = float(os.environ.get("CROSS_SKIP_ROUTER_MATCH_INIT", 4.0))
CROSS_SKIP_ROUTER_OTHER_INIT = float(os.environ.get("CROSS_SKIP_ROUTER_OTHER_INIT", -4.0))
CROSS_SKIP_ROUTER_DECODER_LAYERS_SPEC = os.environ.get("CROSS_SKIP_ROUTER_DECODER_LAYERS", "").strip()
CROSS_SKIP_ROUTER_SOURCE_LAYERS_SPEC = os.environ.get("CROSS_SKIP_ROUTER_SOURCE_LAYERS", "").strip()
SMEAR_GATE_ENABLED = bool(int(os.environ.get("SMEAR_GATE_ENABLED", "0")))
SMEAR_GATE_INIT = float(os.environ.get("SMEAR_GATE_INIT", 0.0))
BIGRAM_HASH_ENABLED = bool(int(os.environ.get("BIGRAM_HASH_ENABLED", "0")))
BIGRAM_HASH_BUCKETS = int(os.environ.get("BIGRAM_HASH_BUCKETS", 2048))
BIGRAM_HASH_DIM = int(os.environ.get("BIGRAM_HASH_DIM", 64))
BIGRAM_HASH_INIT_STD = float(os.environ.get("BIGRAM_HASH_INIT_STD", 0.005))
LOGICAL_BLOCK_ORDER_SPEC = os.environ.get("LOGICAL_BLOCK_ORDER", "").strip()


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def previous_token_ids(input_ids: mx.array) -> mx.array:
    return mx.concatenate((input_ids[:, :1], input_ids[:, :-1]), axis=1)


def parse_layer_selector(spec: str, num_layers: int) -> tuple[bool, ...]:
    spec = spec.strip().lower()
    if not spec or spec == "all":
        return tuple(True for _ in range(num_layers))
    selected = [False] * num_layers
    for raw_part in spec.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part == "all":
            return tuple(True for _ in range(num_layers))
        if part.startswith("top") and part[3:].isdigit():
            count = max(0, min(int(part[3:]), num_layers))
            for idx in range(num_layers - count, num_layers):
                selected[idx] = True
            continue
        if part.startswith("bottom") and part[6:].isdigit():
            count = max(0, min(int(part[6:]), num_layers))
            for idx in range(count):
                selected[idx] = True
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else num_layers
            start = max(0, min(start, num_layers))
            end = max(start, min(end, num_layers))
            for idx in range(start, end):
                selected[idx] = True
            continue
        idx = int(part)
        if idx < 0:
            idx += num_layers
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"layer index out of range for selector {spec!r}: {part!r}")
        selected[idx] = True
    if not any(selected):
        raise ValueError(f"layer selector {spec!r} selected no layers")
    return tuple(selected)


def compact_layer_selector(mask: tuple[bool, ...]) -> str:
    active = [idx for idx, enabled in enumerate(mask) if enabled]
    if len(active) == len(mask):
        return "all"
    return ",".join(str(idx) for idx in active)


@lru_cache(maxsize=None)
def layer_selector_mask(spec: str, num_layers: int) -> tuple[bool, ...]:
    return parse_layer_selector(spec, num_layers)


@lru_cache(maxsize=None)
def second_pass_decoder_mask(num_layers: int, num_encoder_layers: int, spec: str) -> tuple[bool, ...]:
    return tuple(
        enabled if idx >= num_encoder_layers else False
        for idx, enabled in enumerate(layer_selector_mask(spec, num_layers))
    )


@lru_cache(maxsize=None)
def cross_skip_decoder_mask(num_layers: int, num_encoder_layers: int, spec: str) -> tuple[bool, ...]:
    mask = tuple(
        enabled if idx >= num_encoder_layers else False
        for idx, enabled in enumerate(layer_selector_mask(spec, num_layers))
    )
    if not any(mask[idx] for idx in range(num_encoder_layers, num_layers)):
        raise ValueError(f"cross-skip decoder selector {spec!r} selected no decoder layers")
    return mask


@lru_cache(maxsize=None)
def cross_skip_source_mask(num_layers: int, num_encoder_layers: int, spec: str) -> tuple[bool, ...]:
    mask = tuple(
        enabled if idx < num_encoder_layers else False
        for idx, enabled in enumerate(layer_selector_mask(spec, num_layers))
    )
    if not any(mask[:num_encoder_layers]):
        raise ValueError(f"cross-skip source selector {spec!r} selected no encoder layers")
    return mask


@lru_cache(maxsize=None)
def logical_block_order(num_layers: int, num_unique_layers: int, spec: str) -> tuple[int, ...]:
    if not spec or spec.lower() in {"default", "modulo"}:
        return tuple(idx % num_unique_layers for idx in range(num_layers))
    order: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += num_unique_layers
        if idx < 0 or idx >= num_unique_layers:
            raise ValueError(f"logical block index out of range for order {spec!r}: {part!r}")
        order.append(idx)
    if len(order) != num_layers:
        raise ValueError(
            f"logical block order {spec!r} must provide exactly {num_layers} indices, got {len(order)}"
        )
    return tuple(order)


def compact_logical_block_order(order: tuple[int, ...], num_unique_layers: int) -> str:
    default = tuple(idx % num_unique_layers for idx in range(len(order)))
    if order == default:
        return "modulo"
    return ",".join(str(idx) for idx in order)


def configure_global_bus_boundary_lut(
    has_leading_space_lut: np.ndarray, is_boundary_token_lut: np.ndarray
) -> None:
    global GLOBAL_BUS_BOUNDARY_LUT
    boundary_lut = np.logical_or(has_leading_space_lut, is_boundary_token_lut)
    GLOBAL_BUS_BOUNDARY_LUT = mx.array(boundary_lut.astype(np.bool_), dtype=mx.bool_)


def configure_train_token_weight_luts(
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> None:
    global TRAIN_BASE_BYTES_LUT, TRAIN_HAS_LEADING_SPACE_LUT, TRAIN_IS_BOUNDARY_TOKEN_LUT
    TRAIN_BASE_BYTES_LUT = mx.array(base_bytes_lut.astype(np.float32), dtype=mx.float32)
    TRAIN_HAS_LEADING_SPACE_LUT = mx.array(has_leading_space_lut.astype(np.bool_), dtype=mx.bool_)
    TRAIN_IS_BOUNDARY_TOKEN_LUT = mx.array(is_boundary_token_lut.astype(np.bool_), dtype=mx.bool_)


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


def grad_name_is_trainable(name: str) -> bool:
    if TRAIN_GRAD_ONLY_NAME_PATTERNS and not any(pattern in name for pattern in TRAIN_GRAD_ONLY_NAME_PATTERNS):
        return False
    if TRAIN_GRAD_SKIP_NAME_PATTERNS and any(pattern in name for pattern in TRAIN_GRAD_SKIP_NAME_PATTERNS):
        return False
    return True


def filter_grad_tree(grads_tree: dict) -> dict:
    if not TRAIN_GRAD_ONLY_NAME_PATTERNS and not TRAIN_GRAD_SKIP_NAME_PATTERNS:
        return grads_tree
    return tree_unflatten(
        [
            (name, grad if grad_name_is_trainable(name) else mx.zeros_like(grad))
            for name, grad in tree_flatten(grads_tree)
        ]
    )


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def should_train_compression_align(name: str, arr: mx.array) -> bool:
    return (
        TRAIN_COMPRESSION_AWARE_WEIGHT > 0.0
        and arr.ndim == 2
        and any(pattern in name for pattern in TRAIN_COMPRESSION_AWARE_NAME_PATTERNS)
    )


def should_train_int8_align(name: str, arr: mx.array) -> bool:
    return (
        TRAIN_INT8_AWARE_WEIGHT > 0.0
        and mx.issubdtype(arr.dtype, mx.floating)
        and any(pattern in name for pattern in TRAIN_INT8_AWARE_NAME_PATTERNS)
    )


def compression_aware_int4_target(arr: mx.array, block_size: int) -> mx.array:
    if arr.ndim != 2:
        raise ValueError(f"compression-aware int4 target expects 2D tensors, got shape={arr.shape}")
    if block_size <= 0:
        raise ValueError(f"TRAIN_COMPRESSION_AWARE_BLOCK_SIZE must be positive, got {block_size}")
    rows, cols = arr.shape
    blocks_per_row = (cols + block_size - 1) // block_size
    padded_cols = blocks_per_row * block_size
    arr32 = arr.astype(mx.float32)
    if padded_cols != cols:
        padded = mx.concatenate(
            (arr32, mx.zeros((rows, padded_cols - cols), dtype=mx.float32)),
            axis=1,
        )
    else:
        padded = arr32
    reshaped = padded.reshape(rows, blocks_per_row, block_size)
    clip_abs = mx.maximum(mx.max(mx.abs(reshaped), axis=2), mx.array(1.0 / 7.0, dtype=mx.float32))
    scale = clip_abs / 7.0
    q = mx.clip(mx.round(reshaped / scale[..., None]), -7, 7)
    target = (q * scale[..., None]).reshape(rows, padded_cols)[:, :cols]
    return mx.stop_gradient(target.astype(arr.dtype))


def int8_aware_target(arr: mx.array) -> mx.array:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = (
            np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
            if f32.size
            else np.empty((f32.shape[0],), dtype=np.float32)
        )
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        target = q.astype(np.float32, copy=False) * scale[:, None]
    else:
        clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
        scale = clip_abs / 127.0 if clip_abs > 0.0 else 1.0
        q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
        target = q.astype(np.float32, copy=False) * np.float32(scale)
    return mx.stop_gradient(mx.array(target, dtype=arr.dtype))


def should_train_qat(name: str) -> bool:
    return bool(TRAIN_QAT_NAME_PATTERNS) and any(pattern in name for pattern in TRAIN_QAT_NAME_PATTERNS)
def fake_quantize_int4_ste(arr: mx.array, name: str) -> mx.array:
    if not should_train_qat(name):
        return arr
    target = compression_aware_int4_target(arr, TRAIN_QAT_BLOCK_SIZE)
    prob = float(LFQAT_RUNTIME["prob"])
    if prob >= 1.0:
        return arr + mx.stop_gradient(target - arr)
    if prob <= 0.0:
        return arr
    gate = (mx.random.uniform() < prob).astype(arr.dtype)
    mixed = gate * target + (1.0 - gate) * arr
    return arr + mx.stop_gradient(mixed - arr)
def compression_aware_alignment_loss(flat_params: dict[str, mx.array]) -> mx.array:
    if TRAIN_COMPRESSION_AWARE_WEIGHT <= 0.0 or not TRAIN_COMPRESSION_AWARE_NAME_PATTERNS:
        return mx.array(0.0, dtype=mx.float32)
    err = mx.array(0.0, dtype=mx.float32)
    signal = mx.array(0.0, dtype=mx.float32)
    matched = 0
    for name, arr in flat_params.items():
        if not should_train_compression_align(name, arr):
            continue
        target = compression_aware_int4_target(arr, TRAIN_COMPRESSION_AWARE_BLOCK_SIZE)
        arr32 = arr.astype(mx.float32)
        target32 = target.astype(mx.float32)
        diff = arr32 - target32
        err = err + mx.sum(diff * diff)
        signal = signal + mx.sum(target32 * target32)
        matched += 1
    if matched == 0:
        return mx.array(0.0, dtype=mx.float32)
    return mx.array(TRAIN_COMPRESSION_AWARE_WEIGHT, dtype=mx.float32) * err / mx.maximum(
        signal,
        mx.array(1e-6, dtype=mx.float32),
    )


def int8_aware_alignment_loss(flat_params: dict[str, mx.array]) -> mx.array:
    if TRAIN_INT8_AWARE_WEIGHT <= 0.0 or not TRAIN_INT8_AWARE_NAME_PATTERNS:
        return mx.array(0.0, dtype=mx.float32)
    err = mx.array(0.0, dtype=mx.float32)
    signal = mx.array(0.0, dtype=mx.float32)
    matched = 0
    for name, arr in flat_params.items():
        if not should_train_int8_align(name, arr):
            continue
        target = int8_aware_target(arr)
        arr32 = arr.astype(mx.float32)
        target32 = target.astype(mx.float32)
        diff = arr32 - target32
        err = err + mx.sum(diff * diff)
        signal = signal + mx.sum(target32 * target32)
        matched += 1
    if matched == 0:
        return mx.array(0.0, dtype=mx.float32)
    return mx.array(TRAIN_INT8_AWARE_WEIGHT, dtype=mx.float32) * err / mx.maximum(
        signal,
        mx.array(1e-6, dtype=mx.float32),
    )


def should_train_qer_align(name: str, arr: mx.array) -> bool:
    return (
        TRAIN_QER_AWARE_WEIGHT > 0.0
        and TRAIN_QER_AWARE_RANK > 0
        and arr.ndim == 2
        and any(pattern in name for pattern in TRAIN_QER_AWARE_NAME_PATTERNS)
    )


def qer_aware_target(arr: mx.array) -> mx.array:
    arr_np = _np_float32(arr)
    if TRAIN_QER_AWARE_QUANT_FORMAT == "mixed_codebook_int4_int8_packed_v1":
        signed, scale, meta = quantize_codebook_int4_blockwise(
            arr_np,
            TRAIN_QER_AWARE_BLOCK_SIZE,
            clip_q=INT4_CLIP_Q,
            codebook_name=TRAIN_QER_AWARE_CODEBOOK_NAME,
        )
        base = dequantize_codebook_int4_blockwise(signed, scale, meta)
    elif TRAIN_QER_AWARE_QUANT_FORMAT == "mixed_int4_int8_packed_v2":
        target = compression_aware_int4_target(arr, TRAIN_QER_AWARE_BLOCK_SIZE)
        base = _np_float32(target)
    else:
        base = _np_float32(int8_aware_target(arr))
    left, right, kept_rank = compute_lowrank_residual(arr_np, base, TRAIN_QER_AWARE_RANK)
    repaired = apply_lowrank_residual(base, left, right) if kept_rank > 0 else base
    return mx.stop_gradient(mx.array(repaired, dtype=arr.dtype))


def qer_aware_alignment_loss(flat_params: dict[str, mx.array]) -> mx.array:
    if TRAIN_QER_AWARE_WEIGHT <= 0.0 or not TRAIN_QER_AWARE_NAME_PATTERNS or TRAIN_QER_AWARE_RANK <= 0:
        return mx.array(0.0, dtype=mx.float32)
    err = mx.array(0.0, dtype=mx.float32)
    signal = mx.array(0.0, dtype=mx.float32)
    matched = 0
    for name, arr in flat_params.items():
        if not should_train_qer_align(name, arr):
            continue
        target = qer_aware_target(arr)
        arr32 = arr.astype(mx.float32)
        target32 = target.astype(mx.float32)
        diff = arr32 - target32
        err = err + mx.sum(diff * diff)
        signal = signal + mx.sum(target32 * target32)
        matched += 1
    if matched == 0:
        return mx.array(0.0, dtype=mx.float32)
    return mx.array(TRAIN_QER_AWARE_WEIGHT, dtype=mx.float32) * err / mx.maximum(signal, mx.array(1e-6, dtype=mx.float32))
def lfqat_enabled() -> bool:
    return bool(TRAIN_QAT_NAME_PATTERNS) and (LFQAT_KL_WEIGHT > 0.0 or LFQAT_FISHER_WEIGHT > 0.0 or LFQAT_MIN_PROB != 1.0 or LFQAT_MAX_PROB != 1.0)


def int8_aware_enabled() -> bool:
    return TRAIN_INT8_AWARE_WEIGHT > 0.0 and bool(TRAIN_INT8_AWARE_NAME_PATTERNS)


def qer_aware_enabled() -> bool:
    return TRAIN_QER_AWARE_WEIGHT > 0.0 and bool(TRAIN_QER_AWARE_NAME_PATTERNS) and TRAIN_QER_AWARE_RANK > 0


def lfqat_prob_for_step(step: int) -> float:
    if not lfqat_enabled():
        return 1.0
    if LFQAT_FULL_STEP <= LFQAT_START_STEP:
        return LFQAT_MAX_PROB if step >= LFQAT_START_STEP else LFQAT_MIN_PROB
    t = min(max((step - LFQAT_START_STEP) / max(LFQAT_FULL_STEP - LFQAT_START_STEP, 1), 0.0), 1.0)
    return LFQAT_MIN_PROB + (LFQAT_MAX_PROB - LFQAT_MIN_PROB) * t
def lfqat_fisher_alignment_loss(flat_params: dict[str, mx.array]) -> mx.array:
    if LFQAT_FISHER_WEIGHT <= 0.0 or not LFQAT_FISHER_EMA:
        return mx.array(0.0, dtype=mx.float32)
    err = mx.array(0.0, dtype=mx.float32)
    signal = mx.array(0.0, dtype=mx.float32)
    matched = 0
    for name, arr in flat_params.items():
        weight = LFQAT_FISHER_EMA.get(name)
        if weight is None or not should_train_qat(name):
            continue
        target = compression_aware_int4_target(arr, TRAIN_QAT_BLOCK_SIZE)
        diff = arr.astype(mx.float32) - target.astype(mx.float32)
        err = err + mx.array(weight, dtype=mx.float32) * mx.sum(diff * diff)
        signal = signal + mx.array(weight, dtype=mx.float32) * mx.sum(target.astype(mx.float32) * target.astype(mx.float32))
        matched += 1
    if matched == 0:
        return mx.array(0.0, dtype=mx.float32)
    return mx.array(LFQAT_FISHER_WEIGHT, dtype=mx.float32) * err / mx.maximum(signal, mx.array(1e-6, dtype=mx.float32))
def update_lfqat_fisher(grads_tree: dict) -> None:
    if LFQAT_FISHER_WEIGHT <= 0.0:
        return
    for name, grad in tree_flatten(grads_tree):
        if not should_train_qat(name):
            continue
        g2 = float(np.mean(np.square(_np_float32(grad)), dtype=np.float64))
        prev = LFQAT_FISHER_EMA.get(name, g2)
        LFQAT_FISHER_EMA[name] = LFQAT_FISHER_DECAY * prev + (1.0 - LFQAT_FISHER_DECAY) * g2


def train_token_weights(input_ids: mx.array, target_ids: mx.array) -> mx.array | None:
    if TRAIN_TOKEN_WEIGHT_MODE == "uniform":
        return None
    if TRAIN_TOKEN_WEIGHT_MODE != "bytes":
        raise ValueError(f"unsupported TRAIN_TOKEN_WEIGHT_MODE={TRAIN_TOKEN_WEIGHT_MODE!r}")
    if (
        TRAIN_BASE_BYTES_LUT is None
        or TRAIN_HAS_LEADING_SPACE_LUT is None
        or TRAIN_IS_BOUNDARY_TOKEN_LUT is None
    ):
        raise ValueError("TRAIN_TOKEN_WEIGHT_MODE=bytes requires tokenizer weight LUTs")
    prev_ids = input_ids.reshape(-1)
    tgt_ids = target_ids.reshape(-1)
    weights = TRAIN_BASE_BYTES_LUT[tgt_ids]
    weights = weights + (TRAIN_HAS_LEADING_SPACE_LUT[tgt_ids] & ~TRAIN_IS_BOUNDARY_TOKEN_LUT[prev_ids]).astype(mx.float32)
    if TRAIN_TOKEN_WEIGHT_POWER != 1.0:
        weights = mx.power(weights, mx.array(TRAIN_TOKEN_WEIGHT_POWER, dtype=mx.float32))
    return mx.stop_gradient(weights.astype(mx.float32))


def cross_entropy_loss(logits: mx.array, targets: mx.array, token_weights: mx.array | None = None) -> mx.array:
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none").astype(mx.float32)
    if token_weights is None:
        return mx.mean(losses)
    weights = token_weights.astype(mx.float32)
    return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), mx.array(1e-6, dtype=mx.float32))


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)
        self.weight_name = ""

    def __call__(self, x: mx.array, qat: bool = False) -> mx.array:
        w = fake_quantize_int4_ste(self.weight, self.weight_name) if qat else self.weight
        return x @ w.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
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
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, qat: bool = False) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, qat=qat).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x, qat=qat).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x, qat=qat).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y, qat=qat)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array, qat: bool = False) -> mx.array:
        x = nn.relu(self.fc(x, qat=qat))
        return self.proj(x * x, qat=qat)


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
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)

    def __call__(
        self,
        x: mx.array,
        x0: mx.array,
        resid_mix: mx.array,
        attn_scale: mx.array,
        mlp_scale: mx.array,
        global_bus: mx.array | None = None,
        boundary_mask: mx.array | None = None,
        global_bus_read_scale: mx.array | None = None,
        global_bus_write_scale: mx.array | None = None,
        global_bus_decay_logit: mx.array | None = None,
        qat: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        mix = resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if global_bus is not None and global_bus_read_scale is not None:
            x = x + global_bus_read_scale.astype(x.dtype)[None, None, :] * global_bus.astype(x.dtype)
        attn_out = self.attn(self.attn_norm(x), qat=qat)
        x = x + attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x), qat=qat)
        if (
            global_bus is None
            or global_bus_write_scale is None
            or global_bus_decay_logit is None
        ):
            return x, global_bus
        normed = rms_norm(x)
        if GLOBAL_BUS_SUMMARY_MODE == "mean":
            summary = mx.mean(normed, axis=1, keepdims=True)
        elif GLOBAL_BUS_SUMMARY_MODE == "tail":
            summary = normed[:, -1:, :]
        elif GLOBAL_BUS_SUMMARY_MODE == "mean_tail":
            tail_weight = float(np.clip(GLOBAL_BUS_TAIL_WEIGHT, 0.0, 1.0))
            mean_summary = mx.mean(normed, axis=1, keepdims=True)
            tail_summary = normed[:, -1:, :]
            summary = (1.0 - tail_weight) * mean_summary + tail_weight * tail_summary
        elif GLOBAL_BUS_SUMMARY_MODE in {"boundary_mean", "boundary_mean_tail"}:
            if boundary_mask is None:
                raise ValueError("boundary-aware global bus summary requires boundary_mask")
            weights = boundary_mask.astype(normed.dtype)[..., None]
            counts = mx.sum(weights, axis=1, keepdims=True)
            mean_summary = mx.mean(normed, axis=1, keepdims=True)
            boundary_summary = mx.sum(normed * weights, axis=1, keepdims=True) / mx.maximum(counts, 1.0)
            has_boundary = (counts > 0).astype(normed.dtype)
            boundary_summary = has_boundary * boundary_summary + (1.0 - has_boundary) * mean_summary
            if GLOBAL_BUS_SUMMARY_MODE == "boundary_mean":
                summary = boundary_summary
            else:
                tail_weight = float(np.clip(GLOBAL_BUS_TAIL_WEIGHT, 0.0, 1.0))
                tail_summary = normed[:, -1:, :]
                summary = (1.0 - tail_weight) * boundary_summary + tail_weight * tail_summary
        else:
            raise ValueError(f"unsupported GLOBAL_BUS_SUMMARY_MODE={GLOBAL_BUS_SUMMARY_MODE!r}")
        decay = mx.sigmoid(global_bus_decay_logit.astype(x.dtype))
        next_bus = decay * global_bus.astype(x.dtype) + (1.0 - decay) * (
            global_bus_write_scale.astype(x.dtype)[None, None, :] * summary
        )
        return x, next_bus


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - supports shared recurrent blocks via num_unique_layers < num_layers
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_unique_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
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
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.num_unique_layers = num_unique_layers
        self.global_bus_enabled = GLOBAL_BUS_ENABLED
        self.second_pass_enabled = SECOND_PASS_ENABLED
        self.cross_skip_router_enabled = CROSS_SKIP_ROUTER_ENABLED
        self.smear_gate_enabled = SMEAR_GATE_ENABLED
        self.bigram_hash_enabled = BIGRAM_HASH_ENABLED

        self.tok_emb = nn.Embedding(vocab_size, dim)
        if self.smear_gate_enabled:
            self.smear_gate = mx.ones((dim,), dtype=mx.float32) * SMEAR_GATE_INIT
        if self.bigram_hash_enabled:
            if BIGRAM_HASH_BUCKETS <= 0:
                raise ValueError(f"BIGRAM_HASH_BUCKETS must be positive, got {BIGRAM_HASH_BUCKETS}")
            if BIGRAM_HASH_DIM <= 0:
                raise ValueError(f"BIGRAM_HASH_DIM must be positive, got {BIGRAM_HASH_DIM}")
            self.bigram_hash_buckets = BIGRAM_HASH_BUCKETS
            self.bigram_hash_dim = BIGRAM_HASH_DIM
            self.bigram_hash_emb = nn.Embedding(BIGRAM_HASH_BUCKETS, BIGRAM_HASH_DIM)
            self.bigram_hash_proj = nn.Linear(BIGRAM_HASH_DIM, dim, bias=False)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.attn_scales = mx.ones((num_layers, dim), dtype=mx.float32)
        self.mlp_scales = mx.ones((num_layers, dim), dtype=mx.float32)
        self.resid_mixes = mx.array(
            np.stack(
                [
                    np.stack(
                        (
                            np.ones((dim,), dtype=np.float32),
                            np.zeros((dim,), dtype=np.float32),
                        )
                    )
                    for _ in range(num_layers)
                ]
            )
        )
        if self.global_bus_enabled:
            self.global_bus_read_scales = mx.zeros((num_layers, dim), dtype=mx.float32)
            self.global_bus_write_scales = mx.ones((num_layers, dim), dtype=mx.float32) * GLOBAL_BUS_WRITE_INIT
            decay_init = float(np.clip(GLOBAL_BUS_DECAY_INIT, 1e-4, 1.0 - 1e-4))
            self.global_bus_decay_logits = mx.ones((num_layers,), dtype=mx.float32) * np.log(
                decay_init / (1.0 - decay_init)
            )
        if self.second_pass_enabled:
            self.second_pass_gates = mx.ones((self.num_decoder_layers, dim), dtype=mx.float32) * SECOND_PASS_GATE_INIT
            self.second_pass_skip_scales = mx.ones((self.num_skip_weights, dim), dtype=mx.float32) * SECOND_PASS_SKIP_INIT
        if self.cross_skip_router_enabled:
            router_init = np.full(
                (self.num_decoder_layers, self.num_encoder_layers),
                CROSS_SKIP_ROUTER_OTHER_INIT,
                dtype=np.float32,
            )
            for decoder_idx in range(self.num_decoder_layers):
                source_idx = max(0, min(self.num_encoder_layers - 1, self.num_encoder_layers - 1 - decoder_idx))
                router_init[decoder_idx, source_idx] = CROSS_SKIP_ROUTER_MATCH_INIT
            self.cross_skip_router_logits = mx.array(router_init, dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_unique_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        if self.bigram_hash_enabled:
            self.bigram_hash_emb.weight = (
                mx.random.normal(self.bigram_hash_emb.weight.shape, dtype=mx.float32) * BIGRAM_HASH_INIT_STD
            ).astype(COMPUTE_DTYPE)
            self.bigram_hash_proj.weight = mx.zeros_like(self.bigram_hash_proj.weight)
        for i, b in enumerate(self.blocks):
            b.attn.c_q.weight_name = f"blocks.{i}.attn.c_q.weight"
            b.attn.c_k.weight_name = f"blocks.{i}.attn.c_k.weight"
            b.attn.c_v.weight_name = f"blocks.{i}.attn.c_v.weight"
            b.attn.proj.weight_name = f"blocks.{i}.attn.proj.weight"
            b.mlp.fc.weight_name = f"blocks.{i}.mlp.fc.weight"
            b.mlp.proj.weight_name = f"blocks.{i}.mlp.proj.weight"
        if self.bigram_hash_enabled:
            self.bigram_hash_emb.weight_name = "bigram_hash_emb.weight"
            self.bigram_hash_proj.weight_name = "bigram_hash_proj.weight"

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def run_step(
        self,
        step_idx: int,
        x: mx.array,
        x0: mx.array,
        global_bus: mx.array | None = None,
        boundary_mask: mx.array | None = None,
        qat: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        block = self.blocks[logical_block_order(self.num_layers, self.num_unique_layers, LOGICAL_BLOCK_ORDER_SPEC)[step_idx]]
        read_enabled = self.global_bus_enabled and layer_selector_mask(GLOBAL_BUS_READ_LAYERS_SPEC, self.num_layers)[step_idx]
        write_enabled = self.global_bus_enabled and layer_selector_mask(GLOBAL_BUS_WRITE_LAYERS_SPEC, self.num_layers)[step_idx]
        return block(
            x,
            x0,
            self.resid_mixes[step_idx],
            self.attn_scales[step_idx],
            self.mlp_scales[step_idx],
            global_bus=global_bus if (read_enabled or write_enabled) else None,
            boundary_mask=boundary_mask if write_enabled else None,
            global_bus_read_scale=self.global_bus_read_scales[step_idx] if read_enabled else None,
            global_bus_write_scale=self.global_bus_write_scales[step_idx] if write_enabled else None,
            global_bus_decay_logit=self.global_bus_decay_logits[step_idx] if write_enabled else None,
            qat=qat,
        )

    def __call__(self, input_ids: mx.array, qat: bool = False) -> mx.array:
        tok = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        prev_ids = None
        if self.smear_gate_enabled or self.bigram_hash_enabled:
            prev_ids = previous_token_ids(input_ids)
        if self.smear_gate_enabled and prev_ids is not None:
            prev_tok = self.tok_emb(prev_ids).astype(tok.dtype)
            smear_gate = mx.tanh(self.smear_gate.astype(tok.dtype))[None, None, :]
            tok = tok + smear_gate * (prev_tok - tok)
        if self.bigram_hash_enabled and prev_ids is not None:
            hash_ids = ((prev_ids.astype(mx.int32) * 31) + input_ids.astype(mx.int32)) % self.bigram_hash_buckets
            bigram_hidden = self.bigram_hash_emb(hash_ids).astype(tok.dtype)
            tok = tok + self.bigram_hash_proj(bigram_hidden).astype(tok.dtype)
        x = rms_norm(tok)
        x0 = x
        global_bus = mx.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype) if self.global_bus_enabled else None
        boundary_mask = None
        cross_skip_decoder_layers = (
            cross_skip_decoder_mask(self.num_layers, self.num_encoder_layers, CROSS_SKIP_ROUTER_DECODER_LAYERS_SPEC)
            if self.cross_skip_router_enabled
            else None
        )
        cross_skip_source_layers = (
            cross_skip_source_mask(self.num_layers, self.num_encoder_layers, CROSS_SKIP_ROUTER_SOURCE_LAYERS_SPEC)
            if self.cross_skip_router_enabled
            else None
        )
        if self.global_bus_enabled and GLOBAL_BUS_SUMMARY_MODE.startswith("boundary"):
            if GLOBAL_BUS_BOUNDARY_LUT is None:
                raise ValueError("GLOBAL_BUS_BOUNDARY_LUT must be configured for boundary-aware global bus summaries")
            boundary_mask = GLOBAL_BUS_BOUNDARY_LUT[input_ids]
        skips: list[mx.array] = []
        encoder_skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x, global_bus = self.run_step(i, x, x0, global_bus=global_bus, boundary_mask=boundary_mask, qat=qat)
            skips.append(x)
            encoder_skips.append(x)
        stacked_encoder_skips = mx.stack(encoder_skips, axis=0) if self.cross_skip_router_enabled else None
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            decoder_logical_idx = self.num_encoder_layers + i
            if (
                self.cross_skip_router_enabled
                and stacked_encoder_skips is not None
                and cross_skip_decoder_layers is not None
                and cross_skip_decoder_layers[decoder_logical_idx]
            ):
                logits = self.cross_skip_router_logits[i].astype(x.dtype)
                if cross_skip_source_layers is not None:
                    source_mask = mx.array(cross_skip_source_layers[: self.num_encoder_layers], dtype=mx.bool_)
                    logits = mx.where(source_mask, logits, mx.zeros_like(logits) - 1e9)
                weights = mx.softmax(logits, axis=0)
                routed_skip = mx.sum(weights[:, None, None, None] * stacked_encoder_skips.astype(x.dtype), axis=0)
                x = x + self.skip_weights[min(i, self.num_skip_weights - 1)].astype(x.dtype)[None, None, :] * routed_skip
            elif skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x, global_bus = self.run_step(
                decoder_logical_idx,
                x,
                x0,
                global_bus=global_bus,
                boundary_mask=boundary_mask,
                qat=qat,
            )
        second_pass_mask = second_pass_decoder_mask(self.num_layers, self.num_encoder_layers, SECOND_PASS_LAYERS_SPEC)
        if self.second_pass_enabled and any(second_pass_mask):
            for decoder_idx in range(self.num_decoder_layers):
                logical_idx = self.num_encoder_layers + decoder_idx
                if not second_pass_mask[logical_idx]:
                    continue
                candidate = x
                if decoder_idx < self.num_skip_weights:
                    candidate = candidate + self.second_pass_skip_scales[decoder_idx].astype(candidate.dtype)[None, None, :] * encoder_skips[-1 - decoder_idx]
                candidate_x, candidate_bus = self.run_step(
                    logical_idx,
                    candidate,
                    x0,
                    global_bus=global_bus,
                    boundary_mask=boundary_mask,
                    qat=qat,
                )
                gate = mx.tanh(self.second_pass_gates[decoder_idx].astype(candidate_x.dtype))[None, None, :]
                x = x + gate * (candidate_x - x)
                if global_bus is not None and candidate_bus is not None:
                    global_bus = global_bus.astype(candidate_x.dtype) + gate * (
                        candidate_bus.astype(candidate_x.dtype) - global_bus.astype(candidate_x.dtype)
                    )
        return self.final_norm(x)

    def loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        qat: bool = False,
        token_weights: mx.array | None = None,
    ) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids, qat=qat).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        flat_weights = token_weights.reshape(-1).astype(mx.float32) if token_weights is not None else None
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return cross_entropy_loss(logits, y, flat_weights)

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        denom = (
            mx.maximum(mx.sum(flat_weights), mx.array(1e-6, dtype=mx.float32))
            if flat_weights is not None
            else mx.array(float(n), dtype=mx.float32)
        )
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            if flat_weights is None:
                loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum").astype(mx.float32)
            else:
                chunk_losses = nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="none").astype(mx.float32)
                loss_sum = loss_sum + mx.sum(chunk_losses * flat_weights[s:e])
        return loss_sum / denom

    def training_loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        loss = self.loss(
            input_ids,
            target_ids,
            qat=True,
            token_weights=train_token_weights(input_ids, target_ids),
        )
        if LFQAT_KL_WEIGHT > 0.0 and self.logit_chunk_tokens <= 0:
            tau = mx.array(max(LFQAT_TEMPERATURE, 1e-3), dtype=mx.float32)
            student = self.softcap(self(input_ids, qat=True).reshape(-1, self.tok_emb.weight.shape[1]) @ self.tok_emb.weight.astype(COMPUTE_DTYPE).T).astype(mx.float32)
            teacher = mx.stop_gradient(self.softcap(self(input_ids, qat=False).reshape(-1, self.tok_emb.weight.shape[1]) @ self.tok_emb.weight.astype(COMPUTE_DTYPE).T).astype(mx.float32))
            teacher_probs = mx.softmax(teacher / tau, axis=-1)
            student_log_probs = nn.log_softmax(student / tau, axis=-1)
            teacher_log_probs = mx.log(mx.maximum(teacher_probs, mx.array(1e-6, dtype=mx.float32)))
            loss = loss + mx.array(LFQAT_KL_WEIGHT, dtype=mx.float32) * (tau * tau) * mx.mean(mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1))
        flat_params = dict(tree_flatten(self.parameters()))
        return (
            loss
            + compression_aware_alignment_loss(flat_params)
            + int8_aware_alignment_loss(flat_params)
            + qer_aware_alignment_loss(flat_params)
            + lfqat_fisher_alignment_loss(flat_params)
        )

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k != self.embed_key and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k != self.embed_key and k not in self.matrix_keys
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
QUANT_FORMAT = os.environ.get("QUANT_FORMAT", "int8_clean_per_row_v1").strip().lower()
INT4_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("INT4_NAME_PATTERNS", "").split(",")
    if pattern
)
INT4_CODEBOOK_NAME = os.environ.get("INT4_CODEBOOK_NAME", "normal16").strip().lower()
INT4_BLOCK_SIZE = int(os.environ.get("INT4_BLOCK_SIZE", 64))
INT4_CLIP_PERCENTILE = float(os.environ.get("INT4_CLIP_PERCENTILE", 99.9))
INT4_CLIP_Q = INT4_CLIP_PERCENTILE / 100.0
LOWRANK_ERROR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("LOWRANK_ERROR_NAME_PATTERNS", "").split(",")
    if pattern
)
LOWRANK_ERROR_RANK = int(os.environ.get("LOWRANK_ERROR_RANK", 0))


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)

def should_keep_float_array(name: str) -> bool:
    return any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS) or any(
        pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS
    )


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS):
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def should_use_int4_array(name: str, arr: mx.array) -> bool:
    return (
        QUANT_FORMAT in {"mixed_int4_int8_packed_v2", "mixed_codebook_int4_int8_packed_v1"}
        and arr.ndim == 2
        and any(pattern in name for pattern in INT4_NAME_PATTERNS)
    )


def should_use_lowrank_error_array(name: str, arr: mx.array) -> bool:
    return LOWRANK_ERROR_RANK > 0 and arr.ndim == 2 and any(pattern in name for pattern in LOWRANK_ERROR_NAME_PATTERNS)


def pack_int4_values(values: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(np.asarray(values, dtype=np.int8).reshape(-1))
    if flat.size % 2:
        flat = np.concatenate((flat, np.zeros((1,), dtype=np.int8)))
    unsigned = np.asarray(flat + 8, dtype=np.uint8)
    packed = unsigned[0::2] | (unsigned[1::2] << 4)
    return np.ascontiguousarray(packed)


def unpack_int4_values(packed: np.ndarray, count: int) -> np.ndarray:
    packed_u8 = np.asarray(packed, dtype=np.uint8).reshape(-1)
    values = np.empty((packed_u8.size * 2,), dtype=np.int8)
    values[0::2] = (packed_u8 & 0x0F).astype(np.int8) - 8
    values[1::2] = (packed_u8 >> 4).astype(np.int8) - 8
    return values[:count]


def quantize_float_array_int8(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_float_array_int4_blockwise(arr: mx.array) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    f32 = _np_float32(arr)
    if f32.ndim != 2:
        raise ValueError(f"int4 blockwise quantization only supports 2D tensors, got shape={f32.shape}")
    if INT4_BLOCK_SIZE <= 0:
        raise ValueError(f"INT4_BLOCK_SIZE must be positive, got {INT4_BLOCK_SIZE}")

    rows, cols = f32.shape
    blocks_per_row = (cols + INT4_BLOCK_SIZE - 1) // INT4_BLOCK_SIZE
    padded_cols = blocks_per_row * INT4_BLOCK_SIZE
    if padded_cols != cols:
        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = f32
    else:
        padded = f32
    reshaped = padded.reshape(rows, blocks_per_row, INT4_BLOCK_SIZE)
    clip_abs = (
        np.quantile(np.abs(reshaped), INT4_CLIP_Q, axis=2)
        if reshaped.size
        else np.empty((rows, blocks_per_row), dtype=np.float32)
    )
    scale = np.maximum(clip_abs / 7.0, 1.0 / 7.0).astype(np.float32, copy=False)
    clipped = np.clip(reshaped, -clip_abs[..., None], clip_abs[..., None])
    q = np.clip(np.round(clipped / scale[..., None]), -7, 7).astype(np.int8, copy=False)
    meta = {
        "scheme": "per_row_block_int4",
        "axis": 0,
        "shape": [rows, cols],
        "block_size": INT4_BLOCK_SIZE,
    }
    return (
        pack_int4_values(q),
        np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)),
        meta,
    )


def quantize_float_array_codebook_int4_blockwise(arr: mx.array) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    signed, scale, meta = quantize_codebook_int4_blockwise(
        _np_float32(arr),
        INT4_BLOCK_SIZE,
        clip_q=INT4_CLIP_Q,
        codebook_name=INT4_CODEBOOK_NAME,
    )
    return pack_int4_values(signed), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)), meta


def dequantize_quantized_array(q: np.ndarray, scale: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    if meta.get("scheme") == "per_row_block_int4":
        rows, cols = (int(x) for x in meta["shape"])
        block_size = int(meta["block_size"])
        blocks_per_row = (cols + block_size - 1) // block_size
        q_int = unpack_int4_values(np.asarray(q, dtype=np.uint8), rows * blocks_per_row * block_size)
        q_int = q_int.reshape(rows, blocks_per_row, block_size)
        out = q_int.astype(np.float32) * np.asarray(scale, dtype=np.float32).reshape(rows, blocks_per_row, 1)
        return np.ascontiguousarray(out.reshape(rows, blocks_per_row * block_size)[:, :cols])
    if meta.get("scheme") == "per_row_block_codebook_int4":
        rows, cols = (int(x) for x in meta["shape"])
        block_size = int(meta["block_size"])
        blocks_per_row = (cols + block_size - 1) // block_size
        signed = unpack_int4_values(np.asarray(q, dtype=np.uint8), rows * blocks_per_row * block_size)
        return dequantize_codebook_int4_blockwise(signed, np.asarray(scale, dtype=np.float32), meta)
    q_np = np.asarray(q, dtype=np.int8)
    scale32 = np.asarray(scale, dtype=np.float32)
    if meta.get("scheme") == "per_row" or scale32.ndim > 0:
        return np.ascontiguousarray(q_np.astype(np.float32) * scale32.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1)))
    return np.ascontiguousarray(q_np.astype(np.float32) * float(scale32))


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    lowrank_left: dict[str, np.ndarray] = {}
    lowrank_right: dict[str, np.ndarray] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if should_keep_float_array(name) or int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        if QUANT_FORMAT == "mixed_codebook_int4_int8_packed_v1" and should_use_int4_array(name, arr):
            q, s, meta = quantize_float_array_codebook_int4_blockwise(arr)
        elif should_use_int4_array(name, arr):
            q, s, meta = quantize_float_array_int4_blockwise(arr)
        else:
            q, s = quantize_float_array_int8(arr)
            meta = {"scheme": "per_row", "axis": 0} if s.ndim > 0 else {}
        if should_use_lowrank_error_array(name, arr):
            left, right, kept_rank = compute_lowrank_residual(_np_float32(arr), dequantize_quantized_array(q, s, meta), LOWRANK_ERROR_RANK)
            if kept_rank > 0:
                lowrank_left[name] = left
                lowrank_right[name] = right
                meta = dict(meta)
                meta["lowrank_rank"] = kept_rank
                stats["int8_payload_bytes"] += int(left.nbytes + right.nbytes)
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
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


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    lowrank_left = quant_obj.get("lowrank_left", {})
    lowrank_right = quant_obj.get("lowrank_right", {})
    for name, q in quant_obj["quantized"].items():
        meta = qmeta.get(name, {})
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        out_arr = dequantize_quantized_array(np.asarray(q), scale, meta)
        if name in lowrank_left:
            out_arr = apply_lowrank_residual(out_arr, np.asarray(lowrank_left[name]), np.asarray(lowrank_right[name]))
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        total_loss = total_loss + compiled_loss(x, y).astype(mx.float32) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
    total_loss = total_loss / total_tokens
    mx.eval(total_loss)
    val_loss = float(total_loss.item())
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = submission_code_text(Path(__file__), Path(__file__).with_name("quant_reconstruction.py"))
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    if args.val_max_tokens > 0:
        usable_val_tokens = min(((args.val_max_tokens // args.train_seq_len) * args.train_seq_len) + 1, val_tokens.size)
        if usable_val_tokens <= 1:
            raise ValueError(
                f"VAL_MAX_TOKENS={args.val_max_tokens} is too small for TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        if usable_val_tokens < val_tokens.size:
            log(
                f"WARNING: val_loader:subset tokens:{usable_val_tokens - 1}/{val_tokens.size - 1} "
                f"enabled via VAL_MAX_TOKENS={args.val_max_tokens}"
            )
            val_tokens = val_tokens[:usable_val_tokens]

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )
    configure_global_bus_boundary_lut(has_leading_space_lut, is_boundary_token_lut)
    configure_train_token_weight_luts(base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    flat_state_keys = [name for name, _ in tree_flatten(model.state)]
    if args.init_model_path:
        loaded = mx.load(args.init_model_path)
        expected = set(flat_state_keys)
        found = set(loaded.keys())
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if extra or (missing and not ALLOW_INIT_MISSING_KEYS):
            raise ValueError(
                f"INIT_MODEL_PATH mismatch missing={missing[:5]} extra={extra[:5]} "
                f"(missing={len(missing)} extra={len(extra)})"
            )
        model.update(tree_unflatten([(name, loaded[name]) for name in flat_state_keys if name in loaded]))
        if missing:
            log(
                f"WARNING: init_model_path partial load enabled missing={len(missing)} "
                f"keys sample:{','.join(missing[:5])}"
            )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = (
        nn.value_and_grad(model, lambda x, y: model.training_loss(x, y))
        if lfqat_enabled() or int8_aware_enabled() or qer_aware_enabled()
        else mx.compile(nn.value_and_grad(model, lambda x, y: model.training_loss(x, y)), inputs=model.state, outputs=model.state)
    )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    if args.init_model_path:
        log(f"init_model_path:{args.init_model_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"unique_layers:{args.num_unique_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f} "
        f"lr_schedule:{args.lr_schedule} lr_warmup_iters:{args.lr_warmup_iters} min_lr_scale:{args.min_lr_scale:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"quant_format:{QUANT_FORMAT} int4_patterns:{','.join(INT4_NAME_PATTERNS) if INT4_NAME_PATTERNS else '-'} "
        f"int4_block_size:{INT4_BLOCK_SIZE} int4_codebook:{INT4_CODEBOOK_NAME}"
    )
    log(
        f"train_compression_aware_weight:{TRAIN_COMPRESSION_AWARE_WEIGHT} "
        f"train_compression_aware_patterns:{','.join(TRAIN_COMPRESSION_AWARE_NAME_PATTERNS) if TRAIN_COMPRESSION_AWARE_NAME_PATTERNS else '-'} "
        f"train_compression_aware_block_size:{TRAIN_COMPRESSION_AWARE_BLOCK_SIZE}"
    )
    log(
        f"train_int8_aware_weight:{TRAIN_INT8_AWARE_WEIGHT} "
        f"train_int8_aware_patterns:{','.join(TRAIN_INT8_AWARE_NAME_PATTERNS) if TRAIN_INT8_AWARE_NAME_PATTERNS else '-'} "
        f"int8_clip_percentile:{INT8_CLIP_PERCENTILE}"
    )
    log(
        f"train_token_weight_mode:{TRAIN_TOKEN_WEIGHT_MODE} "
        f"train_token_weight_power:{TRAIN_TOKEN_WEIGHT_POWER:.3f}"
    )
    log(
        f"train_qer_aware_weight:{TRAIN_QER_AWARE_WEIGHT} train_qer_aware_rank:{TRAIN_QER_AWARE_RANK} "
        f"train_qer_aware_quant_format:{TRAIN_QER_AWARE_QUANT_FORMAT} "
        f"train_qer_aware_codebook:{TRAIN_QER_AWARE_CODEBOOK_NAME} "
        f"train_qer_aware_patterns:{','.join(TRAIN_QER_AWARE_NAME_PATTERNS) if TRAIN_QER_AWARE_NAME_PATTERNS else '-'}"
    )
    log(
        f"train_grad_only_patterns:{','.join(TRAIN_GRAD_ONLY_NAME_PATTERNS) if TRAIN_GRAD_ONLY_NAME_PATTERNS else '-'} "
        f"train_grad_skip_patterns:{','.join(TRAIN_GRAD_SKIP_NAME_PATTERNS) if TRAIN_GRAD_SKIP_NAME_PATTERNS else '-'}"
    )
    log(
        f"train_qat_patterns:{','.join(TRAIN_QAT_NAME_PATTERNS) if TRAIN_QAT_NAME_PATTERNS else '-'} "
        f"train_qat_block_size:{TRAIN_QAT_BLOCK_SIZE}"
    )
    log(
        f"lfqat_enabled:{lfqat_enabled()} lfqat_kl_weight:{LFQAT_KL_WEIGHT} "
        f"lfqat_fisher_weight:{LFQAT_FISHER_WEIGHT} lfqat_temperature:{LFQAT_TEMPERATURE} "
        f"lfqat_prob:{LFQAT_MIN_PROB}->{LFQAT_MAX_PROB} lfqat_steps:{LFQAT_START_STEP}->{LFQAT_FULL_STEP}"
    )
    log(f"int8_aware_enabled:{int8_aware_enabled()}")
    log(f"qer_aware_enabled:{qer_aware_enabled()}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(
        f"compute_dtype:{COMPUTE_DTYPE} "
        f"train_compile:{not (lfqat_enabled() or int8_aware_enabled() or qer_aware_enabled())} "
        f"eval_compile:True"
    )
    log(
        f"global_bus_enabled:{model.global_bus_enabled} "
        f"global_bus_read_layers:{compact_layer_selector(layer_selector_mask(GLOBAL_BUS_READ_LAYERS_SPEC, args.num_layers))} "
        f"global_bus_write_layers:{compact_layer_selector(layer_selector_mask(GLOBAL_BUS_WRITE_LAYERS_SPEC, args.num_layers))} "
        f"global_bus_summary_mode:{GLOBAL_BUS_SUMMARY_MODE} "
        f"global_bus_tail_weight:{GLOBAL_BUS_TAIL_WEIGHT:.3f} "
        f"allow_init_missing_keys:{ALLOW_INIT_MISSING_KEYS}"
    )
    log(
        f"logical_block_order:{compact_logical_block_order(logical_block_order(args.num_layers, args.num_unique_layers, LOGICAL_BLOCK_ORDER_SPEC), args.num_unique_layers)}"
    )
    log(
        f"second_pass_enabled:{model.second_pass_enabled} "
        f"second_pass_layers:{compact_layer_selector(second_pass_decoder_mask(args.num_layers, model.num_encoder_layers, SECOND_PASS_LAYERS_SPEC)) if model.second_pass_enabled and any(second_pass_decoder_mask(args.num_layers, model.num_encoder_layers, SECOND_PASS_LAYERS_SPEC)) else '-'} "
        f"second_pass_gate_init:{SECOND_PASS_GATE_INIT:.3f} "
        f"second_pass_skip_init:{SECOND_PASS_SKIP_INIT:.3f}"
    )
    log(
        f"cross_skip_router_enabled:{model.cross_skip_router_enabled} "
        f"cross_skip_router_match_init:{CROSS_SKIP_ROUTER_MATCH_INIT:.3f} "
        f"cross_skip_router_other_init:{CROSS_SKIP_ROUTER_OTHER_INIT:.3f} "
        f"cross_skip_router_decoder_layers:{compact_layer_selector(cross_skip_decoder_mask(args.num_layers, model.num_encoder_layers, CROSS_SKIP_ROUTER_DECODER_LAYERS_SPEC)) if model.cross_skip_router_enabled else '-'} "
        f"cross_skip_router_source_layers:{compact_layer_selector(cross_skip_source_mask(args.num_layers, model.num_encoder_layers, CROSS_SKIP_ROUTER_SOURCE_LAYERS_SPEC)) if model.cross_skip_router_enabled else '-'}"
    )
    log(
        f"smear_gate_enabled:{model.smear_gate_enabled} "
        f"smear_gate_init:{SMEAR_GATE_INIT:.3f} "
        f"bigram_hash_enabled:{model.bigram_hash_enabled} "
        f"bigram_hash_buckets:{BIGRAM_HASH_BUCKETS} "
        f"bigram_hash_dim:{BIGRAM_HASH_DIM} "
        f"bigram_hash_init_std:{BIGRAM_HASH_INIT_STD:.4f}"
    )
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        LFQAT_RUNTIME["prob"] = lfqat_prob_for_step(step)
        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        grads = filter_grad_tree(clip_grad_tree(grads, args.grad_clip_norm))
        update_lfqat_fisher(grads)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f} "
                f"lfqat_prob:{LFQAT_RUNTIME['prob']:.3f}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
