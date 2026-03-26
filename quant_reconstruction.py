from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np


def submission_code_text(*paths: str | Path) -> str:
    return "\n\n".join(Path(path).read_text(encoding="utf-8") for path in paths)


def clamp_lowrank_rank(rows: int, cols: int, rank: int) -> int:
    return max(0, min(int(rank), int(rows), int(cols)))


def compute_lowrank_residual(
    target: np.ndarray,
    approx: np.ndarray,
    rank: int,
    store_dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray, int]:
    target32 = np.asarray(target, dtype=np.float32)
    approx32 = np.asarray(approx, dtype=np.float32)
    if target32.ndim != 2 or approx32.shape != target32.shape:
        raise ValueError(
            f"low-rank residual expects matching 2D arrays, got target={target32.shape} approx={approx32.shape}"
        )
    kept_rank = clamp_lowrank_rank(target32.shape[0], target32.shape[1], rank)
    if kept_rank == 0:
        return (
            np.empty((target32.shape[0], 0), dtype=store_dtype),
            np.empty((0, target32.shape[1]), dtype=store_dtype),
            0,
        )
    residual = target32 - approx32
    if not np.isfinite(residual).all():
        raise ValueError("low-rank residual got non-finite values")
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    left = np.ascontiguousarray((u[:, :kept_rank] * s[:kept_rank]).astype(store_dtype, copy=False))
    right = np.ascontiguousarray(vt[:kept_rank, :].astype(store_dtype, copy=False))
    return left, right, kept_rank


def apply_lowrank_residual(base: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    base32 = np.asarray(base, dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return np.ascontiguousarray(base32)
    return np.ascontiguousarray(base32 + np.asarray(left, dtype=np.float32) @ np.asarray(right, dtype=np.float32))


@lru_cache(maxsize=1)
def normal_codebook_4bit() -> np.ndarray:
    dist = NormalDist()
    probs = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    values = np.asarray([dist.inv_cdf(float(p)) for p in probs], dtype=np.float32)
    values /= np.max(np.abs(values))
    return np.ascontiguousarray(values)


@lru_cache(maxsize=1)
def uniform_codebook_4bit() -> np.ndarray:
    return np.ascontiguousarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))


@lru_cache(maxsize=1)
def mulaw_codebook_4bit(mu: float = 8.0) -> np.ndarray:
    levels = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    mag = (np.power(1.0 + mu, np.abs(levels), dtype=np.float32) - 1.0) / mu
    values = np.sign(levels) * mag
    return np.ascontiguousarray(values.astype(np.float32, copy=False))


def quantile_codebook_4bit(normalized_values: np.ndarray) -> np.ndarray:
    flat = np.asarray(normalized_values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return normal_codebook_4bit()
    probs = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    values = np.quantile(flat, probs).astype(np.float32, copy=False)
    values = np.clip(values, -1.0, 1.0)
    values = np.maximum.accumulate(values)
    values[0] = min(values[0], -1e-4)
    values[-1] = max(values[-1], 1e-4)
    return np.ascontiguousarray(values)


def lloyd_codebook_4bit(normalized_values: np.ndarray, steps: int = 24) -> np.ndarray:
    flat = np.asarray(normalized_values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return normal_codebook_4bit()
    flat = np.clip(flat, -1.0, 1.0)
    codebook = quantile_codebook_4bit(flat)
    for _ in range(max(1, int(steps))):
        boundaries = 0.5 * (codebook[:-1] + codebook[1:])
        buckets = np.digitize(flat, boundaries, right=False)
        new_codebook = codebook.copy()
        for idx in range(16):
            members = flat[buckets == idx]
            if members.size:
                new_codebook[idx] = float(np.mean(members, dtype=np.float64))
        new_codebook = np.clip(new_codebook, -1.0, 1.0)
        new_codebook = np.maximum.accumulate(new_codebook)
        new_codebook[0] = min(new_codebook[0], -1e-4)
        new_codebook[-1] = max(new_codebook[-1], 1e-4)
        if np.allclose(new_codebook, codebook, atol=1e-5):
            codebook = new_codebook
            break
        codebook = new_codebook
    return np.ascontiguousarray(codebook.astype(np.float32, copy=False))


def named_codebook_4bit(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key == "normal16":
        return normal_codebook_4bit()
    if key == "uniform16":
        return uniform_codebook_4bit()
    if key == "mulaw16":
        return mulaw_codebook_4bit()
    raise ValueError(f"unsupported fixed 4-bit codebook {name!r}")


def quantize_codebook_int4_blockwise(
    target: np.ndarray,
    block_size: int,
    clip_q: float = 0.999,
    codebook: np.ndarray | None = None,
    codebook_name: str = "normal16",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target32 = np.asarray(target, dtype=np.float32)
    if target32.ndim != 2:
        raise ValueError(f"codebook int4 quantization only supports 2D tensors, got shape={target32.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    rows, cols = target32.shape
    blocks_per_row = (cols + block_size - 1) // block_size
    padded_cols = blocks_per_row * block_size
    if padded_cols != cols:
        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = target32
    else:
        padded = target32
    reshaped = padded.reshape(rows, blocks_per_row, block_size)
    clip_abs = (
        np.quantile(np.abs(reshaped), clip_q, axis=2)
        if reshaped.size
        else np.empty((rows, blocks_per_row), dtype=np.float32)
    )
    scale = np.maximum(clip_abs, 1e-6).astype(np.float32, copy=False)
    clipped = np.clip(reshaped, -clip_abs[..., None], clip_abs[..., None])
    normalized = clipped / scale[..., None]
    if codebook is None:
        if codebook_name.strip().lower() == "quantile16":
            codebook32 = quantile_codebook_4bit(normalized)
        elif codebook_name.strip().lower() == "lloyd16":
            codebook32 = lloyd_codebook_4bit(normalized)
        else:
            codebook32 = named_codebook_4bit(codebook_name)
    else:
        codebook32 = np.asarray(codebook, dtype=np.float32)
    if codebook32.shape != (16,):
        raise ValueError(f"expected 16-entry codebook, got shape={codebook32.shape}")
    distances = np.abs(normalized[..., None] - codebook32.reshape(1, 1, 1, 16))
    indices = np.argmin(distances, axis=-1).astype(np.int8, copy=False)
    signed = np.ascontiguousarray(indices - 8)
    meta = {
        "scheme": "per_row_block_codebook_int4",
        "axis": 0,
        "shape": [rows, cols],
        "block_size": block_size,
        "codebook": codebook_name.strip().lower(),
    }
    if codebook is not None or meta["codebook"] in {"quantile16", "lloyd16"}:
        meta["codebook_values"] = np.ascontiguousarray(codebook32.astype(np.float16, copy=False))
    return signed, np.ascontiguousarray(scale.astype(np.float16, copy=False)), meta


def dequantize_codebook_int4_blockwise(
    signed_values: np.ndarray,
    scale: np.ndarray,
    meta: dict[str, object],
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = (int(x) for x in meta["shape"])
    block_size = int(meta["block_size"])
    blocks_per_row = (cols + block_size - 1) // block_size
    if codebook is not None:
        codebook32 = np.asarray(codebook, dtype=np.float32)
    elif "codebook_values" in meta:
        codebook32 = np.asarray(meta["codebook_values"], dtype=np.float32)
    else:
        codebook32 = named_codebook_4bit(str(meta.get("codebook", "normal16")))
    indices = np.asarray(signed_values, dtype=np.int8).reshape(rows, blocks_per_row, block_size).astype(np.int16) + 8
    values = codebook32[indices]
    out = values.astype(np.float32, copy=False) * np.asarray(scale, dtype=np.float32).reshape(rows, blocks_per_row, 1)
    return np.ascontiguousarray(out.reshape(rows, blocks_per_row * block_size)[:, :cols])
