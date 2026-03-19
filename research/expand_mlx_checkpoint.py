#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_gpt_mlx import GPT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Expand an MLX GPT checkpoint to a larger model dimension.")
    p.add_argument("--input", required=True, help="Source .npz checkpoint.")
    p.add_argument("--output", required=True, help="Destination .npz checkpoint.")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--num-unique-layers", type=int, required=True)
    p.add_argument("--source-dim", type=int, required=True)
    p.add_argument("--target-dim", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--num-kv-heads", type=int, required=True)
    p.add_argument("--mlp-mult", type=int, required=True)
    p.add_argument("--logit-softcap", type=float, default=30.0)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--tied-embed-init-std", type=float, default=0.005)
    p.add_argument("--qk-gain-init", type=float, default=1.5)
    return p


def to_numpy_safe(arr: mx.array) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        return np.array(arr.astype(mx.float16), copy=True)
    return np.array(arr, copy=True)


def to_target_dtype(arr: np.ndarray, dtype: mx.Dtype) -> mx.array:
    out = mx.array(arr)
    if out.dtype != dtype:
        out = out.astype(dtype)
    return out


def copy_into_target(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    out = np.array(tgt, copy=True)
    if src.ndim != tgt.ndim:
        raise ValueError(f"rank mismatch: src={src.shape} tgt={tgt.shape}")
    slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt.shape))
    out[slices] = src[slices]
    return out


def parse_block_name(name: str) -> tuple[int, str] | None:
    if not name.startswith("blocks."):
        return None
    parts = name.split(".")
    if len(parts) < 4:
        return None
    try:
        return int(parts[1]), ".".join(parts[2:])
    except ValueError:
        return None


def expanded_from_last(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    out = np.array(tgt, copy=True)
    common = min(src.shape[0], tgt.shape[0])
    if common > 0:
        out[:common] = src[:common]
    if tgt.shape[0] > src.shape[0] and src.shape[0] > 0:
        out[common:] = src[src.shape[0] - 1]
    return out


def main() -> None:
    args = build_parser().parse_args()
    if args.target_dim < args.source_dim:
        raise ValueError("--target-dim must be >= --source-dim")

    src = mx.load(str(Path(args.input).expanduser().resolve()))
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        dim=args.target_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=0,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    tgt_state = dict(tree_flatten(model.state))

    src_np_by_name = {name: to_numpy_safe(arr) for name, arr in src.items()}
    src_block_map: dict[tuple[int, str], np.ndarray] = {}
    for name, arr in src_np_by_name.items():
        parsed = parse_block_name(name)
        if parsed is not None:
            src_block_map[parsed] = arr

    expanded: dict[str, mx.array] = {}
    for name, tgt_arr in tgt_state.items():
        tgt_np = to_numpy_safe(tgt_arr)
        if name in src_np_by_name:
            src_np = src_np_by_name[name]
            if (
                src_np.ndim >= 1
                and tgt_np.ndim == src_np.ndim
                and tgt_np.shape[1:] == src_np.shape[1:]
                and tgt_np.shape[0] > src_np.shape[0]
                and name in {"attn_scales", "mlp_scales", "resid_mixes"}
            ):
                expanded_np = expanded_from_last(src_np, tgt_np)
            else:
                expanded_np = copy_into_target(src_np, tgt_np)
            expanded[name] = to_target_dtype(expanded_np, tgt_arr.dtype)
            continue
        parsed = parse_block_name(name)
        if parsed is not None:
            block_idx, suffix = parsed
            fallback = src_block_map.get((block_idx, suffix))
            if fallback is None:
                suffix_candidates = [(idx, arr) for (idx, key), arr in src_block_map.items() if key == suffix]
                if suffix_candidates:
                    fallback = max(suffix_candidates, key=lambda item: item[0])[1]
            if fallback is not None:
                expanded[name] = to_target_dtype(copy_into_target(fallback, tgt_np), tgt_arr.dtype)
                continue
        expanded[name] = tgt_arr

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output), **expanded)
    print(output)


if __name__ == "__main__":
    main()
