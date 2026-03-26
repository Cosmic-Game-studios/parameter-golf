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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collapse a full-depth MLX checkpoint into a shared-depth checkpoint.")
    p.add_argument("--input", required=True, help="Source .npz checkpoint.")
    p.add_argument("--output", required=True, help="Destination .npz checkpoint.")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--source-num-unique-layers", type=int, required=True)
    p.add_argument("--target-num-unique-layers", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--num-kv-heads", type=int, required=True)
    p.add_argument("--mlp-mult", type=int, required=True)
    p.add_argument("--logit-softcap", type=float, default=30.0)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--tied-embed-init-std", type=float, default=0.005)
    p.add_argument("--qk-gain-init", type=float, default=1.5)
    p.add_argument(
        "--mode",
        choices=("average_modulo", "first_cycle", "last_cycle", "contiguous_partition"),
        default="average_modulo",
        help="How to map source block families into the target shared blocks.",
    )
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


def source_block_indices_for_target(
    *,
    target_idx: int,
    source_unique_layers: int,
    target_unique_layers: int,
    mode: str,
) -> list[int]:
    if mode == "first_cycle":
        return [target_idx]
    if mode == "last_cycle":
        start = source_unique_layers - target_unique_layers
        return [start + target_idx]
    if mode == "average_modulo":
        return list(range(target_idx, source_unique_layers, target_unique_layers))
    if mode == "contiguous_partition":
        start = (target_idx * source_unique_layers) // target_unique_layers
        end = ((target_idx + 1) * source_unique_layers) // target_unique_layers
        return list(range(start, max(start + 1, end)))
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    from train_gpt_mlx import GPT

    args = build_parser().parse_args()
    if args.target_num_unique_layers <= 0:
        raise ValueError("--target-num-unique-layers must be positive")
    if args.source_num_unique_layers < args.target_num_unique_layers:
        raise ValueError("source unique layers must be >= target unique layers")
    if args.source_num_unique_layers > args.num_layers or args.target_num_unique_layers > args.num_layers:
        raise ValueError("unique layers cannot exceed total layers")
    src = mx.load(str(Path(args.input).expanduser().resolve()))
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.target_num_unique_layers,
        dim=args.dim,
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

    collapsed: dict[str, mx.array] = {}
    for name, tgt_arr in tgt_state.items():
        parsed = parse_block_name(name)
        if parsed is None:
            if name not in src_np_by_name:
                raise KeyError(f"missing non-block tensor in source checkpoint: {name}")
            src_np = src_np_by_name[name]
            if src_np.shape != tgt_arr.shape:
                raise ValueError(f"shape mismatch for {name}: src={src_np.shape} tgt={tuple(tgt_arr.shape)}")
            collapsed[name] = to_target_dtype(src_np, tgt_arr.dtype)
            continue

        target_block_idx, suffix = parsed
        source_indices = source_block_indices_for_target(
            target_idx=target_block_idx,
            source_unique_layers=args.source_num_unique_layers,
            target_unique_layers=args.target_num_unique_layers,
            mode=args.mode,
        )
        source_tensors: list[np.ndarray] = []
        for source_idx in source_indices:
            source_name = f"blocks.{source_idx}.{suffix}"
            if source_name not in src_np_by_name:
                raise KeyError(f"missing source block tensor: {source_name}")
            source_tensors.append(src_np_by_name[source_name].astype(np.float32, copy=False))
        merged = np.mean(np.stack(source_tensors, axis=0), axis=0, dtype=np.float32)
        collapsed[name] = to_target_dtype(merged, tgt_arr.dtype)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output), **collapsed)
    print(output)


if __name__ == "__main__":
    main()
