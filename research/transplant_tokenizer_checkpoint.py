#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import sentencepiece as spm


SPECIAL_IDS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warm-start a new SentencePiece vocabulary by transplanting a source MLX checkpoint body."
    )
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--source-tokenizer", required=True)
    p.add_argument("--target-tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-json", default="")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--init-std", type=float, default=0.005)
    return p.parse_args()


def token_surface_text(sp: spm.SentencePieceProcessor, token_id: int) -> str:
    if token_id in SPECIAL_IDS:
        return ""
    text = sp.decode([token_id])
    if text:
        return text
    piece = sp.id_to_piece(token_id)
    if not piece or piece.startswith("<"):
        return ""
    return piece.replace("▁", " ")


def build_transplanted_embedding(
    *,
    source_emb: np.ndarray,
    source_sp: spm.SentencePieceProcessor,
    target_sp: spm.SentencePieceProcessor,
    init_std: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    target_vocab = int(target_sp.vocab_size())
    dim = int(source_emb.shape[1])
    out = rng.normal(0.0, init_std, size=(target_vocab, dim)).astype(np.float32)
    source_piece_to_id = {source_sp.id_to_piece(i): i for i in range(int(source_sp.vocab_size()))}
    copied_special = 0
    copied_exact_piece = 0
    mapped = 0
    random_init = 0
    total_source_pieces = 0

    for token_id in range(target_vocab):
        if token_id in SPECIAL_IDS and token_id < int(source_emb.shape[0]):
            out[token_id] = source_emb[token_id]
            copied_special += 1
            continue

        target_piece = target_sp.id_to_piece(token_id)
        source_piece_id = source_piece_to_id.get(target_piece) if target_piece else None
        if source_piece_id is not None and 0 <= source_piece_id < int(source_emb.shape[0]):
            out[token_id] = source_emb[source_piece_id]
            copied_exact_piece += 1
            mapped += 1
            total_source_pieces += 1
            continue

        text = token_surface_text(target_sp, token_id)
        if not text:
            random_init += 1
            continue

        source_ids = source_sp.encode(text, out_type=int)
        source_ids = [i for i in source_ids if 0 <= i < int(source_emb.shape[0])]
        if not source_ids:
            random_init += 1
            continue

        out[token_id] = source_emb[source_ids].mean(axis=0)
        mapped += 1
        total_source_pieces += len(source_ids)

    summary = {
        "target_vocab": target_vocab,
        "source_vocab": int(source_emb.shape[0]),
        "copied_special": copied_special,
        "exact_piece_rows": copied_exact_piece,
        "averaged_rows": mapped - copied_exact_piece,
        "mapped_rows": mapped,
        "random_rows": random_init,
        "mapping_coverage": float((copied_special + mapped) / max(target_vocab, 1)),
        "mean_source_pieces_per_mapped_row": float(total_source_pieces / max(mapped, 1)),
    }
    return out, summary


def main() -> None:
    args = parse_args()
    source_checkpoint = Path(args.source_checkpoint).resolve()
    source_tokenizer = Path(args.source_tokenizer).resolve()
    target_tokenizer = Path(args.target_tokenizer).resolve()
    output_path = Path(args.output).resolve()

    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if not source_tokenizer.is_file():
        raise FileNotFoundError(source_tokenizer)
    if not target_tokenizer.is_file():
        raise FileNotFoundError(target_tokenizer)

    flat = dict(mx.load(str(source_checkpoint)).items())
    if "tok_emb.weight" not in flat:
        raise KeyError("source checkpoint missing tok_emb.weight")

    source_sp = spm.SentencePieceProcessor(model_file=str(source_tokenizer))
    target_sp = spm.SentencePieceProcessor(model_file=str(target_tokenizer))

    source_emb = np.asarray(flat["tok_emb.weight"].astype(mx.float32))
    target_emb, summary = build_transplanted_embedding(
        source_emb=source_emb,
        source_sp=source_sp,
        target_sp=target_sp,
        init_std=args.init_std,
        seed=args.seed,
    )
    flat["tok_emb.weight"] = mx.array(target_emb, dtype=flat["tok_emb.weight"].dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output_path), **flat)

    summary.update(
        {
            "source_checkpoint": str(source_checkpoint),
            "source_tokenizer": str(source_tokenizer),
            "target_tokenizer": str(target_tokenizer),
            "output": str(output_path),
            "output_bytes": int(output_path.stat().st_size),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
