#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import sentencepiece as spm


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_SHARD_TOKENS = 100_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local SentencePiece tokenizer from existing SP1024 shards and retokenize them."
    )
    parser.add_argument(
        "--input-data-path",
        default="data/datasets/fineweb10B_sp1024",
        help="Existing shard directory with fineweb_train_*.bin and fineweb_val_*.bin.",
    )
    parser.add_argument(
        "--input-tokenizer-path",
        default="data/tokenizers/fineweb_1024_bpe.model",
        help="SentencePiece model used to decode the existing shards.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where the new tokenizer and dataset export will be written.",
    )
    parser.add_argument("--dataset-suffix", required=True, help="Dataset suffix, for example spu4096_local.")
    parser.add_argument("--vocab-size", type=int, required=True, help="New tokenizer vocab size.")
    parser.add_argument(
        "--model-type",
        choices=("unigram", "bpe"),
        default="unigram",
        help="SentencePiece model type for the new tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-train-docs",
        type=int,
        default=40_000,
        help="How many train docs to use for tokenizer training.",
    )
    parser.add_argument(
        "--train-doc-limit",
        type=int,
        default=None,
        help="Optional train-doc export cap for faster local experiments.",
    )
    parser.add_argument(
        "--val-doc-limit",
        type=int,
        default=None,
        help="Optional val-doc export cap for faster local experiments.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help="Target number of tokens per output shard.",
    )
    return parser.parse_args()


def load_data_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * np.dtype("<i4").itemsize)
    if tokens.size != num_tokens:
        raise ValueError(f"short read for {path}")
    return tokens.astype(np.int32, copy=False)


def iter_docs_from_shards(
    shard_paths: list[Path],
    sp: spm.SentencePieceProcessor,
    *,
    doc_limit: int | None = None,
) -> Iterator[str]:
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("input tokenizer must define bos_id")
    seen = 0
    current: list[int] = []
    for shard_path in shard_paths:
        tokens = load_data_shard(shard_path)
        for token in tokens.tolist():
            if token == bos_id:
                if current:
                    yield sp.decode(current)
                    seen += 1
                    if doc_limit is not None and seen >= doc_limit:
                        return
                    current = []
            else:
                current.append(token)
    if current and (doc_limit is None or seen < doc_limit):
        yield sp.decode(current)


def count_docs(shard_paths: list[Path], bos_id: int) -> int:
    total = 0
    for shard_path in shard_paths:
        total += int((load_data_shard(shard_path) == bos_id).sum())
    return total


def train_sentencepiece(
    docs_iter_factory,
    model_prefix: Path,
    *,
    vocab_size: int,
    model_type: str,
) -> Path:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    for artifact in (model_prefix.with_suffix(".model"), model_prefix.with_suffix(".vocab")):
        if artifact.exists():
            artifact.unlink()
    spm.SentencePieceTrainer.train(
        sentence_iterator=docs_iter_factory(),
        model_prefix=str(model_prefix),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=0.999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )
    return model_prefix.with_suffix(".model")


def write_datafile(path: Path, toks: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(toks.size)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.astype("<u2", copy=False).tobytes())


def write_split(
    *,
    docs_iter: Iterator[str],
    out_dir: Path,
    split: str,
    tok: spm.SentencePieceProcessor,
    shard_size: int,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"fineweb_{split}_*.bin"):
        stale.unlink()

    buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    shard_idx = 0
    stats = {"docs": 0, "tokens": 0, "files": 0}

    def flush() -> None:
        nonlocal fill, shard_idx
        if fill == 0:
            return
        write_datafile(out_dir / f"fineweb_{split}_{shard_idx:06d}.bin", buf[:fill])
        shard_idx += 1
        stats["files"] += 1
        fill = 0

    for doc in docs_iter:
        pieces = tok.encode(doc, out_type=int)
        toks = np.empty((len(pieces) + 1,), dtype=np.int32)
        toks[0] = int(tok.bos_id())
        toks[1:] = np.asarray(pieces, dtype=np.int32)
        if not ((0 <= toks).all() and (toks < int(tok.vocab_size())).all()):
            raise ValueError(f"token out of range for split={split}")

        stats["docs"] += 1
        stats["tokens"] += int(toks.size)

        pos = 0
        while pos < toks.size:
            take = min(shard_size - fill, toks.size - pos)
            buf[fill : fill + take] = toks[pos : pos + take]
            fill += take
            pos += take
            if fill == shard_size:
                flush()
        if stats["docs"] % 10_000 == 0:
            print(f"{split}: {stats['docs']} docs", flush=True)

    flush()
    return stats


def main() -> None:
    args = parse_args()
    input_data_path = Path(args.input_data_path).resolve()
    output_root = Path(args.output_root).resolve()
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets" / f"fineweb10B_{args.dataset_suffix}"
    model_prefix = tokenizers_dir / f"fineweb_{args.vocab_size}_{args.model_type}"

    input_sp = spm.SentencePieceProcessor(model_file=str(Path(args.input_tokenizer_path).resolve()))
    train_shards = sorted(input_data_path.glob("fineweb_train_*.bin"))
    val_shards = sorted(input_data_path.glob("fineweb_val_*.bin"))
    if not train_shards or not val_shards:
        raise FileNotFoundError(f"missing train/val shards under {input_data_path}")

    print(
        f"source train_shards:{len(train_shards)} val_shards:{len(val_shards)} "
        f"train_docs:{count_docs(train_shards, int(input_sp.bos_id()))} "
        f"val_docs:{count_docs(val_shards, int(input_sp.bos_id()))}",
        flush=True,
    )

    def train_docs_iter() -> Iterator[str]:
        return iter_docs_from_shards(
            train_shards,
            input_sp,
            doc_limit=args.tokenizer_train_docs,
        )

    model_path = train_sentencepiece(
        train_docs_iter,
        model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
    tok = spm.SentencePieceProcessor(model_file=str(model_path))

    train_stats = write_split(
        docs_iter=iter_docs_from_shards(train_shards, input_sp, doc_limit=args.train_doc_limit),
        out_dir=datasets_dir,
        split="train",
        tok=tok,
        shard_size=args.shard_size,
    )
    val_stats = write_split(
        docs_iter=iter_docs_from_shards(val_shards, input_sp, doc_limit=args.val_doc_limit),
        out_dir=datasets_dir,
        split="val",
        tok=tok,
        shard_size=args.shard_size,
    )

    manifest = {
        "source_kind": "local_decoded_sp_shards",
        "source_data_path": str(input_data_path),
        "source_tokenizer_path": str(Path(args.input_tokenizer_path).resolve()),
        "dataset_suffix": args.dataset_suffix,
        "dataset_name": f"fineweb10B_{args.dataset_suffix}",
        "model_type": args.model_type,
        "vocab_size": int(tok.vocab_size()),
        "tokenizer_train_docs": args.tokenizer_train_docs,
        "train_doc_limit": args.train_doc_limit,
        "val_doc_limit": args.val_doc_limit,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "tokenizer_model_path": str(model_path),
        "tokenizer_vocab_path": str(model_path.with_suffix('.vocab')),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "local_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
