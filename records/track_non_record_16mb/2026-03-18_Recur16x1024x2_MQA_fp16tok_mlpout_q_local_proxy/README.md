This folder captures the strongest legal local prototype produced in the current leaderboard hunt.

Summary:
- Architecture: `NUM_LAYERS=16`, `NUM_UNIQUE_LAYERS=2`, `MODEL_DIM=1024`, `NUM_HEADS=8`, `NUM_KV_HEADS=1`, `MLP_MULT=2`
- Core idea: use two shared transformer blocks recurrently across sixteen unrolled steps, with top-level per-step residual/attention/MLP control tensors.
- Winning legal export: keep `tok_emb.weight`, `mlp.proj.weight`, and `attn.c_q.weight` in fp16 during int8 export via `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight,mlp.proj.weight,attn.c_q.weight`

Why this direction:
- The public dense baseline spends almost all counted bytes on unique dense block weights.
- Shared recurrent blocks buy much more effective depth per counted byte.
- The official metric is post-roundtrip `val_bpb`, so export quality is a first-class optimization target.
- The 100-step local checkpoint still had a large quantization gap, and selective fp16 passthrough on the MLP output and Q projection recovered much more of the raw-model gain while staying under the `16,000,000` byte cap.

Measured local result:
- Run path: `train_gpt_mlx.py` on Apple Silicon, using the published `fineweb10B_sp1024` data/tokenizer path.
- Train config: `ITERATIONS=100`, `TRAIN_BATCH_TOKENS=8192`, `WARMUP_STEPS=5`.
- Validation scope: exploratory local prefix of the official validation split, `VAL_MAX_TOKENS=262144`.
- Base 100-step direct-export metric: `final_int8_zlib_roundtrip_exact val_loss:5.43058348 val_bpb:3.21585663`.
- Final packaged metric after re-exporting the same checkpoint with the best legal fp16 tensor set: `final_int8_zlib_roundtrip_exact val_loss:4.77079535 val_bpb:2.82514649`.
- Compressed model bytes for the packaged export: `15,538,873`.
- Counted code bytes for `train_gpt.py`: `49,685`.
- Counted total: `15,588,558`.

What changed versus the public dense baseline:
- Replaced unique-per-layer blocks with a recurrent shared-block stack (`2` unique blocks, `16` total steps).
- Moved residual/attention/MLP scaling to top-level per-step tensors so shared blocks can specialize by unrolled step.
- Switched to `NUM_KV_HEADS=1` to improve byte efficiency.
- Added opt-in fp16 passthrough export for selected large tensors.
- Spent the legal byte headroom on `mlp.proj.weight` and `attn.c_q.weight`, which proved to be much higher leverage than extra int8-only matrices.

Runtime notes:
- Main local training run: `390.4s` to final validation plus `21.3s` for the direct int8 roundtrip eval.
- Final legal export verification: `0.7s` quantization plus `21.4s` eval.
- The best over-cap frontier seen locally was `tok_emb.weight + mlp.proj.weight + attn.proj.weight`, which reached `val_bpb:2.76068568` at `16,589,204` total counted bytes and therefore did not qualify.

Reproduction:
- The measured `train.log` contains the 100-step local training run plus a final offline export-verification block for the packaged fp16 tensor set.
- To reproduce the packaged path directly from training, rerun `train_gpt_mlx.py` or `train_gpt.py` with the architecture above and `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight,mlp.proj.weight,attn.c_q.weight`.

Included files:
- `train_gpt.py`: CUDA/PyTorch path with recurrent sharing and selective fp16-export support.
- `train_gpt_mlx.py`: local MLX helper used for the measured proxy runs in this folder.
- `train.log`: the 100-step local training log plus the final legal export verification.
- `submission.json`: metadata for this local submission candidate.

Current status:
- This is still a local proxy result, not a record-eligible leaderboard submission.
- The metric in this folder is from a `262,144`-token prefix of the official validation split, not the full official validation run.
- The path is nonetheless plausibly record-competitive: the legal export recovered a very large share of the raw-model gain while staying under the official byte cap.

Highest-leverage next experiments:
- Run the packaged recipe on the full official validation split and real CUDA hardware to see whether the local export win survives end-to-end.
- Try to pull the over-cap `mlp.proj + attn.proj` export back under `16,000,000` bytes, for example by shrinking code bytes or making the fp16 payload compress more cleanly.
- Re-test nearby training budgets and widths around this exact export recipe, especially `ITERATIONS` in the `100-200` range and `NUM_LAYERS` in the `16-20` range.
