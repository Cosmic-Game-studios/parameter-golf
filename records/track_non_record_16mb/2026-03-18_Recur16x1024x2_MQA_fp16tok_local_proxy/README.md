This folder captures the strongest measured local prototype from the initial SOTA hunt.

Summary:
- Architecture: `NUM_LAYERS=16`, `NUM_UNIQUE_LAYERS=2`, `MODEL_DIM=1024`, `NUM_HEADS=8`, `NUM_KV_HEADS=1`, `MLP_MULT=2`
- Core idea: use two shared transformer blocks recurrently across sixteen unrolled steps, with top-level per-step residual/attention/MLP control tensors.
- Compression-aware export: keep the tied embedding `tok_emb.weight` as fp16 passthrough during int8 export via `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight`

Why this direction:
- The baseline spends almost all counted bytes on unique dense block weights.
- Shared recurrent blocks let us buy much more effective depth per counted byte.
- The official metric is post-roundtrip `val_bpb`, so reducing the quantization hit matters as much as improving pre-quant loss.
- Storing only the tied embedding in fp16 materially improved post-quant quality in local testing while keeping the payload proxy under the 16 MB cap.

Measured local result:
- Run path: `train_gpt_mlx.py` on Apple Silicon, using the published `fineweb10B_sp1024` data/tokenizer path
- Train config: `ITERATIONS=30`, `TRAIN_BATCH_TOKENS=8192`, `WARMUP_STEPS=2`
- Validation scope: exploratory local prefix of the official validation split, `VAL_MAX_TOKENS=262144`
- Final in-log metric: `final_int8_zlib_roundtrip_exact val_loss:5.54644108 val_bpb:3.28446462`
- Compressed model bytes from the log: `4,885,507`
- Counted code bytes for `train_gpt.py`: `49,685`
- Estimated counted total if only `train_gpt.py` is charged: `4,935,192`

What changed versus the public dense baseline:
- Replaced unique-per-layer blocks with a recurrent shared-block stack (`2` unique blocks, `16` total steps)
- Moved residual/attention/MLP scaling to top-level per-step tensors so shared blocks can specialize by unrolled step
- Switched to `NUM_KV_HEADS=1` to improve byte efficiency
- Added opt-in fp16 passthrough export for selected large tensors; the winning local recipe uses it only for the tied embedding

Included files:
- `train_gpt.py`: CUDA/PyTorch path with the recurrent architecture and fp16-export support
- `train_gpt_mlx.py`: local MLX helper used to produce the measured log in this folder
- `train.log`: exact local MLX log for the measured prototype
- `submission.json`: metadata for this exploratory handoff

Current status:
- This is not record-eligible yet.
- The measured metric in this folder is from a local proxy validation prefix, not the full official 50k-document validation run.
- The PyTorch/CUDA path has been ported and syntax-checked locally, but not executed on CUDA in this environment.

Highest-leverage next experiments:
- Run the selected recipe on the full official validation split and on CUDA to confirm the local gain survives the real evaluation path.
- Sweep `NUM_LAYERS` in the `16-20` range with `NUM_UNIQUE_LAYERS=2` under the fp16-embedding export, since `16` and `20` were nearly tied locally.
- Use the remaining byte headroom to selectively keep one additional tensor family in higher precision during export if CUDA runs show the quantization gap is still the main bottleneck.
