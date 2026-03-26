# Staged SOTA Trial Search

Objective:

`J = shipped_val_bpb + lambda1*byte_penalty + lambda2*runtime_penalty + lambda3*clean_to_shipped_gap`

- Target shipped bpb: `1.1000`
- Counted code bytes: `66325`
- Public baseline shipped bpb: `1.2244`

## 1.1 Verdict

- Best evidence-backed candidate today is `trial_12x608_continue_lfqat_ramp_fcproj_attnhi_fp16` at estimated shipped `1.4301` bpb.
- Even under the searcher's optimistic export floor, the closest candidate is `trial_12x608_continue_lfqat_ramp_fcproj_attnhi_fp16` at `1.3729` bpb.
- Clean improvement still needed after best-case export: `0.2729` bpb.
- Current verdict: no legal candidate in the present search space is yet target-reachable or even target-stretch for `1.1`; the current family still needs a materially stronger raw checkpoint, not just marginal export tuning.

## Stage 1 Screeners

| Rank | Candidate | Shape | Recipe | Export | Est. shipped bpb | Best-case shipped | Clean gain still needed | Legal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1` | `trial_12x608_continue_lfqat_ramp_fcproj_attnhi_fp16` | `12x608` | `continue_lfqat_ramp` | `fcproj_attnhi_fp16` | `1.4301` | `1.3729` | `0.2729` | `yes` |
| `2` | `trial_12x608_continue_int8_recovery_int8_tok_fp16` | `12x608` | `continue_int8_recovery` | `int8_tok_fp16` | `1.4501` | `1.3879` | `0.2879` | `yes` |
| `3` | `trial_12x608_continue_lfqat_base_fcproj_attnhi_fp16` | `12x608` | `continue_lfqat_base` | `fcproj_attnhi_fp16` | `1.4701` | `1.3929` | `0.2929` | `yes` |
| `4` | `trial_12x608_continue_int8_recovery_int8_all` | `12x608` | `continue_int8_recovery` | `int8_all` | `1.4701` | `1.3929` | `0.2929` | `yes` |
| `5` | `trial_12x608_continue_lfqat_ramp_int8_tok_fp16` | `12x608` | `continue_lfqat_ramp` | `int8_tok_fp16` | `1.4801` | `1.3779` | `0.2779` | `yes` |
| `6` | `trial_12x608_continue_int8_recovery_fcproj_attnhi_fp16` | `12x608` | `continue_int8_recovery` | `fcproj_attnhi_fp16` | `1.4901` | `1.3829` | `0.2829` | `yes` |
| `7` | `trial_12x608_continue_lfqat_base_int8_tok_fp16` | `12x608` | `continue_lfqat_base` | `int8_tok_fp16` | `1.4901` | `1.3979` | `0.2979` | `yes` |
| `8` | `trial_12x608_continue_lfqat_ramp_int8_all` | `12x608` | `continue_lfqat_ramp` | `int8_all` | `1.5001` | `1.3829` | `0.2829` | `yes` |

## Best Moonshots

| Candidate | Shape | Recipe | Export | Est. shipped bpb | Best-case shipped | Clean gain still needed | Legal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `trial_12x576_warmstart_lfqat_fc_hi_int4` | `12x576` | `warmstart_lfqat` | `fc_hi_int4` | `2.0655` | `2.0255` | `0.9255` | `yes` |
| `trial_14x576_scratch_cosine_stable_fc_hi_int4` | `14x576` | `scratch_cosine_stable` | `fc_hi_int4` | `2.0662` | `1.9962` | `0.8962` | `yes` |
| `trial_14x576_scratch_cosine_fast_fc_hi_int4` | `14x576` | `scratch_cosine_fast` | `fc_hi_int4` | `2.0962` | `2.0162` | `0.9162` | `yes` |

## Commands

### `trial_12x608_continue_lfqat_ramp_fcproj_attnhi_fp16`

- Estimated clean bpb: `1.3129`
- Estimated shipped bpb: `1.4301`
- Estimated gap: `0.1172`
- Best-case shipped with optimistic export: `1.3729`
- Clean gain still needed after best export: `0.2729`
- Estimated bytes: `15642824`

1xH100 screen:
```bash
RUN_ID=trial_12x608_continue_lfqat_ramp_fcproj_attnhi_fp16 \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  LR_SCHEDULE=cosine \
  LR_WARMUP_ITERS=20 \
  MIN_LR_SCALE=0.15 \
  TIED_EMBED_LR=0.0022 \
  MATRIX_LR=0.0016 \
  SCALAR_LR=0.0016 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  ITERATIONS=2500 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=25 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=40 \
  LFQAT_MIN_PROB=0.2 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.05 \
  LFQAT_FISHER_WEIGHT=0.01 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

8xH100 final:
```bash
RUN_ID=final_12x608_continue_lfqat_ramp_fcproj_attnhi_fp16 \
  NPROC_PER_NODE=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  LR_SCHEDULE=cosine \
  LR_WARMUP_ITERS=20 \
  MIN_LR_SCALE=0.15 \
  TIED_EMBED_LR=0.0022 \
  MATRIX_LR=0.0016 \
  SCALAR_LR=0.0016 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  ITERATIONS=2500 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=25 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=40 \
  LFQAT_MIN_PROB=0.2 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.05 \
  LFQAT_FISHER_WEIGHT=0.01 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

Rationale:
- Closest measured shape anchor is `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` at `1.6101` shipped bpb.
- This shape also has official-path evidence with clean `1.3329` bpb.
- Official dense anchor keeps this shape's clean bpb prior at the measured best official value.
- Recipe `continue_lfqat_ramp` contributes `-0.0200` clean-bpb delta and `-0.0200` export-gap delta.
- Policy `fcproj_attnhi_fp16` contributes `0.1400` expected shipped-bpb recovery for roughly `4.8MB` extra bytes.
- Estimated runtime ratio vs the current `12x608` official anchor is `1.030`.
- Estimated clean `1.3129` and shipped `1.4301` bpb still leave `0.3301` to the `1.1` goal.
- With optimistic export floor `0.0600`, best-case shipped becomes `1.3729` and still needs `0.2729` more clean improvement.

### `trial_12x608_continue_int8_recovery_int8_tok_fp16`

- Estimated clean bpb: `1.3229`
- Estimated shipped bpb: `1.4501`
- Estimated gap: `0.1272`
- Best-case shipped with optimistic export: `1.3879`
- Clean gain still needed after best export: `0.2879`
- Estimated bytes: `15442824`

1xH100 screen:
```bash
RUN_ID=trial_12x608_continue_int8_recovery_int8_tok_fp16 \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  LR_SCHEDULE=cosine \
  LR_WARMUP_ITERS=20 \
  MIN_LR_SCALE=0.2 \
  TIED_EMBED_LR=0.0018 \
  MATRIX_LR=0.0012 \
  SCALAR_LR=0.0012 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  ITERATIONS=2500 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=25 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=20 \
  LFQAT_MIN_PROB=0.1 \
  LFQAT_MAX_PROB=0.7 \
  LFQAT_KL_WEIGHT=0.008 \
  LFQAT_FISHER_WEIGHT=0.002 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0001 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=int8_clean_per_row_v1 \
  TARGET_EXPORT_NAME_PATTERNS= \
  INT4_NAME_PATTERNS= \
  TRAIN_QAT_NAME_PATTERNS=__never__ \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

8xH100 final:
```bash
RUN_ID=final_12x608_continue_int8_recovery_int8_tok_fp16 \
  NPROC_PER_NODE=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  LR_SCHEDULE=cosine \
  LR_WARMUP_ITERS=20 \
  MIN_LR_SCALE=0.2 \
  TIED_EMBED_LR=0.0018 \
  MATRIX_LR=0.0012 \
  SCALAR_LR=0.0012 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  ITERATIONS=2500 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=25 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=20 \
  LFQAT_MIN_PROB=0.1 \
  LFQAT_MAX_PROB=0.7 \
  LFQAT_KL_WEIGHT=0.008 \
  LFQAT_FISHER_WEIGHT=0.002 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0001 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=int8_clean_per_row_v1 \
  TARGET_EXPORT_NAME_PATTERNS= \
  INT4_NAME_PATTERNS= \
  TRAIN_QAT_NAME_PATTERNS=__never__ \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

Rationale:
- Closest measured shape anchor is `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` at `1.6101` shipped bpb.
- This shape also has official-path evidence with clean `1.3329` bpb.
- Official dense anchor keeps this shape's clean bpb prior at the measured best official value.
- Recipe `continue_int8_recovery` contributes `-0.0100` clean-bpb delta and `-0.0300` export-gap delta.
- Policy `int8_tok_fp16` contributes `0.1200` expected shipped-bpb recovery for roughly `4.6MB` extra bytes.
- Estimated runtime ratio vs the current `12x608` official anchor is `1.000`.
- Estimated clean `1.3229` and shipped `1.4501` bpb still leave `0.3501` to the `1.1` goal.
- With optimistic export floor `0.0650`, best-case shipped becomes `1.3879` and still needs `0.2879` more clean improvement.

### `trial_12x608_continue_lfqat_base_fcproj_attnhi_fp16`

- Estimated clean bpb: `1.3329`
- Estimated shipped bpb: `1.4701`
- Estimated gap: `0.1372`
- Best-case shipped with optimistic export: `1.3929`
- Clean gain still needed after best export: `0.2929`
- Estimated bytes: `15642824`

1xH100 screen:
```bash
RUN_ID=trial_12x608_continue_lfqat_base_fcproj_attnhi_fp16 \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  LR_SCHEDULE=cosine \
  LR_WARMUP_ITERS=20 \
  MIN_LR_SCALE=0.2 \
  TIED_EMBED_LR=0.002 \
  MATRIX_LR=0.0015 \
  SCALAR_LR=0.0015 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  ITERATIONS=2500 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=25 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.05 \
  LFQAT_FISHER_WEIGHT=0.01 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

8xH100 final:
```bash
RUN_ID=final_12x608_continue_lfqat_base_fcproj_attnhi_fp16 \
  NPROC_PER_NODE=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  LR_SCHEDULE=cosine \
  LR_WARMUP_ITERS=20 \
  MIN_LR_SCALE=0.2 \
  TIED_EMBED_LR=0.002 \
  MATRIX_LR=0.0015 \
  SCALAR_LR=0.0015 \
  TRAIN_BATCH_TOKENS=524288 \
  VAL_BATCH_SIZE=524288 \
  ITERATIONS=2500 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=25 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.05 \
  LFQAT_FISHER_WEIGHT=0.01 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

Rationale:
- Closest measured shape anchor is `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` at `1.6101` shipped bpb.
- This shape also has official-path evidence with clean `1.3329` bpb.
- Official dense anchor keeps this shape's clean bpb prior at the measured best official value.
- Recipe `continue_lfqat_base` contributes `+0.0000` clean-bpb delta and `+0.0000` export-gap delta.
- Policy `fcproj_attnhi_fp16` contributes `0.1400` expected shipped-bpb recovery for roughly `4.8MB` extra bytes.
- Estimated runtime ratio vs the current `12x608` official anchor is `1.000`.
- Estimated clean `1.3329` and shipped `1.4701` bpb still leave `0.3701` to the `1.1` goal.
- With optimistic export floor `0.0600`, best-case shipped becomes `1.3929` and still needs `0.2929` more clean improvement.
