# SOTA Search Plan

- Target shipped bpb: `1.1000`
- Counted code bytes: `66325`
- Public baseline shipped bpb: `1.2244`

## Top Candidates

| Rank | Candidate | Shape | Init | Export policy | Est. shipped bpb | Est. bytes | Legal | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1` | `planner_dense_u4k_12x608_fcproj_attnhi_fp16` | `12x608` | `continue` | `fcproj_attnhi_fp16` | `1.4429` | `15642824` | `yes` | `0.93` |
| `2` | `planner_dense_u4k_12x608_int8_tok_fp16` | `12x608` | `continue` | `int8_tok_fp16` | `1.4601` | `15442824` | `yes` | `0.94` |
| `3` | `planner_dense_u4k_12x608_int8_all` | `12x608` | `continue` | `int8_all` | `1.4801` | `13442824` | `yes` | `0.95` |
| `4` | `planner_dense_u4k_12x608_fc_hi_int4` | `12x608` | `continue` | `fc_hi_int4` | `1.5201` | `11442824` | `yes` | `0.95` |
| `5` | `planner_dense_u4k_12x608_fcproj_hi_int4` | `12x608` | `continue` | `fcproj_hi_int4` | `1.5801` | `10842824` | `yes` | `0.95` |
| `6` | `planner_dense_u4k_12x576_fc_hi_int4` | `12x576` | `warmstart_expand` | `fc_hi_int4` | `2.2127` | `15777663` | `yes` | `0.54` |
| `7` | `planner_dense_u4k_13x576_fc_hi_int4` | `13x576` | `warmstart_expand` | `fc_hi_int4` | `2.2267` | `14521399` | `yes` | `0.29` |
| `8` | `planner_dense_u4k_12x576_fcproj_hi_int4` | `12x576` | `warmstart_expand` | `fcproj_hi_int4` | `2.2727` | `15177663` | `yes` | `0.55` |

## High-Upside Scale-Ups

| Candidate | Shape | Init | Export policy | Est. shipped bpb | Est. bytes | Legal |
| --- | --- | --- | --- | --- | --- | --- |
| `planner_dense_u4k_14x576_fc_hi_int4` | `14x576` | `scratch` | `fc_hi_int4` | `2.2934` | `15780590` | `yes` |
| `planner_dense_u4k_12x640_fcproj_hi_int4` | `12x640` | `warmstart_expand` | `fcproj_hi_int4` | `2.3125` | `16275371` | `no` |
| `planner_dense_u4k_15x576_fcproj_hi_int4` | `15x576` | `scratch` | `fcproj_hi_int4` | `2.3784` | `16497461` | `no` |

## Commands

### `planner_dense_u4k_12x608_fcproj_attnhi_fp16`

- Estimated clean bpb: `1.3329`
- Estimated clean-to-shipped gap: `0.1100`
- Distance to `1.1000`: `0.3429`

```bash
RUN_ID=planner_dense_u4k_12x608_fcproj_attnhi_fp16 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015 \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.fc.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.fc.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.11.mlp.proj.weight \
  LFQAT_KL_WEIGHT=0.05 \
  LFQAT_FISHER_WEIGHT=0.01 \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

Rationale:
- Closest measured shape anchor is `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` at `1.6101` shipped bpb.
- This shape also has official-path evidence with clean `1.3329` bpb.
- Official dense anchor keeps this shape's clean bpb prior at the measured best official value.
- Policy prior `fcproj_attnhi_fp16` assumes `0.14` bpb of export-gap recovery for `4.8MB` extra pressure.
- Estimated clean-to-shipped gap is `0.1100` bpb after init-mode adjustment.
- Current target remains `1.100` shipped bpb, so this candidate still needs `0.3429` bpb of further gain.

### `planner_dense_u4k_12x608_int8_tok_fp16`

- Estimated clean bpb: `1.3329`
- Estimated clean-to-shipped gap: `0.1272`
- Distance to `1.1000`: `0.3601`

```bash
RUN_ID=planner_dense_u4k_12x608_int8_tok_fp16 \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=int8_clean_per_row_v1 \
  TARGET_EXPORT_NAME_PATTERNS= \
  INT4_NAME_PATTERNS= \
  TRAIN_QAT_NAME_PATTERNS=__never__ \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0 \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  LFQAT_KL_WEIGHT=0 \
  LFQAT_FISHER_WEIGHT=0 \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

Rationale:
- Closest measured shape anchor is `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` at `1.6101` shipped bpb.
- This shape also has official-path evidence with clean `1.3329` bpb.
- Official dense anchor keeps this shape's clean bpb prior at the measured best official value.
- Policy prior `int8_tok_fp16` assumes `0.12` bpb of export-gap recovery for `4.6MB` extra pressure.
- Estimated clean-to-shipped gap is `0.1272` bpb after init-mode adjustment.
- Current target remains `1.100` shipped bpb, so this candidate still needs `0.3601` bpb of further gain.

### `planner_dense_u4k_12x608_int8_all`

- Estimated clean bpb: `1.3329`
- Estimated clean-to-shipped gap: `0.1472`
- Distance to `1.1000`: `0.3801`

```bash
RUN_ID=planner_dense_u4k_12x608_int8_all \
  MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=608 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt \
  QUANT_FORMAT=int8_clean_per_row_v1 \
  TARGET_EXPORT_NAME_PATTERNS= \
  INT4_NAME_PATTERNS= \
  TRAIN_QAT_NAME_PATTERNS=__never__ \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0 \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  LFQAT_KL_WEIGHT=0 \
  LFQAT_FISHER_WEIGHT=0 \
  scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh
```

Rationale:
- Closest measured shape anchor is `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` at `1.6101` shipped bpb.
- This shape also has official-path evidence with clean `1.3329` bpb.
- Official dense anchor keeps this shape's clean bpb prior at the measured best official value.
- Policy prior `int8_all` assumes `0.10` bpb of export-gap recovery for `2.6MB` extra pressure.
- Estimated clean-to-shipped gap is `0.1472` bpb after init-mode adjustment.
- Current target remains `1.100` shipped bpb, so this candidate still needs `0.3801` bpb of further gain.
