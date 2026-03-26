# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fc_top4_int4`
- Clean `val_bpb`: `1.98710000`
- Shipped `val_bpb`: `1.98453131`
- Export gap: `-0.00256869`
- Total bytes: `14937204`
- Int4 groups: `fc_top4`
- FP16 groups: `-`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fc_top4_int4` | `1.98453131` | `-0.00256869` | `14937204` | `True` |
| `fc_top4_attn_top1_qer_proj_top1_r64` | `1.98454109` | `-0.00255891` | `15637181` | `True` |
| `fc_top4_attn_top1_fp16` | `1.98454171` | `-0.00255829` | `15420572` | `True` |
