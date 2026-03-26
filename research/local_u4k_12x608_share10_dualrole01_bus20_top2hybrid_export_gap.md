# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fc_top4_attn_top2_fp16`
- Clean `val_bpb`: `1.98710000`
- Shipped `val_bpb`: `1.98453859`
- Export gap: `-0.00256141`
- Total bytes: `15911253`
- Int4 groups: `fc_top4`
- FP16 groups: `attn_top2`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fc_top4_attn_top2_fp16` | `1.98453859` | `-0.00256141` | `15911253` | `True` |
| `fc_top4_proj_top2_fp16` | `1.98454338` | `-0.00255662` | `16972692` | `False` |
| `fc_top4_proj_top2_attn_top2_fp16` | `1.98454088` | `-0.00255912` | `17947288` | `False` |
