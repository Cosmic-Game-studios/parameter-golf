# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_secondpass89_gate40_fc_top4_m4_retry1_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fc_top4_attn_top1_qer_proj_top1_r64`
- Clean `val_bpb`: `1.99930000`
- Shipped `val_bpb`: `1.99238898`
- Export gap: `-0.00691102`
- Total bytes: `15646322`
- Int4 groups: `fc_top4`
- FP16 groups: `attn_top1`
- Low-rank groups: `proj_top1`
- Low-rank rank: `64`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fc_top4_attn_top1_qer_proj_top1_r64` | `1.99238898` | `-0.00691102` | `15646322` | `True` |
| `fc_top4_int4` | `1.99239355` | `-0.00690645` | `14946411` | `True` |
