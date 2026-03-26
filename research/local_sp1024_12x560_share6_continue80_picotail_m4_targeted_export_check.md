# Targeted Export Frontier Check

- Preset: `local_sp1024_12x560_share6_picotail_targeted`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_continue80_picotail_m4_mlx_model.npz`
- Scope: actual shared-depth winner candidates only
- Note: excludes stalled non-winning policies that were already dominated on earlier checkpoints.

## Best Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.19010000`
- Shipped `val_bpb`: `2.19618704`
- Export gap: `0.00608704`
- Total bytes: `14309510`

## Ranked Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.19618704` | `0.00608704` | `14309510` | `True` |
| `proj_top6_attn_top5_fp16` | `2.19882042` | `0.00872042` | `13802801` | `True` |
| `proj_top5_attn_top6_fp16` | `2.21525219` | `0.02515219` | `13244900` | `True` |
| `projhi_attnhi_fp16` | `2.25754488` | `0.06744488` | `9595136` | `True` |
| `int8_all` | `2.30014747` | `0.11004747` | `4909478` | `True` |
