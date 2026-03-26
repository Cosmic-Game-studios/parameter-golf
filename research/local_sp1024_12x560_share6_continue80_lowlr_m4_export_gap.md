# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_continue80_lowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.33840000`
- Shipped `val_bpb`: `2.34223064`
- Export gap: `0.00383064`
- Total bytes: `14290864`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.34223064`
- Reference export gap: `0.00383064`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.34223064` | `0.00383064` | `14290864` | `True` |
| `proj_top6_attn_top5_fp16` | `2.34302721` | `0.00462721` | `13774633` | `True` |
| `proj_top5_attn_top6_fp16` | `2.35150718` | `0.01310718` | `13220592` | `True` |
| `projhi_attnhi_fp16` | `2.37541919` | `0.03701919` | `9532525` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.39029299` | `0.05189299` | `6879580` | `True` |
| `tok_attnhi_fp16` | `2.39304270` | `0.05464270` | `6887384` | `True` |
| `int8_tok_fp16` | `2.39385620` | `0.05545620` | `5333310` | `True` |
| `int8_all` | `2.39604908` | `0.05764908` | `4775793` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.42337056` | `0.08497056` | `6558553` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.42337056` | `0.08497056` | `6558553` | `True` |
| `fcproj_attnhi_fp16` | `2.61098229` | `0.27258229` | `5091622` | `True` |
| `fchi_attnhi_fp16` | `2.61098229` | `0.27258229` | `5367813` | `True` |
| `fcproj_hi` | `2.61492360` | `0.27652360` | `3530736` | `True` |
| `fchi_only` | `2.61492360` | `0.27652360` | `3801596` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.99118619` | `0.65278619` | `5561557` | `True` |
| `fcproj_top5_attn_top5_fp16` | `4.02128112` | `1.68288112` | `5305398` | `True` |
