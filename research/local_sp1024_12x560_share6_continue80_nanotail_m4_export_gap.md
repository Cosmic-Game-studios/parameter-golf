# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_continue80_nanotail_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.21460000`
- Shipped `val_bpb`: `2.21983179`
- Export gap: `0.00523179`
- Total bytes: `14307260`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.21983179`
- Reference export gap: `0.00523179`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.21983179` | `0.00523179` | `14307260` | `True` |
| `proj_top6_attn_top5_fp16` | `2.22186584` | `0.00726584` | `13797902` | `True` |
| `proj_top5_attn_top6_fp16` | `2.23591808` | `0.02131808` | `13242041` | `True` |
| `projhi_attnhi_fp16` | `2.27118670` | `0.05658670` | `9586457` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.29550320` | `0.08090320` | `6961104` | `True` |
| `tok_attnhi_fp16` | `2.29804382` | `0.08344382` | `6968790` | `True` |
| `int8_tok_fp16` | `2.30169542` | `0.08709542` | `5435562` | `True` |
| `int8_all` | `2.30413466` | `0.08953466` | `4884194` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.34481093` | `0.13021093` | `6637488` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.34481093` | `0.13021093` | `6637488` | `True` |
| `fcproj_hi` | `2.60135887` | `0.38675887` | `3602272` | `True` |
| `fchi_only` | `2.60139388` | `0.38679388` | `3907284` | `True` |
| `fcproj_attnhi_fp16` | `2.60331173` | `0.38871173` | `5141661` | `True` |
| `fchi_attnhi_fp16` | `2.60333319` | `0.38873319` | `5448321` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.99965529` | `0.78505529` | `5634382` | `True` |
| `fcproj_top5_attn_top5_fp16` | `4.15950394` | `1.94490394` | `5329214` | `True` |
