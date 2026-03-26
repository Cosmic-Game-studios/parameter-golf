# Fixed Export Gap Search

- Preset: `local_sp1024_12x544_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x544_kv2_share6_896ctx_modern_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.56750000`
- Shipped `val_bpb`: `2.56969060`
- Export gap: `0.00219060`
- Total bytes: `13467285`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.56969060`
- Reference export gap: `0.00219060`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.56969060` | `0.00219060` | `13467285` | `True` |
| `proj_top6_attn_top5_fp16` | `2.57273821` | `0.00523821` | `12942872` | `True` |
| `proj_top5_attn_top6_fp16` | `2.57540886` | `0.00790886` | `12423946` | `True` |
| `projhi_attnhi_fp16` | `2.61409626` | `0.04659626` | `8757068` | `True` |
| `tok_attnhi_fp16` | `2.65317644` | `0.08567644` | `6179468` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.65813625` | `0.09063625` | `6134503` | `True` |
| `int8_tok_fp16` | `2.67405623` | `0.10655623` | `4591350` | `True` |
| `int8_all` | `2.68034544` | `0.11284544` | `4023003` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.70693445` | `0.13943445` | `5830461` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.70693445` | `0.13943445` | `5830461` | `True` |
| `fcproj_attnhi_fp16` | `2.99472343` | `0.42722343` | `4566969` | `True` |
| `fchi_attnhi_fp16` | `2.99472343` | `0.42722343` | `4705787` | `True` |
| `fcproj_hi` | `3.06334902` | `0.49584902` | `2972392` | `True` |
| `fchi_only` | `3.06334902` | `0.49584902` | `3108552` | `True` |
| `fchi_top4_attn_top4_fp16` | `3.38499294` | `0.81749294` | `4932242` | `True` |
| `fcproj_top5_attn_top5_fp16` | `3.91753585` | `1.35003585` | `4911391` | `True` |
