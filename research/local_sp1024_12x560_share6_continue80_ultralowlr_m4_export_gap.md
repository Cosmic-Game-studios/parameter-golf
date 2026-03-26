# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_continue80_ultralowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.25960000`
- Shipped `val_bpb`: `2.26380416`
- Export gap: `0.00420416`
- Total bytes: `14300632`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.26380416`
- Reference export gap: `0.00420416`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.26380416` | `0.00420416` | `14300632` | `True` |
| `proj_top6_attn_top5_fp16` | `2.26560271` | `0.00600271` | `13788603` | `True` |
| `proj_top5_attn_top6_fp16` | `2.27721556` | `0.01761556` | `13233341` | `True` |
| `projhi_attnhi_fp16` | `2.30771595` | `0.04811595` | `9565650` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.32577812` | `0.06617812` | `6930615` | `True` |
| `tok_attnhi_fp16` | `2.32744325` | `0.06784325` | `6942308` | `True` |
| `int8_tok_fp16` | `2.33011334` | `0.07051334` | `5397030` | `True` |
| `int8_all` | `2.33248030` | `0.07288030` | `4843614` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.36699132` | `0.10739132` | `6606745` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.36699132` | `0.10739132` | `6606745` | `True` |
| `fcproj_attnhi_fp16` | `2.59854082` | `0.33894082` | `5123429` | `True` |
| `fchi_attnhi_fp16` | `2.59856285` | `0.33896285` | `5418169` | `True` |
| `fchi_only` | `2.60035759` | `0.34075759` | `3867114` | `True` |
| `fcproj_hi` | `2.60036493` | `0.34076493` | `3576338` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.99556969` | `0.73596969` | `5607583` | `True` |
| `fcproj_top5_attn_top5_fp16` | `4.11520530` | `1.85560530` | `5322496` | `True` |
