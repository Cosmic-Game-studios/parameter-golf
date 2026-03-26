# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_fp16_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.57240000`
- Shipped `val_bpb`: `2.65395662`
- Export gap: `0.08155662`
- Total bytes: `15067027`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.75095835`
- Reference export gap: `0.17855835`
- Shipped delta vs best: `0.09700173`
- Gap delta vs best: `0.09700173`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.65395662` | `0.08155662` | `15067027` | `True` |
| `tok_attnhi_fp16` | `2.66680555` | `0.09440555` | `10097011` | `True` |
| `int8_tok_fp16` | `2.68009074` | `0.10769074` | `7288141` | `True` |
| `int8_all` | `2.68681847` | `0.11441847` | `6701967` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.75095835` | `0.17855835` | `7676812` | `True` |
| `fchi_attnhi_fp16` | `2.77334147` | `0.20094147` | `7920597` | `True` |
| `fcproj_attnhi_fp16` | `2.77334712` | `0.20094712` | `7668899` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.77334712` | `0.20094712` | `7668899` | `True` |
| `fchi_only` | `2.79650675` | `0.22410675` | `5099925` | `True` |
| `fcproj_hi` | `2.79651099` | `0.22411099` | `4853002` | `True` |
