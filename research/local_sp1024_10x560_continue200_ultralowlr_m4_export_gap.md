# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.22030000`
- Shipped `val_bpb`: `2.31488601`
- Export gap: `0.09458601`
- Total bytes: `15504049`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.39279421`
- Reference export gap: `0.17249421`
- Shipped delta vs best: `0.07790820`
- Gap delta vs best: `0.07790820`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.31488601` | `0.09458601` | `15504049` | `True` |
| `int8_tok_fp16` | `2.32480409` | `0.10450409` | `8086659` | `True` |
| `tok_attnhi_fp16` | `2.32586622` | `0.10556622` | `10711401` | `True` |
| `int8_all` | `2.32755465` | `0.10725465` | `7534199` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.39279421` | `0.17249421` | `8349223` | `True` |
| `fchi_only` | `2.42303285` | `0.20273285` | `5908120` | `True` |
| `fcproj_hi` | `2.42303878` | `0.20273878` | `5468002` | `True` |
| `fchi_attnhi_fp16` | `2.42381698` | `0.20351698` | `8551561` | `True` |
| `fcproj_attnhi_fp16` | `2.42382122` | `0.20352122` | `8102984` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.42382122` | `0.20352122` | `8102984` | `True` |
