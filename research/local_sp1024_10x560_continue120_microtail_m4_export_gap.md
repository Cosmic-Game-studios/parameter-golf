# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.16110000`
- Shipped `val_bpb`: `2.28212046`
- Export gap: `0.12102046`
- Total bytes: `15538191`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.37250881`
- Reference export gap: `0.21140881`
- Shipped delta vs best: `0.09038835`
- Gap delta vs best: `0.09038835`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.28212046` | `0.12102046` | `15538191` | `True` |
| `int8_tok_fp16` | `2.29425499` | `0.13315499` | `8158337` | `True` |
| `tok_attnhi_fp16` | `2.29517763` | `0.13407763` | `10765830` | `True` |
| `int8_all` | `2.29614941` | `0.13504941` | `7610007` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.37250881` | `0.21140881` | `8409587` | `True` |
| `fchi_only` | `2.41049198` | `0.24939198` | `5982008` | `True` |
| `fcproj_hi` | `2.41049707` | `0.24939707` | `5515976` | `True` |
| `fcproj_attnhi_fp16` | `2.41127923` | `0.25017923` | `8133319` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.41127923` | `0.25017923` | `8133319` | `True` |
| `fchi_attnhi_fp16` | `2.41129165` | `0.25019165` | `8607616` | `True` |
