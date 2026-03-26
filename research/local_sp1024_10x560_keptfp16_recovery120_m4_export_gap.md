# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.11750000`
- Shipped `val_bpb`: `2.27454986`
- Export gap: `0.15704986`
- Total bytes: `15570613`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.38474697`
- Reference export gap: `0.26724697`
- Shipped delta vs best: `0.11019711`
- Gap delta vs best: `0.11019711`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.27454986` | `0.15704986` | `15570613` | `True` |
| `tok_attnhi_fp16` | `2.28465571` | `0.16715571` | `10842203` | `True` |
| `int8_tok_fp16` | `2.28514618` | `0.16764618` | `8254005` | `True` |
| `int8_all` | `2.28863034` | `0.17113034` | `7710477` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.38474697` | `0.26724697` | `8486848` | `True` |
| `fcproj_attnhi_fp16` | `2.43602268` | `0.31852268` | `8165908` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.43602268` | `0.31852268` | `8165908` | `True` |
| `fchi_attnhi_fp16` | `2.43602748` | `0.31852748` | `8680072` | `True` |
| `fchi_only` | `2.43644595` | `0.31894595` | `6076421` | `True` |
| `fcproj_hi` | `2.43645329` | `0.31895329` | `5571448` | `True` |
