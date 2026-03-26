# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120_continue80_keptfp16_continue80_keptfp16_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.11240000`
- Shipped `val_bpb`: `2.27250127`
- Export gap: `0.16010127`
- Total bytes: `15569626`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.38489239`
- Reference export gap: `0.27249239`
- Shipped delta vs best: `0.11239112`
- Gap delta vs best: `0.11239112`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.27250127` | `0.16010127` | `15569626` | `True` |
| `tok_attnhi_fp16` | `2.28112440` | `0.16872440` | `10855848` | `True` |
| `int8_tok_fp16` | `2.28234720` | `0.16994720` | `8278155` | `True` |
| `int8_all` | `2.28589771` | `0.17349771` | `7733510` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.38489239` | `0.27249239` | `8501059` | `True` |
| `fcproj_attnhi_fp16` | `2.43662130` | `0.32422130` | `8166076` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.43662130` | `0.32422130` | `8166076` | `True` |
| `fchi_attnhi_fp16` | `2.43663655` | `0.32423655` | `8696279` | `True` |
| `fchi_only` | `2.43892995` | `0.32652995` | `6101780` | `True` |
| `fcproj_hi` | `2.43893193` | `0.32653193` | `5579371` | `True` |
