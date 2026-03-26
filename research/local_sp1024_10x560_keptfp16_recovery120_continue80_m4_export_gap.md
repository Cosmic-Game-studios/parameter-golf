# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120_continue80_keptfp16_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.11450000`
- Shipped `val_bpb`: `2.27347403`
- Export gap: `0.15897403`
- Total bytes: `15570260`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.38518097`
- Reference export gap: `0.27068097`
- Shipped delta vs best: `0.11170694`
- Gap delta vs best: `0.11170694`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.27347403` | `0.15897403` | `15570260` | `True` |
| `tok_attnhi_fp16` | `2.28231939` | `0.16781939` | `10848258` | `True` |
| `int8_tok_fp16` | `2.28308489` | `0.16858489` | `8266249` | `True` |
| `int8_all` | `2.28667267` | `0.17217267` | `7720605` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.38518097` | `0.27068097` | `8492030` | `True` |
| `fcproj_attnhi_fp16` | `2.43641884` | `0.32191884` | `8165989` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.43641884` | `0.32191884` | `8165989` | `True` |
| `fchi_attnhi_fp16` | `2.43642307` | `0.32192307` | `8686461` | `True` |
| `fchi_only` | `2.43773948` | `0.32323948` | `6088686` | `True` |
| `fcproj_hi` | `2.43775670` | `0.32325670` | `5574601` | `True` |
