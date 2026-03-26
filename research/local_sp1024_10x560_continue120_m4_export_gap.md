# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue120_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.47920000`
- Shipped `val_bpb`: `2.52046659`
- Export gap: `0.04126659`
- Total bytes: `15296349`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.57173693`
- Reference export gap: `0.09253693`
- Shipped delta vs best: `0.05127034`
- Gap delta vs best: `0.05127034`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.52046659` | `0.04126659` | `15296349` | `True` |
| `tok_attnhi_fp16` | `2.53005444` | `0.05085444` | `10402362` | `True` |
| `int8_tok_fp16` | `2.53124575` | `0.05204575` | `7684428` | `True` |
| `int8_all` | `2.53319241` | `0.05399241` | `7119552` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.57173693` | `0.09253693` | `8015485` | `True` |
| `fcproj_attnhi_fp16` | `2.58930961` | `0.11010961` | `7901523` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.58930961` | `0.11010961` | `7901523` | `True` |
| `fchi_attnhi_fp16` | `2.58931046` | `0.11011046` | `8239754` | `True` |
| `fcproj_hi` | `2.59398197` | `0.11478197` | `5175001` | `True` |
| `fchi_only` | `2.59398564` | `0.11478564` | `5506617` | `True` |
