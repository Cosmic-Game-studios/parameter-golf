# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue80_lowlr_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.38210000`
- Shipped `val_bpb`: `2.43972002`
- Export gap: `0.05762002`
- Total bytes: `15387766`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.49826193`
- Reference export gap: `0.11616193`
- Shipped delta vs best: `0.05854191`
- Gap delta vs best: `0.05854191`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.43972002` | `0.05762002` | `15387766` | `True` |
| `int8_tok_fp16` | `2.45013097` | `0.06803097` | `7852003` | `True` |
| `tok_attnhi_fp16` | `2.45149566` | `0.06939566` | `10527110` | `True` |
| `int8_all` | `2.45225495` | `0.07015495` | `7293525` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.49826193` | `0.11616193` | `8152207` | `True` |
| `fchi_attnhi_fp16` | `2.51939585` | `0.13729585` | `8368750` | `True` |
| `fcproj_attnhi_fp16` | `2.51939811` | `0.13729811` | `7990943` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.51939811` | `0.13729811` | `7990943` | `True` |
| `fchi_only` | `2.51971916` | `0.13761916` | `5675607` | `True` |
| `fcproj_hi` | `2.51972001` | `0.13762001` | `5306191` | `True` |
