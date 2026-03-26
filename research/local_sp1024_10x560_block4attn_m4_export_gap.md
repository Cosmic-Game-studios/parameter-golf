# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_hybrid_int8aware_block4attn_keptfp16_80_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.10930000`
- Shipped `val_bpb`: `2.27195757`
- Export gap: `0.16265757`
- Total bytes: `15569615`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.38498699`
- Reference export gap: `0.27568699`
- Shipped delta vs best: `0.11302942`
- Gap delta vs best: `0.11302942`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.27195757` | `0.16265757` | `15569615` | `True` |
| `tok_attnhi_fp16` | `2.27888648` | `0.16958648` | `10871172` | `True` |
| `int8_tok_fp16` | `2.28076127` | `0.17146127` | `8299531` | `True` |
| `int8_all` | `2.28432491` | `0.17502491` | `7754385` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.38498699` | `0.27568699` | `8518667` | `True` |
| `fcproj_attnhi_fp16` | `2.43795634` | `0.32865634` | `8167307` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.43795634` | `0.32865634` | `8167307` | `True` |
| `fchi_attnhi_fp16` | `2.43795690` | `0.32865690` | `8708512` | `True` |
| `fcproj_hi` | `2.44168587` | `0.33238587` | `5585628` | `True` |
| `fchi_only` | `2.44169350` | `0.33239350` | `6122646` | `True` |
| `proj_top5_attn_top6_fp16` | `2.27159515` | `0.16229515` | `16086317` | `False` |
| `proj_top6_attn_top5_fp16` | `2.26131451` | `0.15201451` | `16629511` | `False` |
| `proj_top6_attn_top6_fp16` | `2.26079975` | `0.15149975` | `17146518` | `False` |
