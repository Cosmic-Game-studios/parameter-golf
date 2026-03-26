# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/lab_sp1024_10x560_joint_boundary_recovery_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.10710000`
- Shipped `val_bpb`: `2.27104283`
- Export gap: `0.16394283`
- Total bytes: `15570847`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.38264756`
- Reference export gap: `0.27554756`
- Shipped delta vs best: `0.11160473`
- Gap delta vs best: `0.11160473`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.27104283` | `0.16394283` | `15570847` | `True` |
| `tok_attnhi_fp16` | `2.27690467` | `0.16980467` | `10881179` | `True` |
| `int8_tok_fp16` | `2.27887279` | `0.17177279` | `8315090` | `True` |
| `int8_all` | `2.28248965` | `0.17538965` | `7768717` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.28824236` | `0.18114236` | `15765324` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.29897239` | `0.19187239` | `15766206` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.32102712` | `0.21392712` | `15442900` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.38264756` | `0.27554756` | `8530411` | `True` |
| `fchi_attnhi_fp16` | `2.43683759` | `0.32973759` | `8720832` | `True` |
| `fcproj_attnhi_fp16` | `2.43684126` | `0.32974126` | `8168511` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.43684126` | `0.32974126` | `8168511` | `True` |
| `fcproj_hi` | `2.44061287` | `0.33351287` | `5595570` | `True` |
| `fchi_only` | `2.44062162` | `0.33352162` | `6137947` | `True` |
| `proj_top5_attn_top6_fp16` | `2.27174028` | `0.16464028` | `16087789` | `False` |
| `proj_top6_attn_top5_fp16` | `2.26094362` | `0.15384362` | `16629831` | `False` |
| `proj_top6_attn_top6_fp16` | `2.26134613` | `0.15424613` | `17146643` | `False` |
