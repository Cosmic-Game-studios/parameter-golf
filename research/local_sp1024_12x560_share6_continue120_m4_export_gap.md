# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_continue120_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.45420000`
- Shipped `val_bpb`: `2.45636399`
- Export gap: `0.00216399`
- Total bytes: `14277891`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.45636399`
- Reference export gap: `0.00216399`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.45636399` | `0.00216399` | `14277891` | `True` |
| `proj_top6_attn_top5_fp16` | `2.45720771` | `0.00300771` | `13752547` | `True` |
| `proj_top5_attn_top6_fp16` | `2.46172787` | `0.00752787` | `13201511` | `True` |
| `projhi_attnhi_fp16` | `2.47619477` | `0.02199477` | `9473579` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.48683134` | `0.03263134` | `6792043` | `True` |
| `tok_attnhi_fp16` | `2.48864528` | `0.03444528` | `6804978` | `True` |
| `int8_tok_fp16` | `2.49004018` | `0.03584018` | `5217915` | `True` |
| `int8_all` | `2.49116599` | `0.03696599` | `4654094` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.50625552` | `0.05205552` | `6472561` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.50625552` | `0.05205552` | `6472561` | `True` |
| `fcproj_attnhi_fp16` | `2.65796485` | `0.20376485` | `5034463` | `True` |
| `fchi_attnhi_fp16` | `2.65796485` | `0.20376485` | `5279705` | `True` |
| `fcproj_hi` | `2.66420663` | `0.21000663` | `3442959` | `True` |
| `fchi_only` | `2.66420663` | `0.21000663` | `3687258` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.99259352` | `0.53839352` | `5482955` | `True` |
| `fcproj_top5_attn_top5_fp16` | `3.86776320` | `1.41356320` | `5278932` | `True` |
