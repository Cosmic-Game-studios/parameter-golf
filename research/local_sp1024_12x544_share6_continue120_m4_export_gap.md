# Fixed Export Gap Search

- Preset: `local_sp1024_12x544_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x544_kv2_share6_continue120_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.47930000`
- Shipped `val_bpb`: `2.48159904`
- Export gap: `0.00229904`
- Total bytes: `13507819`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.48159904`
- Reference export gap: `0.00229904`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.48159904` | `0.00229904` | `13507819` | `True` |
| `proj_top6_attn_top5_fp16` | `2.48219371` | `0.00289371` | `13014708` | `True` |
| `proj_top5_attn_top6_fp16` | `2.48567730` | `0.00637730` | `12495236` | `True` |
| `projhi_attnhi_fp16` | `2.49622463` | `0.01692463` | `8986705` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.50443706` | `0.02513706` | `6463419` | `True` |
| `tok_attnhi_fp16` | `2.50445598` | `0.02515598` | `6489078` | `True` |
| `int8_tok_fp16` | `2.50538864` | `0.02608864` | `4999699` | `True` |
| `int8_all` | `2.50687193` | `0.02757193` | `4456028` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.51954070` | `0.04024070` | `6158888` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.51954070` | `0.04024070` | `6158888` | `True` |
| `fcproj_attnhi_fp16` | `2.65910986` | `0.17980986` | `4793806` | `True` |
| `fchi_attnhi_fp16` | `2.65910986` | `0.17980986` | `5031782` | `True` |
| `fcproj_hi` | `2.66148233` | `0.18218233` | `3299484` | `True` |
| `fchi_only` | `2.66148233` | `0.18218233` | `3533749` | `True` |
| `fchi_top4_attn_top4_fp16` | `3.01028253` | `0.53098253` | `5220605` | `True` |
| `fcproj_top5_attn_top5_fp16` | `3.89220762` | `1.41290762` | `5011984` | `True` |
