# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_896ctx_modern_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.56280000`
- Shipped `val_bpb`: `2.56620786`
- Export gap: `0.00340786`
- Total bytes: `14233452`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.56620786`
- Reference export gap: `0.00340786`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.56620786` | `0.00340786` | `14233452` | `True` |
| `proj_top6_attn_top5_fp16` | `2.56913829` | `0.00633829` | `13674127` | `True` |
| `proj_top5_attn_top6_fp16` | `2.57303216` | `0.01023216` | `13126654` | `True` |
| `projhi_attnhi_fp16` | `2.61956490` | `0.05676490` | `9228427` | `True` |
| `tok_attnhi_fp16` | `2.66180140` | `0.09900140` | `6472682` | `True` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `2.66602508` | `0.10322508` | `6443604` | `True` |
| `int8_tok_fp16` | `2.68454342` | `0.12174342` | `4787551` | `True` |
| `int8_all` | `2.69119321` | `0.12839321` | `4202217` | `True` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `2.71834470` | `0.15554470` | `6124897` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `2.71834470` | `0.15554470` | `6124897` | `True` |
| `fcproj_attnhi_fp16` | `2.96153968` | `0.39873968` | `4790511` | `True` |
| `fchi_attnhi_fp16` | `2.96153968` | `0.39873968` | `4935153` | `True` |
| `fcproj_hi` | `3.03400578` | `0.47120578` | `3098449` | `True` |
| `fchi_only` | `3.03400578` | `0.47120578` | `3236499` | `True` |
| `fchi_top4_attn_top4_fp16` | `3.29632054` | `0.73352054` | `5175524` | `True` |
| `fcproj_top5_attn_top5_fp16` | `3.84274868` | `1.27994868` | `5168030` | `True` |
