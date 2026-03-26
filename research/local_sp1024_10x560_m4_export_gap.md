# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/local_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_compiled_longtail_fchi_top4_attn_top4_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.58580000`
- Shipped `val_bpb`: `2.67617117`
- Export gap: `0.09037117`
- Total bytes: `15036838`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.78245296`
- Reference export gap: `0.19665296`
- Shipped delta vs best: `0.10628179`
- Gap delta vs best: `0.10628179`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.67617117` | `0.09037117` | `15036838` | `True` |
| `tok_attnhi_fp16` | `2.68988161` | `0.10408161` | `10053612` | `True` |
| `int8_tok_fp16` | `2.70624914` | `0.12044914` | `7235086` | `True` |
| `int8_all` | `2.71414616` | `0.12834616` | `6645377` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.78245296` | `0.19665296` | `7629350` | `True` |
| `fcproj_attnhi_fp16` | `2.80553748` | `0.21973748` | `7640594` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.80553748` | `0.21973748` | `7640594` | `True` |
| `fchi_attnhi_fp16` | `2.80553777` | `0.21973777` | `7877793` | `True` |
| `fchi_only` | `2.83320091` | `0.24740091` | `5046949` | `True` |
| `fcproj_hi` | `2.83320261` | `0.24740261` | `4812093` | `True` |
