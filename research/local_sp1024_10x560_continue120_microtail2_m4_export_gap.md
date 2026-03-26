# Fixed Export Gap Search

- Preset: `local_sp1024_10x560_m4`
- Checkpoint: `./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.12670000`
- Shipped `val_bpb`: `2.28205904`
- Export gap: `0.15535904`
- Total bytes: `15571706`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`

## Reference Policy Comparison

- Reference policy: `fchi_top4_attn_top4_fp16`
- Reference shipped `val_bpb`: `2.38667499`
- Reference export gap: `0.25997499`
- Shipped delta vs best: `0.10461595`
- Gap delta vs best: `0.10461595`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.28205904` | `0.15535904` | `15571706` | `True` |
| `int8_tok_fp16` | `2.29205336` | `0.16535336` | `8230877` | `True` |
| `tok_attnhi_fp16` | `2.29228405` | `0.16558405` | `10826808` | `True` |
| `int8_all` | `2.29550970` | `0.16880970` | `7686851` | `True` |
| `fchi_top4_attn_top4_fp16` | `2.38667499` | `0.25997499` | `8469991` | `True` |
| `fchi_only` | `2.43605910` | `0.30935910` | `6055135` | `True` |
| `fcproj_hi` | `2.43607068` | `0.30937068` | `5563625` | `True` |
| `fcproj_attnhi_fp16` | `2.43617628` | `0.30947628` | `8165872` | `True` |
| `fcproj_top5_attn_top5_fp16` | `2.43617628` | `0.30947628` | `8165872` | `True` |
| `fchi_attnhi_fp16` | `2.43618984` | `0.30948984` | `8665541` | `True` |
