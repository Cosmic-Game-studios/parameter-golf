# Fixed Export Gap Search

- Preset: `local_sp1024_12x576_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.13780000`
- Shipped `val_bpb`: `2.14541706`
- Export gap: `0.00761706`
- Total bytes: `15168374`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.14541706`
- Reference export gap: `0.00761706`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.14541706` | `0.00761706` | `15168374` | `True` |
| `codebook_qer_projattn_top6_r192` | `2.14562503` | `0.00782503` | `13337365` | `True` |
| `codebook_qer_projattn_top6_r128` | `2.14647369` | `0.00867369` | `11299287` | `True` |
| `codebook_qer_projattn_top6_r64` | `2.14730823` | `0.00950823` | `9259603` | `True` |
| `codebook_projattn_top6` | `2.14803957` | `0.01023957` | `7216082` | `True` |
| `qer_projattn_top6_r320_proj_top1_fp16` | `2.15715600` | `0.01935600` | `15649008` | `True` |
| `qer_projattn_top6_r320_attn_top1_fp16` | `2.15784060` | `0.02004060` | `15429223` | `True` |
| `qer_projattn_top6_r320` | `2.15800706` | `0.02020706` | `15580968` | `True` |
| `qer_projattn_top6_r288_proj_top1_fp16` | `2.16121265` | `0.02341265` | `14729081` | `True` |
| `qer_projattn_top6_r288_attn_top1_fp16` | `2.16217200` | `0.02437200` | `14474898` | `True` |
