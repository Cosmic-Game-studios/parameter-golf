# Fixed Export Gap Search

- Preset: `local_sp1024_12x576_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_ultralow_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.13240000`
- Shipped `val_bpb`: `2.14029037`
- Export gap: `0.00789037`
- Total bytes: `15168592`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.14029037`
- Reference export gap: `0.00789037`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.14029037` | `0.00789037` | `15168592` | `True` |
| `codebook_qer_projattn_top6_r192` | `2.14138257` | `0.00898257` | `13341510` | `True` |
