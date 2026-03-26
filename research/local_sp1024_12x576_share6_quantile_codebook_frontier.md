# Fixed Export Gap Search

- Preset: `local_sp1024_12x576_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_codebook_qer192_tail40_retry1_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.13480000`
- Shipped `val_bpb`: `2.14182222`
- Export gap: `0.00702222`
- Total bytes: `15169596`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.14182222`
- Reference export gap: `0.00702222`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.14182222` | `0.00702222` | `15169596` | `True` |
| `codebook_qer_projattn_top6_r192` | `2.14263403` | `0.00783403` | `13340674` | `True` |
| `codebook_qer_projattn_top6_r128` | `2.14274162` | `0.00794162` | `11301956` | `True` |
| `quantile_codebook_qer_projattn_top6_r224` | `2.14345714` | `0.00865714` | `14482823` | `True` |
| `quantile_codebook_qer_projattn_top6_r192` | `2.14379245` | `0.00899245` | `13459660` | `True` |
| `quantile_codebook_qer_projattn_top6_r160` | `2.14442129` | `0.00962129` | `12435879` | `True` |
| `quantile_codebook_qer_projattn_top6_r128` | `2.14481604` | `0.01001604` | `11411562` | `True` |
| `codebook_projattn_top6` | `2.14488480` | `0.01008480` | `7219102` | `True` |
| `quantile_codebook_projattn_top6` | `2.15687631` | `0.02207631` | `7305929` | `True` |
