# Fixed Export Gap Search

- Preset: `local_sp1024_12x576_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_lowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.13120000`
- Shipped `val_bpb`: `2.13867183`
- Export gap: `0.00747183`
- Total bytes: `15175610`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.13867183`
- Reference export gap: `0.00747183`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.13867183` | `0.00747183` | `15175610` | `True` |
| `lloyd_codebook_qer_projattn_top6_r224` | `2.13943493` | `0.00823493` | `14409932` | `True` |
| `codebook_qer_projattn_top6_r192` | `2.13976149` | `0.00856149` | `13346786` | `True` |
