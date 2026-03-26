# Extended Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_tail20_fc_top4_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fc_top4_int4`
- Clean `val_bpb`: `1.98710000`
- Shipped `val_bpb`: `1.98455440`
- Export gap: `-0.00254560`
- Total bytes: `14890371`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fc_top4_int4` | `1.98455440` | `-0.00254560` | `14890371` | `True` |
| `fc_top3_int4` | `1.98542579` | `-0.00167421` | `15280900` | `True` |
| `fc_top4_plus_block5_int4` | `1.98607194` | `-0.00102806` | `14492290` | `True` |
| `fchi_only` | `1.98607194` | `-0.00102806` | `14492290` | `True` |
| `fc_top4_plus_block1_int4` | `1.98615763` | `-0.00094237` | `14497280` | `True` |
| `fc_top2_int4` | `1.98620858` | `-0.00089142` | `15678122` | `True` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `1.98630923` | `-0.00079077` | `23724576` | `False` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `1.98668191` | `-0.00041809` | `24118939` | `False` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `1.98760904` | `0.00050904` | `24115164` | `False` |
| `quantile_codebook_qer_projattn_top6_r160` | `1.98790373` | `0.00080373` | `22028562` | `False` |
| `quantile_codebook_qer_projattn_top6_r192` | `1.98799190` | `0.00089190` | `23120093` | `False` |
| `quantile_codebook_qer_projattn_top6_r224` | `1.98810254` | `0.00100254` | `24210527` | `False` |

## Notes

- Completed medium-rank QER and codebook-QER policies on this checkpoint all lost to `fc_top4_int4` and/or went over budget.
- The near-cap `proj_top5_attn_top6_*` family improved loss slightly but remained far over the legal byte cap on this `u4k` winner.
