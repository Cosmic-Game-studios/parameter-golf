# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_eval0_int8all_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fc_top4_int4`
- Clean `val_bpb`: `1.98690000`
- Shipped `val_bpb`: `1.98518496`
- Export gap: `-0.00171504`
- Total bytes: `14890142`
- Int4 groups: `fc_top4`
- FP16 groups: `-`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fc_top4_int4` | `1.98518496` | `-0.00171504` | `14890142` | `True` |
| `fc_top4_plus_block1_int4` | `1.98712239` | `0.00022239` | `14497181` | `True` |
| `fc_top4_plus_block5_int4` | `1.98721868` | `0.00031868` | `14491839` | `True` |
| `fchi_only` | `1.98721868` | `0.00031868` | `14491839` | `True` |
| `fc_top4_plus_block3_int4` | `1.98744183` | `0.00054183` | `14499747` | `True` |
| `fc_top4_plus_block2_int4` | `1.98761756` | `0.00071756` | `14500974` | `True` |
| `fc_top4_plus_block4_int4` | `1.98911784` | `0.00221784` | `14498216` | `True` |
| `fc_top4_plus_block0_int4` | `2.01679362` | `0.02989362` | `14501066` | `True` |
