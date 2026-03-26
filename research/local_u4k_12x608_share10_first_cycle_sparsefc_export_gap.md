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
| `fc_top3_int4` | `1.98570883` | `-0.00119117` | `15280736` | `True` |
| `fc_top2_int4` | `1.98624331` | `-0.00065669` | `15677726` | `True` |
| `fchi_only` | `1.98721868` | `0.00031868` | `14491839` | `True` |
| `fc_top1_int4` | `1.98739108` | `0.00049108` | `16074014` | `False` |
| `int8_all` | `1.98805305` | `0.00115305` | `16470409` | `False` |
