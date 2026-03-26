# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_eval0_int8all_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `int8_all`
- Clean `val_bpb`: `1.98690000`
- Shipped `val_bpb`: `1.98805305`
- Export gap: `0.00115305`
- Total bytes: `16470409`
- Int4 groups: `-`
- FP16 groups: `-`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `int8_all` | `1.98805305` | `0.00115305` | `16470409` | `False` |
| `qer_projattn_top6_r64` | `1.98805097` | `0.00115097` | `18627938` | `False` |
| `proj_top5_attn_top6_fc5_fc9_int4_fp16` | `1.98670458` | `-0.00019542` | `23724367` | `False` |
| `proj_top5_attn_top6_fc9_int4_fp16` | `1.98742748` | `0.00052748` | `24115089` | `False` |
| `proj_top5_attn_top6_fc5_int4_fp16` | `1.98692752` | `0.00002752` | `24118685` | `False` |
| `proj_top5_attn_top6_fp16` | `1.98805284` | `0.00115284` | `24512208` | `False` |
