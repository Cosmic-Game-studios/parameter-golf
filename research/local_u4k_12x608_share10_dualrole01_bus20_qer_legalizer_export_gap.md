# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fc_top4_attn_top1_qer_proj_top1_r64`
- Clean `val_bpb`: `1.98710000`
- Shipped `val_bpb`: `1.98453631`
- Export gap: `-0.00256369`
- Total bytes: `15637017`
- Int4 groups: `fc_top4`
- FP16 groups: `attn_top1`
- Low-rank groups: `proj_top1`
- Low-rank rank: `64`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fc_top4_attn_top1_qer_proj_top1_r64` | `1.98453631` | `-0.00256369` | `15637017` | `True` |
| `fc_top4_attn_top1_fp16` | `1.98453797` | `-0.00256203` | `15420393` | `True` |
| `fc_top4_int4` | `1.98454462` | `-0.00255538` | `14937014` | `True` |
| `fc_top4_attn_top1_qer_proj_top1_r32` | `1.98454525` | `-0.00255475` | `15529090` | `True` |
| `fc_top4_attn_top1_qer_proj_top1_r96` | `1.98454608` | `-0.00255392` | `15745265` | `True` |
