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
- Total bytes: `15159153`
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
| `proj_top6_attn_top6_fp16` | `2.14541706` | `0.00761706` | `15159153` | `True` |
| `qer_projattn_top6_r320` | `2.15800706` | `0.02020706` | `15571747` | `True` |
| `qer_projattn_top6_r256` | `2.16664007` | `0.02884007` | `13526942` | `True` |
| `qer_projattn_top6_r192` | `2.17947545` | `0.04167545` | `11484305` | `True` |
| `qer_projattn_top6_r128` | `2.19538766` | `0.05758766` | `9442835` | `True` |
| `qer_projattn_top6_r64` | `2.21784843` | `0.08004843` | `7402684` | `True` |
