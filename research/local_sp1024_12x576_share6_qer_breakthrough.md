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
| `qer_projattn_top6_r64` | `2.21784843` | `0.08004843` | `7402684` | `True` |
| `qer_projattn_top6_r32` | `2.23367155` | `0.09587155` | `6382553` | `True` |
| `qer_projattn_top6_r16` | `2.23750994` | `0.09970994` | `5872063` | `True` |
| `qer_projattn_top6_r8` | `2.24230344` | `0.10450344` | `5617093` | `True` |
| `int8_all` | `2.24681880` | `0.10901880` | `5360616` | `True` |
