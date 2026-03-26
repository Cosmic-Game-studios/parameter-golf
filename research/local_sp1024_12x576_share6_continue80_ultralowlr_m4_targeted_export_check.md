# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.13780000`
- Shipped `val_bpb`: `2.14541706`
- Export gap: `0.00761706`
- Total bytes: `15155616`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.14541706`
- Reference export gap: `0.00761706`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.14541706` | `0.00761706` | `15155616` | `True` |
| `proj_top6_attn_top5_fp16` | `2.14838321` | `0.01058321` | `14625094` | `True` |
| `proj_top5_attn_top6_fp16` | `2.15910406` | `0.02130406` | `14038903` | `True` |
| `projhi_attnhi_fp16` | `2.20244434` | `0.06464434` | `10232541` | `True` |
| `int8_all` | `2.24681880` | `0.10901880` | `5357079` | `True` |
