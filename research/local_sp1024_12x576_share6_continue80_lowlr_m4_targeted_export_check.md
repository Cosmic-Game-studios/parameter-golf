# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_continue80_lowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.17140000`
- Shipped `val_bpb`: `2.17698622`
- Export gap: `0.00558622`
- Total bytes: `15151562`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.17698622`
- Reference export gap: `0.00558622`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.17698622` | `0.00558622` | `15151562` | `True` |
| `proj_top6_attn_top5_fp16` | `2.17849845` | `0.00709845` | `14619480` | `True` |
| `proj_top5_attn_top6_fp16` | `2.18831953` | `0.01691953` | `14034570` | `True` |
| `projhi_attnhi_fp16` | `2.22356768` | `0.05216768` | `10214038` | `True` |
| `int8_all` | `2.25662831` | `0.08522831` | `5318713` | `True` |
