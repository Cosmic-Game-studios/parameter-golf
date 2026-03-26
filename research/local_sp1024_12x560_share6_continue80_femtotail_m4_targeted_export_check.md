# Fixed Export Gap Search

- Preset: `local_sp1024_12x560_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x560_kv2_share6_continue80_femtotail_m4_mlx_model.npz`
- Counted code path: `train_gpt.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `proj_top6_attn_top6_fp16`
- Clean `val_bpb`: `2.17900000`
- Shipped `val_bpb`: `2.18650446`
- Export gap: `0.00750446`
- Total bytes: `14310286`
- Int4 groups: `-`
- FP16 groups: `proj_top6,attn_top6`

## Reference Policy Comparison

- Reference policy: `proj_top6_attn_top6_fp16`
- Reference shipped `val_bpb`: `2.18650446`
- Reference export gap: `0.00750446`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `proj_top6_attn_top6_fp16` | `2.18650446` | `0.00750446` | `14310286` | `True` |
| `proj_top6_attn_top5_fp16` | `2.18926575` | `0.01026575` | `13805296` | `True` |
| `proj_top5_attn_top6_fp16` | `2.20812815` | `0.02912815` | `13247973` | `True` |
| `projhi_attnhi_fp16` | `2.25661588` | `0.07761588` | `9605332` | `True` |
| `int8_all` | `2.30635564` | `0.12735564` | `4925063` | `True` |
