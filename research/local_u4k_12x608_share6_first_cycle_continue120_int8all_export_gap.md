# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share6_first_cycle_continue120_int8all_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `projhi_attnhi_fp16`
- Clean `val_bpb`: `2.00040000`
- Shipped `val_bpb`: `2.00187835`
- Export gap: `0.00147835`
- Total bytes: `15423066`
- Int4 groups: `-`
- FP16 groups: `proj_hi,attn_hi`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `fcproj_hi`
- Reference shipped `val_bpb`: `2.03759065`
- Reference export gap: `0.03719065`
- Shipped delta vs best: `0.03571230`
- Gap delta vs best: `0.03571230`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `projhi_attnhi_fp16` | `2.00187835` | `0.00147835` | `15423066` | `True` |
| `int8_all` | `2.00194844` | `0.00154844` | `10988599` | `True` |
| `fchi_only` | `2.00325427` | `0.00285427` | `9814025` | `True` |
| `fchi_attnhi_fp16` | `2.00330959` | `0.00290959` | `11270859` | `True` |
| `fcproj_hi` | `2.03759065` | `0.03719065` | `8746449` | `True` |
