# Fixed Export Gap Search

- Preset: `local_u4k_12x608_lfqat60_m4`
- Checkpoint: `./logs/lab_u4k_12x608_kv2_share9_first_cycle_eval0_int8all_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `int8_all`
- Clean `val_bpb`: `1.98740000`
- Shipped `val_bpb`: `1.98858087`
- Export gap: `0.00118087`
- Total bytes: `15087368`
- Int4 groups: `-`
- FP16 groups: `-`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `fcproj_hi`
- Reference shipped `val_bpb`: `2.00097286`
- Reference export gap: `0.01357286`
- Shipped delta vs best: `0.01239199`
- Gap delta vs best: `0.01239199`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `int8_all` | `1.98858087` | `0.00118087` | `15087368` | `True` |
| `fchi_attnhi_fp16` | `1.99400343` | `0.00660343` | `15582389` | `True` |
| `fchi_only` | `1.99401009` | `0.00661009` | `13114575` | `True` |
| `fcproj_hi` | `2.00097286` | `0.01357286` | `11420445` | `True` |
| `projhi_attnhi_fp16` | `1.98856007` | `0.00116007` | `22630537` | `False` |
