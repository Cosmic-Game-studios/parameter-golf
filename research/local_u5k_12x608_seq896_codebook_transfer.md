# Fixed Export Gap Search

- Preset: `local_u5k_12x608_seq896_exactcopy_m4`
- Checkpoint: `./logs/lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `fcproj_hi`
- Clean `val_bpb`: `2.10870000`
- Shipped `val_bpb`: `2.10282351`
- Export gap: `-0.00587649`
- Total bytes: `15391582`
- Int4 groups: `fc_hi,proj_hi`
- FP16 groups: `-`
- Low-rank groups: `-`
- Low-rank rank: `0`

## Reference Policy Comparison

- Reference policy: `fcproj_hi`
- Reference shipped `val_bpb`: `2.10282351`
- Reference export gap: `-0.00587649`
- Shipped delta vs best: `0.00000000`
- Gap delta vs best: `0.00000000`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `fcproj_hi` | `2.10282351` | `-0.00587649` | `15391582` | `True` |
| `codebook_projattn_top6` | `2.11013217` | `0.00143217` | `19606731` | `False` |
| `codebook_qer_projattn_top6_r128` | `2.10975498` | `0.00105498` | `23959556` | `False` |
| `codebook_qer_projattn_top6_r192` | `2.10979250` | `0.00109250` | `26125898` | `False` |
| `mulaw_codebook_qer_projattn_top6_r192` | `2.10977233` | `0.00107233` | `26231667` | `False` |
| `lloyd_codebook_qer_projattn_top6_r224` | `2.10974429` | `0.00104429` | `27346667` | `False` |
| `projhi_attnhi_fp16` | `2.10979956` | `0.00109956` | `28801897` | `False` |
| `proj_top6_attn_top6_fp16` | `2.10979956` | `0.00109956` | `28801897` | `False` |
