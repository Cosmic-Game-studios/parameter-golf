# Fixed Export Gap Search

- Preset: `local_sp1024_12x576_share6_m4`
- Checkpoint: `./logs/lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_lowlr_m4_mlx_model.npz`
- Counted code path: `train_gpt.py,quant_reconstruction.py`
- Legal budget: `16000000`

## Best Legal Policy

- Name: `lloyd_codebook_qer_projattn_top6_r224`
- Clean `val_bpb`: `2.13120000`
- Shipped `val_bpb`: `2.13943493`
- Export gap: `0.00823493`
- Total bytes: `14409932`
- Int4 groups: `proj_top6,attn_top6`
- FP16 groups: `-`
- Low-rank groups: `proj_top6,attn_top6`
- Low-rank rank: `224`

## Top Policies

| Name | Shipped bpb | Gap | Total bytes | Legal |
| --- | --- | --- | --- | --- |
| `lloyd_codebook_qer_projattn_top6_r224` | `2.13943493` | `0.00823493` | `14409932` | `True` |
| `codebook_qer_projattn_top6_r192` | `2.13976149` | `0.00856149` | `13346786` | `True` |
| `lloyd_codebook_qer_projattn_top6_r192` | `2.13976629` | `0.00856629` | `13390408` | `True` |
| `mulaw_codebook_qer_projattn_top6_r192` | `2.13997765` | `0.00877765` | `13366348` | `True` |
| `lloyd_codebook_projattn_top6` | `2.14291344` | `0.01171344` | `7266152` | `True` |
