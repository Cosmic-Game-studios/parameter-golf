# BPB Frontier Search

- Checkpoint: `logs/dense_u4k_12x512_kv2_m2_local_cosine_300_lr30_mlx_model.npz`
- Baseline total bytes: `15484288`
- Baseline val_bpb: `2.05310549`

## Top Single Candidates

| Name | Bytes saved | bpb delta | total bytes | val_bpb |
| --- | --- | --- | --- | --- |
| `mlp.fc.weight__layer_10` | `286632` | `0.00000665` | `15197656` | `2.05311214` |
| `mlp.fc.weight__layer_11` | `287721` | `0.00001393` | `15196567` | `2.05311942` |
| `mlp.proj.weight__layer_11` | `261824` | `0.00001351` | `15222464` | `2.05311900` |
| `mlp.proj.weight__layer_7` | `257631` | `0.00002246` | `15226657` | `2.05312795` |
| `mlp.proj.weight__layer_10` | `261226` | `0.00004076` | `15223062` | `2.05314625` |
| `mlp.proj.weight__layer_9` | `260348` | `0.00008651` | `15223940` | `2.05319200` |
| `mlp.proj.weight__layer_8` | `259983` | `0.00009025` | `15224305` | `2.05319574` |
| `mlp.fc.weight__layer_9` | `289472` | `0.00013122` | `15194816` | `2.05323671` |

## Best Greedy Combo

- Name: `greedy_step_4__mlp.fc.weight__layer_9__mlp.fc.weight__layer_11__mlp.fc.weight__layer_10__mlp.proj.weight__layer_10`
- Patterns: `blocks.10.mlp.fc.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.9.mlp.fc.weight`
- Total bytes: `14354342`
- Val bpb: `2.05328808`
- Bytes saved vs baseline: `1129946`
- bpb delta vs baseline: `0.00018259`
