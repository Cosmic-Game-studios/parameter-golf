# Priorities

1. Keep [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt) as the reproduced local regression bar.
2. Treat [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt) as the first post-reset `revert`: sparse top-decoder-only `cross_skip` did not help.
3. Do not spend the next cycle on another near-clone of the same donor unless it is materially more orthogonal.
4. Highest-EV next local move: one orthogonal branch that targets stronger scaling, not a smaller micro-control tweak.
5. Keep official H100 work parked until the next local branch is either reproduced or clearly promotion-worthy.
