# Conduct lever (D-063 item 2), qwen3.5-27b, position reply32, layer 32

Rows 799 (dishonest 185): AR 200/82, DC 200/8, RH 200/18, CF 199/77

| held-out task | transfer AUROC within strata | transfer raw | within-task 5-fold (strata / raw) |
| --- | --- | --- | --- |
| AR | 0.62 | 0.54 | 0.77 / 0.88 |
| CF | 0.40 | 0.19 | 0.98 / 0.99 |
| DC | 0.78 | 0.59 | 0.99 / 0.97 |
| RH | 0.16 | 0.22 | 0.82 / 0.82 |

Cosine of the pooled conduct direction with the item 4 directions (mean / max |cos| per group):

- act cells (c): +0.021 / 0.134
- self-describe cells (d): -0.000 / 0.065
- own-reply cells (a, b): +0.013 / 0.052
- frame levers: +0.023 / 0.044
- shared axes (dm_all, probe_all): +0.046 / 0.047
- all_frames: +0.040 / 0.040
- K1: +0.024 / 0.024
- K2: -0.029 / 0.029
- random (16): +0.000 / 0.031

Controls:

- family probe RH (one-off vs conflicting, neutral prompt, n 100): cos with conduct -0.149; family direction AUROC for conduct 0.34 (raw 0.33); conduct direction AUROC for the family split on neutral rows 0.40
- family probe FT (error vs no_call, neutral prompt, n 100): cos with conduct +0.006; family direction AUROC for conduct 0.51 (raw 0.63); conduct direction AUROC for the family split on neutral rows 0.77
- family probe CF (fictitious_reference vs kuq, neutral prompt, n 100): cos with conduct +0.106; family direction AUROC for conduct 0.66 (raw 0.60); conduct direction AUROC for the family split on neutral rows 1.00
- reply length: cos with conduct +0.034, corr(conduct score, log length) -0.38; length direction AUROC for conduct 0.45 (raw 0.31); conduct probe after projecting length out: train 0.98, LOTO AR 0.62, CF 0.41, DC 0.80, RH 0.16, cos with the original 0.999
