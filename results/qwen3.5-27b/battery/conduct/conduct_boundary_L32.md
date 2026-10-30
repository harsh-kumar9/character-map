# Conduct lever (D-063 item 2), qwen3.5-27b, position boundary, layer 32

Rows 799 (dishonest 185): AR 200/82, DC 200/8, RH 200/18, CF 199/77

| held-out task | transfer AUROC within strata | transfer raw | within-task 5-fold (strata / raw) |
| --- | --- | --- | --- |
| AR | 0.57 | 0.23 | 0.77 / 0.85 |
| CF | 0.22 | 0.41 | 0.89 / 0.97 |
| DC | 0.93 | 0.86 | 0.97 / 0.97 |
| RH | 0.31 | 0.53 | 0.64 / 0.70 |

Cosine of the pooled conduct direction with the item 4 directions (mean / max |cos| per group):

- act cells (c): +0.005 / 0.041
- self-describe cells (d): +0.004 / 0.047
- own-reply cells (a, b): +0.002 / 0.050
- frame levers: +0.008 / 0.017
- shared axes (dm_all, probe_all): +0.012 / 0.018
- all_frames: +0.013 / 0.013
- K1: -0.020 / 0.020
- K2: -0.003 / 0.003
- random (16): +0.001 / 0.026

Controls:

- family probe RH (one-off vs conflicting, neutral prompt, n 100): cos with conduct -0.158; family direction AUROC for conduct 0.34 (raw 0.17); conduct direction AUROC for the family split on neutral rows 0.25
- family probe FT (error vs no_call, neutral prompt, n 100): cos with conduct +0.001; family direction AUROC for conduct 0.46 (raw 0.47); conduct direction AUROC for the family split on neutral rows 0.55
- family probe CF (fictitious_reference vs kuq, neutral prompt, n 100): cos with conduct +0.088; family direction AUROC for conduct 0.61 (raw 0.82); conduct direction AUROC for the family split on neutral rows 1.00
- reply length: cos with conduct +0.033, corr(conduct score, log length) -0.33; length direction AUROC for conduct 0.44 (raw 0.30); conduct probe after projecting length out: train 0.99, LOTO AR 0.57, CF 0.26, DC 0.93, RH 0.32, cos with the original 0.999
