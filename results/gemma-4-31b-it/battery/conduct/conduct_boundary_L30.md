# Conduct lever (D-063 item 2), gemma-4-31b-it, position boundary, layer 30

Rows 796 (dishonest 349): AR 200/97, DC 200/31, RH 200/82, CF 196/139

| held-out task | transfer AUROC within strata | transfer raw | within-task 5-fold (strata / raw) |
| --- | --- | --- | --- |
| AR | 0.57 | 0.11 | 0.93 / 0.99 |
| CF | 0.39 | 0.22 | 0.78 / 0.82 |
| DC | 0.48 | 0.41 | 0.95 / 0.93 |
| RH | 0.60 | 0.63 | 0.63 / 0.63 |

Cosine of the pooled conduct direction with the item 4 directions (mean / max |cos| per group):

- act cells (c): +0.010 / 0.036
- self-describe cells (d): -0.016 / 0.039
- own-reply cells (a, b): -0.011 / 0.031
- frame levers: -0.012 / 0.033
- shared axes (dm_all, probe_all): -0.011 / 0.021
- all_frames: -0.019 / 0.019
- K1: +0.031 / 0.031
- K2: -0.019 / 0.019
- random (16): +0.000 / 0.038

Controls:

- family probe RH (one-off vs conflicting, neutral prompt, n 100): cos with conduct -0.007; family direction AUROC for conduct 0.36 (raw 0.39); conduct direction AUROC for the family split on neutral rows 0.51
- family probe FT (error vs no_call, neutral prompt, n 100): cos with conduct -0.032; family direction AUROC for conduct 0.53 (raw 0.63); conduct direction AUROC for the family split on neutral rows 0.02
- family probe CF (fictitious_reference vs kuq, neutral prompt, n 100): cos with conduct +0.032; family direction AUROC for conduct 0.52 (raw 0.54); conduct direction AUROC for the family split on neutral rows 0.89
- reply length: cos with conduct +0.066, corr(conduct score, log length) -0.32; length direction AUROC for conduct 0.55 (raw 0.34); conduct probe after projecting length out: train 0.97, LOTO AR 0.56, CF 0.38, DC 0.49, RH 0.60, cos with the original 0.998
