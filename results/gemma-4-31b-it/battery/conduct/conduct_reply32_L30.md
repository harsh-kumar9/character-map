# Conduct lever (D-063 item 2), gemma-4-31b-it, position reply32, layer 30

Rows 796 (dishonest 349): AR 200/97, DC 200/31, RH 200/82, CF 196/139

| held-out task | transfer AUROC within strata | transfer raw | within-task 5-fold (strata / raw) |
| --- | --- | --- | --- |
| AR | 0.50 | 0.36 | 0.97 / 0.99 |
| CF | 0.13 | 0.23 | 1.00 / 0.99 |
| DC | 0.42 | 0.46 | 0.96 / 0.93 |
| RH | 0.67 | 0.63 | 0.80 / 0.79 |

Cosine of the pooled conduct direction with the item 4 directions (mean / max |cos| per group):

- act cells (c): -0.002 / 0.057
- self-describe cells (d): -0.026 / 0.059
- own-reply cells (a, b): -0.001 / 0.065
- frame levers: -0.012 / 0.046
- shared axes (dm_all, probe_all): -0.014 / 0.016
- all_frames: -0.020 / 0.020
- K1: -0.004 / 0.004
- K2: +0.006 / 0.006
- random (16): +0.005 / 0.036

Controls:

- family probe RH (one-off vs conflicting, neutral prompt, n 100): cos with conduct -0.042; family direction AUROC for conduct 0.54 (raw 0.50); conduct direction AUROC for the family split on neutral rows 0.55
- family probe FT (error vs no_call, neutral prompt, n 100): cos with conduct +0.027; family direction AUROC for conduct 0.38 (raw 0.57); conduct direction AUROC for the family split on neutral rows 0.81
- family probe CF (fictitious_reference vs kuq, neutral prompt, n 100): cos with conduct +0.018; family direction AUROC for conduct 0.46 (raw 0.49); conduct direction AUROC for the family split on neutral rows 0.61
- reply length: cos with conduct +0.023, corr(conduct score, log length) -0.37; length direction AUROC for conduct 0.58 (raw 0.35); conduct probe after projecting length out: train 1.00, LOTO AR 0.50, CF 0.12, DC 0.42, RH 0.67, cos with the original 1.000
