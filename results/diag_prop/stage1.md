# D-060 stage 1: direction, dose, hook and spillover anatomy

Columns: gemma-4-31b-it, qwen3.5-27b, gemma-3-4b-it

| check | gemma-4-31b-it | qwen3.5-27b | gemma-3-4b-it |
|---|---|---|---|
| steering layer / readout layers | 30 / 39, 51 | 32 / 42, 54 | 17 / 22, 29 |
| residual multiplier product, injection to readout 3 / 4 | 0.087, 0.007 | 1.000, 1.000 | 1.000, 1.000 |
| median token norm at the 4 depths | 278, 272, 266, 127 | 48, 65, 105, 193 | 10150, 38536, 43970, 66077 |
| top-8 dims' share of token mean square, 4 depths | 0.96, 0.95, 0.80, 0.68 | 0.82, 0.65, 0.66, 0.29 | 1.00, 1.00, 0.98, 0.96 |
| top-32 dims' share of token mean square, 4 depths | 0.97, 0.96, 0.81, 0.69 | 0.83, 0.68, 0.70, 0.38 | 1.00, 1.00, 0.98, 0.96 |
| top-128 dims' share of token mean square, 4 depths | 0.98, 0.97, 0.83, 0.71 | 0.85, 0.73, 0.75, 0.46 | 1.00, 1.00, 0.98, 0.97 |
| injected raw diff of means (S5a): median share on top-8 outlier dims | 0.085 | 0.025 | 0.636 |
| injected raw diff of means (S5a): median share on top-32 outlier dims | 0.169 | 0.055 | 0.643 |
| injected raw diff of means (S5a): median share on top-128 outlier dims | 0.261 | 0.113 | 0.675 |
| standardized direction: median share on top-8 outlier dims | 0.001 | 0.001 | 0.003 |
| standardized direction: median share on top-32 outlier dims | 0.003 | 0.003 | 0.010 |
| standardized direction: median share on top-128 outlier dims | 0.015 | 0.018 | 0.042 |
| primal raw form sd*u: median share on top-8 outlier dims | 0.085 | 0.025 | 0.448 |
| primal raw form sd*u: median share on top-32 outlier dims | 0.169 | 0.055 | 0.466 |
| primal raw form sd*u: median share on top-128 outlier dims | 0.261 | 0.113 | 0.509 |
| dual raw form u/sd (levers |resid): median share on top-8 outlier dims | 0.000 | 0.000 | 0.000 |
| dual raw form u/sd (levers |resid): median share on top-32 outlier dims | 0.000 | 0.000 | 0.001 |
| dual raw form u/sd (levers |resid): median share on top-128 outlier dims | 0.000 | 0.005 | 0.006 |
| probe (standardized): median share on top-8 outlier dims | 0.001 | 0.001 | 0.002 |
| probe (standardized): median share on top-32 outlier dims | 0.003 | 0.003 | 0.009 |
| probe (standardized): median share on top-128 outlier dims | 0.015 | 0.017 | 0.037 |
| cos(standardized, raw diff of means), median | 0.707 | 0.921 | 0.559 |
| cos(diff of means, probe) per cell, median | 0.892 | 0.878 | 0.887 |
| cos(primal raw form, injected vector), median | 1.000 | 1.000 | 1.000 |
| pooled levers cos(dm_all, probe_all) | 0.604 | 0.581 | 0.632 |
| dose 1/256 | 1.1 raw = 0.1 gaps; std 0.017 of token norm = 0.1 std gaps | 0.3 raw = 0.1 gaps; std 0.008 of token norm = 0.1 std gaps | 127.9 raw = 0.3 gaps; std 0.034 of token norm = 0.3 std gaps |
| dose 1/128 | 2.2 raw = 0.3 gaps; std 0.033 of token norm = 0.3 std gaps | 0.6 raw = 0.1 gaps; std 0.016 of token norm = 0.1 std gaps | 255.8 raw = 0.5 gaps; std 0.067 of token norm = 0.5 std gaps |
| dose 1/64 | 4.4 raw = 0.5 gaps; std 0.067 of token norm = 0.5 std gaps | 1.3 raw = 0.2 gaps; std 0.033 of token norm = 0.2 std gaps | 511.6 raw = 1.0 gaps; std 0.134 of token norm = 1.1 std gaps |
| dose 1/32 | 8.8 raw = 1.0 gaps; std 0.134 of token norm = 1.0 std gaps | 2.5 raw = 0.4 gaps; std 0.065 of token norm = 0.4 std gaps | 1023.1 raw = 2.0 gaps; std 0.268 of token norm = 2.2 std gaps |
| dose 1/16 | 17.6 raw = 2.0 gaps; std 0.267 of token norm = 2.1 std gaps | 5.0 raw = 0.8 gaps; std 0.131 of token norm = 0.9 std gaps | 2046.3 raw = 4.0 gaps; std 0.536 of token norm = 4.4 std gaps |
| cell cosines (std), mean |off-diag| / same-def / other-def | 0.105, 0.195, 0.032 | 0.107, 0.191, 0.034 | 0.099, 0.204, 0.014 |
| cell cosines (raw), mean |off-diag| / same-def / other-def | 0.133, 0.159, 0.023 | 0.119, 0.177, 0.032 | 0.442, 0.233, 0.036 |
| native maintenance of the steering-depth gap at depths 3 / 4, median over cells | 0.705, 0.151 | 1.137, 2.237 | 1.119, 1.481 |
| cos(direction at depth 3 / 4, direction at steering depth), raw, median | 0.331, 0.168 | 0.723, 0.479 | 0.702, 0.493 |

| grid run, readout | gemma-4-31b-it | qwen3.5-27b | gemma-3-4b-it |
|---|---|---|---|
| c0.0078125, readout 3 | slope 0.56 (R2 0.47); pass-through 0.087; self z naive/pass/fit -0.3/1.4/0.7; spill 0.00/0.00/0.00; off/diag naive n/a pass 0.07 raw 0.07; gaps defined 67/67 | slope 0.87 (R2 0.89); pass-through 1.000; self z naive/pass/fit -5.6/-5.5/-0.6; spill 0.20/0.19/0.16; off/diag naive n/a pass n/a raw 0.06; gaps defined 67/67 | n/a |
| c0.0078125, readout 4 | slope 0.16 (R2 0.16); pass-through 0.007; self z naive/pass/fit -0.2/1.1/0.9; spill 0.00/0.00/0.00; off/diag naive n/a pass 0.07 raw 0.07; gaps defined 66/67 | slope 1.87 (R2 0.69); pass-through 1.000; self z naive/pass/fit 7.4/7.3/-0.8; spill 0.24/0.25/0.21; off/diag naive 0.12 pass 0.12 raw 0.08; gaps defined 67/67 | n/a |
| c0.015625, readout 3 | slope 0.64 (R2 0.49); pass-through 0.087; self z naive/pass/fit -0.2/1.4/0.6; spill 0.00/0.00/0.00; off/diag naive n/a pass 0.07 raw 0.07; gaps defined 67/67 | slope 0.87 (R2 0.90); pass-through 1.000; self z naive/pass/fit -5.4/-5.5/-0.4; spill 0.18/0.18/0.15; off/diag naive n/a pass n/a raw 0.06; gaps defined 67/67 | n/a |
| c0.015625, readout 4 | slope 0.16 (R2 0.16); pass-through 0.007; self z naive/pass/fit -0.4/1.3/0.9; spill 0.00/0.00/0.00; off/diag naive n/a pass 0.07 raw 0.07; gaps defined 66/67 | slope 1.85 (R2 0.69); pass-through 1.000; self z naive/pass/fit 7.2/7.2/-0.6; spill 0.24/0.24/0.21; off/diag naive 0.12 pass 0.12 raw 0.08; gaps defined 67/67 | n/a |
| c0.03125, readout 3 | slope 0.60 (R2 0.50); pass-through 0.087; self z naive/pass/fit -0.3/1.6/0.7; spill 0.00/0.00/0.00; off/diag naive n/a pass 0.07 raw 0.07; gaps defined 67/67 | slope 0.86 (R2 0.89); pass-through 1.000; self z naive/pass/fit -5.4/-5.3/-0.2; spill 0.18/0.18/0.16; off/diag naive n/a pass n/a raw 0.06; gaps defined 67/67 | n/a |
| c0.03125, readout 4 | slope 0.15 (R2 0.16); pass-through 0.007; self z naive/pass/fit -0.6/1.6/0.9; spill 0.17/0.18/0.17; off/diag naive n/a pass 0.08 raw 0.08; gaps defined 66/67 | slope 1.86 (R2 0.69); pass-through 1.000; self z naive/pass/fit 6.8/6.8/-0.5; spill 0.24/0.24/0.21; off/diag naive 0.12 pass 0.12 raw 0.07; gaps defined 67/67 | n/a |
| c0.0625, readout 3 | slope 0.58 (R2 0.50); pass-through 0.087; self z naive/pass/fit 0.1/2.0/0.9; spill 0.05/0.05/0.06; off/diag naive 5.70 pass 0.06 raw 0.06; gaps defined 67/67 | slope 0.85 (R2 0.89); pass-through 1.000; self z naive/pass/fit -4.9/-5.0/0.0; spill 0.18/0.18/0.15; off/diag naive n/a pass n/a raw 0.06; gaps defined 67/67 | n/a |
| c0.0625, readout 4 | slope 0.20 (R2 0.14); pass-through 0.007; self z naive/pass/fit 0.3/2.4/1.6; spill 0.46/0.46/0.44; off/diag naive n/a pass 0.08 raw 0.08; gaps defined 66/67 | slope 1.87 (R2 0.71); pass-through 1.000; self z naive/pass/fit 6.7/6.7/-0.3; spill 0.24/0.24/0.20; off/diag naive 0.11 pass 0.11 raw 0.07; gaps defined 67/67 | n/a |
| base, readout 3 | n/a | n/a | slope 0.35 (R2 1.00); pass-through 1.000; self z naive/pass/fit 0.9/0.9/1.4; spill 0.01/0.01/0.07; off/diag naive -0.32 pass -0.32 raw -0.03; gaps defined 214/224 |
| base, readout 4 | n/a | n/a | slope 0.91 (R2 0.70); pass-through 1.000; self z naive/pass/fit 2.6/2.6/1.9; spill 0.02/0.02/0.01; off/diag naive 0.05 pass 0.05 raw 0.04; gaps defined 211/224 |
