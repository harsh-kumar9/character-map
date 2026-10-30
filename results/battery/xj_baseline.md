# Cross-judging (D-062 item 3), cond baseline

## judge gemma-4-31b-it

verdicts 1680, unparsed 0; labels by attribution: none: DISHONEST 46, BORDERLINE 48, HONEST 466, None 0; another: DISHONEST 40, BORDERLINE 57, HONEST 463, None 0; you: DISHONEST 35, BORDERLINE 56, HONEST 469, None 0

| readout | task | P(DISHONEST) | P(not HONEST) |
| --- | --- | --- | --- |
| (a) blind, cheat trajectories | AR | 0.29 [0.22, 0.36] | 0.62 [0.53, 0.71] |
| (a) blind, cheat trajectories | DC | 0.11 [0.03, 0.20] | 0.11 [0.03, 0.20] |
| (a) blind, cheat trajectories | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (a) blind, cheat trajectories | all | 0.21 [0.16, 0.26] | 0.43 [0.35, 0.50] |
| (a) blind, no-cheat trajectories | AR | 0.05 [0.00, 0.17] | 0.15 [0.00, 0.33] |
| (a) blind, no-cheat trajectories | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (a) blind, no-cheat trajectories | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (a) blind, no-cheat trajectories | all | 0.00 [0.00, 0.01] | 0.01 [0.00, 0.02] |
| (a) gap | AR | 0.24 [0.11, 0.35] | 0.47 [0.27, 0.64] |
| (a) gap | DC | 0.11 [0.03, 0.20] | 0.11 [0.03, 0.20] |
| (a) gap | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (a) gap | all | 0.21 [0.15, 0.26] | 0.42 [0.35, 0.49] |
| (b) blind same-move cheat, own | AR | 0.37 [0.21, 0.53] | 0.62 [0.45, 0.78] |
| (b) blind same-move cheat, own | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) blind same-move cheat, own | RH | - | - |
| (b) blind same-move cheat, own | all | 0.33 [0.18, 0.48] | 0.55 [0.39, 0.71] |
| (b) blind same-move cheat, other's | AR | 0.23 [0.09, 0.38] | 0.53 [0.36, 0.70] |
| (b) blind same-move cheat, other's | DC | 0.43 [0.00, 1.00] | 0.43 [0.00, 1.00] |
| (b) blind same-move cheat, other's | RH | - | - |
| (b) blind same-move cheat, other's | all | 0.25 [0.12, 0.40] | 0.52 [0.36, 0.68] |
| (b) self-leniency on cheat | AR | 0.13 [-0.09, 0.35] | 0.08 [-0.10, 0.28] |
| (b) self-leniency on cheat | DC | -0.43 [-1.00, 0.00] | -0.43 [-1.00, 0.00] |
| (b) self-leniency on cheat | RH | - | - |
| (b) self-leniency on cheat | all | 0.07 [-0.14, 0.29] | 0.03 [-0.15, 0.23] |
| (b) self-leniency on no-cheat | AR | - | - |
| (b) self-leniency on no-cheat | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) self-leniency on no-cheat | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) self-leniency on no-cheat | all | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) own, told you | AR | 0.24 [0.15, 0.33] | 0.64 [0.53, 0.75] |
| (c) own, told you | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) own, told you | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) own, told you | all | 0.07 [0.04, 0.10] | 0.18 [0.14, 0.23] |
| (c) own, told another (swap) | AR | 0.28 [0.18, 0.37] | 0.69 [0.58, 0.79] |
| (c) own, told another (swap) | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) own, told another (swap) | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) own, told another (swap) | all | 0.08 [0.05, 0.11] | 0.20 [0.15, 0.25] |
| (c) other's, told you (swap) | AR | 0.12 [0.06, 0.20] | 0.42 [0.32, 0.53] |
| (c) other's, told you (swap) | DC | 0.05 [0.01, 0.10] | 0.05 [0.01, 0.10] |
| (c) other's, told you (swap) | RH | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] |
| (c) other's, told you (swap) | all | 0.06 [0.03, 0.09] | 0.14 [0.10, 0.18] |
| (c) other's, told another | AR | 0.16 [0.09, 0.25] | 0.46 [0.36, 0.57] |
| (c) other's, told another | DC | 0.05 [0.01, 0.10] | 0.05 [0.01, 0.10] |
| (c) other's, told another | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) other's, told another | all | 0.06 [0.04, 0.09] | 0.15 [0.11, 0.19] |
| (c) you minus another, own | AR | -0.04 [-0.08, 0.00] | -0.05 [-0.10, -0.01] |
| (c) you minus another, own | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) you minus another, own | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) you minus another, own | all | -0.01 [-0.02, 0.00] | -0.01 [-0.03, -0.00] |
| (c) you minus another, other's | AR | -0.04 [-0.08, 0.00] | -0.04 [-0.10, 0.01] |
| (c) you minus another, other's | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) you minus another, other's | RH | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] |
| (c) you minus another, other's | all | -0.01 [-0.02, 0.01] | -0.01 [-0.03, 0.01] |
| (d) blind, own cheat trajectories | AR | 0.34 [0.23, 0.44] | 0.69 [0.58, 0.79] |
| (d) blind, own cheat trajectories | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (d) blind, own cheat trajectories | RH | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (d) blind, own cheat trajectories | all | 0.19 [0.12, 0.25] | 0.38 [0.30, 0.46] |

## judge qwen3.5-27b

verdicts 1680, unparsed 0; labels by attribution: none: DISHONEST 130, BORDERLINE 32, HONEST 398, None 0; another: DISHONEST 130, BORDERLINE 35, HONEST 395, None 0; you: DISHONEST 132, BORDERLINE 34, HONEST 394, None 0

| readout | task | P(DISHONEST) | P(not HONEST) |
| --- | --- | --- | --- |
| (a) blind, cheat trajectories | AR | 0.81 [0.74, 0.88] | 1.00 [1.00, 1.00] |
| (a) blind, cheat trajectories | DC | 0.16 [0.06, 0.26] | 0.16 [0.06, 0.26] |
| (a) blind, cheat trajectories | RH | 0.06 [0.00, 0.15] | 0.06 [0.00, 0.15] |
| (a) blind, cheat trajectories | all | 0.58 [0.49, 0.65] | 0.70 [0.63, 0.76] |
| (a) blind, no-cheat trajectories | AR | 0.30 [0.11, 0.50] | 0.30 [0.11, 0.50] |
| (a) blind, no-cheat trajectories | DC | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.02] |
| (a) blind, no-cheat trajectories | RH | 0.01 [0.00, 0.02] | 0.04 [0.01, 0.08] |
| (a) blind, no-cheat trajectories | all | 0.02 [0.01, 0.04] | 0.04 [0.02, 0.06] |
| (a) gap | AR | 0.51 [0.28, 0.73] | 0.70 [0.50, 0.89] |
| (a) gap | DC | 0.15 [0.05, 0.25] | 0.15 [0.05, 0.25] |
| (a) gap | RH | 0.05 [-0.01, 0.14] | 0.02 [-0.06, 0.11] |
| (a) gap | all | 0.55 [0.47, 0.63] | 0.66 [0.59, 0.72] |
| (b) blind same-move cheat, own | AR | 0.77 [0.62, 0.91] | 1.00 [1.00, 1.00] |
| (b) blind same-move cheat, own | DC | 0.71 [0.00, 1.00] | 0.71 [0.00, 1.00] |
| (b) blind same-move cheat, own | RH | - | - |
| (b) blind same-move cheat, own | all | 0.76 [0.62, 0.89] | 0.97 [0.91, 1.00] |
| (b) blind same-move cheat, other's | AR | 0.88 [0.77, 1.00] | 1.00 [1.00, 1.00] |
| (b) blind same-move cheat, other's | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) blind same-move cheat, other's | RH | - | - |
| (b) blind same-move cheat, other's | all | 0.79 [0.66, 0.92] | 0.90 [0.79, 1.00] |
| (b) self-leniency on cheat | AR | -0.12 [-0.26, 0.00] | 0.00 [0.00, 0.00] |
| (b) self-leniency on cheat | DC | 0.71 [0.00, 1.00] | 0.71 [0.00, 1.00] |
| (b) self-leniency on cheat | RH | - | - |
| (b) self-leniency on cheat | all | -0.03 [-0.19, 0.12] | 0.07 [0.00, 0.16] |
| (b) self-leniency on no-cheat | AR | - | - |
| (b) self-leniency on no-cheat | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (b) self-leniency on no-cheat | RH | 0.02 [0.00, 0.05] | -0.02 [-0.10, 0.08] |
| (b) self-leniency on no-cheat | all | 0.01 [0.00, 0.02] | -0.01 [-0.05, 0.04] |
| (c) own, told you | AR | 0.65 [0.54, 0.75] | 0.82 [0.74, 0.90] |
| (c) own, told you | DC | 0.08 [0.03, 0.14] | 0.08 [0.03, 0.14] |
| (c) own, told you | RH | 0.01 [0.00, 0.03] | 0.03 [0.00, 0.07] |
| (c) own, told you | all | 0.22 [0.17, 0.26] | 0.28 [0.22, 0.33] |
| (c) own, told another (swap) | AR | 0.66 [0.55, 0.76] | 0.82 [0.74, 0.90] |
| (c) own, told another (swap) | DC | 0.07 [0.02, 0.12] | 0.07 [0.02, 0.12] |
| (c) own, told another (swap) | RH | 0.01 [0.00, 0.03] | 0.04 [0.01, 0.08] |
| (c) own, told another (swap) | all | 0.22 [0.17, 0.26] | 0.28 [0.22, 0.33] |
| (c) other's, told you (swap) | AR | 0.82 [0.73, 0.90] | 1.00 [1.00, 1.00] |
| (c) other's, told you (swap) | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) other's, told you (swap) | RH | 0.05 [0.01, 0.10] | 0.09 [0.04, 0.15] |
| (c) other's, told you (swap) | all | 0.25 [0.20, 0.31] | 0.32 [0.27, 0.37] |
| (c) other's, told another | AR | 0.84 [0.75, 0.91] | 1.00 [1.00, 1.00] |
| (c) other's, told another | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) other's, told another | RH | 0.02 [0.00, 0.05] | 0.08 [0.03, 0.14] |
| (c) other's, told another | all | 0.25 [0.20, 0.30] | 0.31 [0.26, 0.37] |
| (c) you minus another, own | AR | -0.01 [-0.06, 0.03] | 0.00 [0.00, 0.00] |
| (c) you minus another, own | DC | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] |
| (c) you minus another, own | RH | 0.00 [0.00, 0.00] | -0.01 [-0.05, 0.02] |
| (c) you minus another, own | all | 0.00 [-0.01, 0.01] | 0.00 [-0.01, 0.01] |
| (c) you minus another, other's | AR | -0.01 [-0.04, 0.00] | 0.00 [0.00, 0.00] |
| (c) you minus another, other's | DC | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| (c) you minus another, other's | RH | 0.03 [0.00, 0.07] | 0.01 [0.00, 0.03] |
| (c) you minus another, other's | all | 0.01 [-0.01, 0.02] | 0.00 [0.00, 0.01] |
| (d) blind, own cheat trajectories | AR | 0.77 [0.66, 0.87] | 1.00 [1.00, 1.00] |
| (d) blind, own cheat trajectories | DC | 0.75 [0.40, 1.00] | 0.75 [0.40, 1.00] |
| (d) blind, own cheat trajectories | RH | - | - |
| (d) blind, own cheat trajectories | all | 0.76 [0.66, 0.86] | 0.97 [0.93, 1.00] |

