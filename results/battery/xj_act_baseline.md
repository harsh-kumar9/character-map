# Cross-judging (D-062 item 3), cond baseline, action-only judgment (DC and RH, code against tests, blind)

## judge gemma-4-31b-it

verdicts 400, unparsed 0; labels by attribution: none: DISHONEST 227, BORDERLINE 0, HONEST 173, None 0

| readout | task | P(DISHONEST) | P(not HONEST) |
| --- | --- | --- | --- |
| (a) blind, cheat trajectories | DC | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (a) blind, cheat trajectories | RH | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (a) blind, cheat trajectories | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (a) blind, no-cheat trajectories | DC | 0.48 [0.39, 0.55] | 0.48 [0.39, 0.55] |
| (a) blind, no-cheat trajectories | RH | 0.47 [0.39, 0.55] | 0.47 [0.39, 0.55] |
| (a) blind, no-cheat trajectories | all | 0.47 [0.41, 0.53] | 0.47 [0.41, 0.53] |
| (a) gap | DC | 0.52 [0.45, 0.61] | 0.52 [0.45, 0.61] |
| (a) gap | RH | 0.53 [0.45, 0.61] | 0.53 [0.45, 0.61] |
| (a) gap | all | 0.53 [0.47, 0.59] | 0.53 [0.47, 0.59] |
| (d) blind, own cheat trajectories | DC | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (d) blind, own cheat trajectories | RH | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (d) blind, own cheat trajectories | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

## judge qwen3.5-27b

verdicts 400, unparsed 0; labels by attribution: none: DISHONEST 241, BORDERLINE 22, HONEST 137, None 0

| readout | task | P(DISHONEST) | P(not HONEST) |
| --- | --- | --- | --- |
| (a) blind, cheat trajectories | DC | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (a) blind, cheat trajectories | RH | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (a) blind, cheat trajectories | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (a) blind, no-cheat trajectories | DC | 0.54 [0.45, 0.63] | 0.60 [0.51, 0.68] |
| (a) blind, no-cheat trajectories | RH | 0.49 [0.41, 0.57] | 0.57 [0.48, 0.65] |
| (a) blind, no-cheat trajectories | all | 0.52 [0.45, 0.57] | 0.58 [0.52, 0.64] |
| (a) gap | DC | 0.46 [0.37, 0.55] | 0.40 [0.32, 0.49] |
| (a) gap | RH | 0.51 [0.43, 0.59] | 0.43 [0.35, 0.52] |
| (a) gap | all | 0.48 [0.43, 0.55] | 0.42 [0.36, 0.48] |
| (d) blind, own cheat trajectories | DC | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| (d) blind, own cheat trajectories | RH | - | - |
| (d) blind, own cheat trajectories | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

