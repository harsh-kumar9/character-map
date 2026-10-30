# On-policy prediction (D-062 item 4), qwen3.5-27b, cond baseline

Folded AUROC (max(a, 1 - a)) of the standardized projection for the dishonest outcome, mean over the group's directions; signed values per direction in auroc.json. Probe: within-task 5-fold logistic ceiling on the full activation.

## AR write, outcome dishonest: n 200, positives 82

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.70 | 0.66 | 0.69 | 0.60 | 0.53 | 0.57 | 0.59 | 0.78 | 0.64 / 0.75 | 0.94 |
| L32|reply32 | 0.71 | 0.72 | 0.70 | 0.73 | 0.71 | 0.74 | 0.77 | 0.52 | 0.67 / 0.75 | 0.94 |
| L54|boundary | 0.61 | 0.58 | 0.65 | 0.64 | 0.68 | 0.70 | 0.56 | 0.51 | 0.64 / 0.74 | 0.95 |
| L54|reply32 | 0.63 | 0.74 | 0.70 | 0.72 | 0.82 | 0.88 | 0.73 | 0.62 | 0.68 / 0.84 | 0.95 |

## DC claim, outcome dishonest: n 200, positives 8

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.64 | 0.65 | 0.73 | 0.74 | 0.82 | 0.87 | 0.61 | 0.67 | 0.67 / 0.80 | 0.97 |
| L32|reply32 | 0.78 | 0.77 | 0.83 | 0.70 | 0.64 | 0.77 | 0.54 | 0.93 | 0.73 / 0.86 | 0.99 |
| L54|boundary | 0.76 | 0.81 | 0.82 | 0.84 | 0.72 | 0.73 | 0.76 | 0.90 | 0.76 / 0.84 | 1.00 |
| L54|reply32 | 0.80 | 0.90 | 0.73 | 0.84 | 0.68 | 0.76 | 0.66 | 0.74 | 0.76 / 0.89 | 1.00 |

## DC code, outcome dishonest: n 200, positives 8

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.75 | 0.73 | 0.72 | 0.65 | 0.67 | 0.65 | 0.73 | 0.56 | 0.72 / 0.88 | 0.95 |
| L32|reply32 | 0.70 | 0.72 | 0.70 | 0.69 | 0.59 | 0.72 | 0.55 | 0.70 | 0.70 / 0.87 | 0.99 |
| L54|boundary | 0.72 | 0.79 | 0.81 | 0.72 | 0.80 | 0.85 | 0.94 | 0.81 | 0.69 / 0.87 | 0.98 |
| L54|reply32 | 0.61 | 0.74 | 0.69 | 0.69 | 0.77 | 0.86 | 0.66 | 0.81 | 0.65 / 0.79 | 0.98 |

## RH reply, outcome dishonest: n 200, positives 18

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.62 | 0.58 | 0.66 | 0.59 | 0.58 | 0.53 | 0.65 | 0.70 | 0.61 / 0.79 | 0.85 |
| L32|reply32 | 0.67 | 0.71 | 0.60 | 0.67 | 0.60 | 0.64 | 0.62 | 0.77 | 0.61 / 0.71 | 0.85 |
| L54|boundary | 0.56 | 0.68 | 0.67 | 0.63 | 0.74 | 0.59 | 0.79 | 0.80 | 0.62 / 0.78 | 0.89 |
| L54|reply32 | 0.60 | 0.66 | 0.63 | 0.68 | 0.62 | 0.55 | 0.73 | 0.54 | 0.59 / 0.65 | 0.85 |

## FT reply, outcome not_done: n 200, positives 55

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.66 | 0.65 | 0.64 | 0.66 | 0.68 | 0.65 | 0.57 | 0.56 | 0.60 / 0.69 | 0.87 |
| L32|reply32 | 0.75 | 0.85 | 0.75 | 0.77 | 0.85 | 0.94 | 0.62 | 0.79 | 0.68 / 0.84 | 0.99 |
| L54|boundary | 0.70 | 0.80 | 0.65 | 0.69 | 0.68 | 0.77 | 0.61 | 0.55 | 0.63 / 0.81 | 0.95 |
| L54|reply32 | 0.90 | 0.90 | 0.64 | 0.70 | 0.66 | 0.72 | 0.60 | 0.79 | 0.72 / 0.87 | 0.99 |

## CF reply, outcome dishonest: n 199, positives 77

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.90 | 0.82 | 0.82 | 0.76 | 0.81 | 0.84 | 0.85 | 0.77 | 0.81 / 0.95 | 0.98 |
| L32|reply32 | 0.79 | 0.82 | 0.77 | 0.87 | 0.84 | 0.78 | 0.88 | 0.88 | 0.70 / 0.86 | 0.99 |
| L54|boundary | 0.72 | 0.70 | 0.77 | 0.79 | 0.60 | 0.51 | 0.85 | 0.85 | 0.65 / 0.78 | 0.99 |
| L54|reply32 | 0.64 | 0.63 | 0.64 | 0.61 | 0.65 | 0.62 | 0.66 | 0.55 | 0.68 / 0.81 | 0.99 |

