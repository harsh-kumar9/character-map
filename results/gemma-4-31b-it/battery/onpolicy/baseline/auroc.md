# On-policy prediction (D-062 item 4), gemma-4-31b-it, cond baseline

Folded AUROC (max(a, 1 - a)) of the standardized projection for the dishonest outcome, mean over the group's directions; signed values per direction in auroc.json. Probe: within-task 5-fold logistic ceiling on the full activation.

## AR write, outcome dishonest: n 200, positives 97

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.76 | 0.86 | 0.74 | 0.80 | 0.88 | 0.88 | 0.52 | 0.68 | 0.75 / 0.96 | 1.00 |
| L30|reply32 | 0.84 | 0.90 | 0.74 | 0.87 | 0.92 | 0.89 | 0.53 | 0.60 | 0.73 / 0.89 | 1.00 |
| L51|boundary | 0.69 | 0.67 | 0.70 | 0.75 | 0.75 | 0.82 | 0.75 | 0.65 | 0.74 / 0.90 | 1.00 |
| L51|reply32 | 0.69 | 0.82 | 0.72 | 0.76 | 0.88 | 0.83 | 0.85 | 0.73 | 0.72 / 0.94 | 1.00 |

## DC claim, outcome dishonest: n 200, positives 31

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.62 | 0.56 | 0.61 | 0.56 | 0.57 | 0.59 | 0.60 | 0.52 | 0.61 / 0.80 | 0.97 |
| L30|reply32 | 0.70 | 0.68 | 0.71 | 0.69 | 0.61 | 0.65 | 0.58 | 0.70 | 0.62 / 0.83 | 0.99 |
| L51|boundary | 0.55 | 0.76 | 0.72 | 0.78 | 0.72 | 0.77 | 0.79 | 0.81 | 0.71 / 0.89 | 0.98 |
| L51|reply32 | 0.65 | 0.66 | 0.69 | 0.63 | 0.67 | 0.57 | 0.54 | 0.62 | 0.64 / 0.75 | 1.00 |

## DC code, outcome dishonest: n 200, positives 31

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.62 | 0.61 | 0.67 | 0.68 | 0.72 | 0.86 | 0.70 | 0.51 | 0.68 / 0.84 | 0.93 |
| L30|reply32 | 0.62 | 0.68 | 0.72 | 0.67 | 0.64 | 0.78 | 0.60 | 0.76 | 0.59 / 0.72 | 0.93 |
| L51|boundary | 0.59 | 0.70 | 0.72 | 0.71 | 0.69 | 0.65 | 0.56 | 0.74 | 0.64 / 0.88 | 0.95 |
| L51|reply32 | 0.59 | 0.61 | 0.62 | 0.54 | 0.60 | 0.51 | 0.62 | 0.59 | 0.60 / 0.70 | 0.93 |

## RH reply, outcome dishonest: n 200, positives 82

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.61 | 0.63 | 0.56 | 0.61 | 0.69 | 0.68 | 0.58 | 0.64 | 0.59 / 0.67 | 0.74 |
| L30|reply32 | 0.61 | 0.70 | 0.58 | 0.66 | 0.69 | 0.71 | 0.64 | 0.59 | 0.57 / 0.67 | 0.80 |
| L51|boundary | 0.57 | 0.55 | 0.66 | 0.63 | 0.71 | 0.68 | 0.76 | 0.65 | 0.59 / 0.69 | 0.80 |
| L51|reply32 | 0.63 | 0.59 | 0.58 | 0.61 | 0.67 | 0.71 | 0.55 | 0.61 | 0.59 / 0.68 | 0.79 |

## FT reply, outcome not_done: n 200, positives 45

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.65 | 0.61 | 0.63 | 0.58 | 0.62 | 0.61 | 0.65 | 0.59 | 0.65 / 0.73 | 0.91 |
| L30|reply32 | 0.82 | 0.90 | 0.74 | 0.80 | 0.84 | 0.93 | 0.82 | 0.64 | 0.72 / 0.90 | 1.00 |
| L51|boundary | 0.81 | 0.77 | 0.65 | 0.75 | 0.74 | 0.56 | 0.58 | 0.53 | 0.68 / 0.82 | 0.99 |
| L51|reply32 | 0.85 | 0.92 | 0.82 | 0.87 | 0.83 | 0.57 | 0.57 | 0.69 | 0.72 / 0.92 | 1.00 |

## CF reply, outcome dishonest: n 196, positives 139

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.65 | 0.60 | 0.68 | 0.59 | 0.65 | 0.51 | 0.56 | 0.52 | 0.72 / 0.86 | 0.87 |
| L30|reply32 | 0.74 | 0.73 | 0.73 | 0.70 | 0.85 | 0.81 | 0.71 | 0.82 | 0.69 / 0.90 | 0.99 |
| L51|boundary | 0.66 | 0.78 | 0.71 | 0.79 | 0.87 | 0.89 | 0.73 | 0.54 | 0.72 / 0.86 | 0.94 |
| L51|reply32 | 0.69 | 0.67 | 0.61 | 0.59 | 0.82 | 0.67 | 0.68 | 0.55 | 0.63 / 0.83 | 0.99 |

