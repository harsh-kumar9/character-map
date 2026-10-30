# On-policy prediction within item strata (D-062 item 4), qwen3.5-27b, cond baseline

Same readout as auroc.md, computed within each item stratum (AR: one-turn / recipient-state contract family; other tasks: the item factor) and averaged over strata with weights = positives; strata need 5 items of each class.

## AR write, outcome dishonest: strata one-turn (n 80, positives 60), recipient-state (n 120, positives 22)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.63 | 0.59 | 0.61 | 0.62 | 0.61 | 0.61 | 0.73 | 0.60 | 0.62 / 0.71 | 0.86 |
| L32|reply32 | 0.62 | 0.64 | 0.62 | 0.59 | 0.59 | 0.59 | 0.58 | 0.60 | 0.62 / 0.73 | 0.84 |
| L54|boundary | 0.66 | 0.73 | 0.65 | 0.67 | 0.68 | 0.76 | 0.64 | 0.63 | 0.65 / 0.75 | 0.82 |
| L54|reply32 | 0.67 | 0.75 | 0.69 | 0.73 | 0.69 | 0.76 | 0.55 | 0.64 | 0.64 / 0.80 | 0.87 |

stratum one-turn

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.62 | 0.56 | 0.62 | 0.61 | 0.61 | 0.60 | 0.79 | 0.58 | 0.59 / 0.67 | 0.84 |
| L32|reply32 | 0.61 | 0.64 | 0.61 | 0.60 | 0.56 | 0.57 | 0.54 | 0.62 | 0.62 / 0.73 | 0.83 |
| L54|boundary | 0.65 | 0.73 | 0.65 | 0.67 | 0.70 | 0.75 | 0.66 | 0.65 | 0.65 / 0.73 | 0.81 |
| L54|reply32 | 0.67 | 0.75 | 0.68 | 0.73 | 0.72 | 0.76 | 0.51 | 0.68 | 0.62 / 0.77 | 0.85 |

stratum recipient-state

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.64 | 0.65 | 0.61 | 0.66 | 0.60 | 0.66 | 0.57 | 0.66 | 0.68 / 0.82 | 0.91 |
| L32|reply32 | 0.65 | 0.64 | 0.65 | 0.57 | 0.67 | 0.64 | 0.69 | 0.54 | 0.61 / 0.72 | 0.87 |
| L54|boundary | 0.68 | 0.72 | 0.68 | 0.65 | 0.64 | 0.79 | 0.59 | 0.56 | 0.64 / 0.79 | 0.87 |
| L54|reply32 | 0.69 | 0.74 | 0.72 | 0.73 | 0.61 | 0.76 | 0.66 | 0.54 | 0.70 / 0.86 | 0.91 |

## DC claim, outcome dishonest: strata conflicting (n 100, positives 8); skipped one-off (n 100, positives 0)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.71 | 0.70 | 0.78 | 0.81 | 0.92 | 0.97 | 0.53 | 0.66 | 0.69 / 0.90 | 1.00 |
| L32|reply32 | 0.88 | 0.84 | 0.92 | 0.88 | 0.74 | 0.97 | 0.54 | 1.00 | 0.83 / 1.00 | 1.00 |
| L54|boundary | 0.90 | 0.96 | 0.95 | 0.99 | 0.77 | 0.75 | 0.96 | 0.98 | 0.88 / 0.99 | 1.00 |
| L54|reply32 | 0.93 | 0.98 | 0.86 | 0.97 | 0.74 | 0.95 | 0.85 | 0.81 | 0.90 / 0.98 | 1.00 |

stratum conflicting

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.71 | 0.70 | 0.78 | 0.81 | 0.92 | 0.97 | 0.53 | 0.66 | 0.69 / 0.90 | 1.00 |
| L32|reply32 | 0.88 | 0.84 | 0.92 | 0.88 | 0.74 | 0.97 | 0.54 | 1.00 | 0.83 / 1.00 | 1.00 |
| L54|boundary | 0.90 | 0.96 | 0.95 | 0.99 | 0.77 | 0.75 | 0.96 | 0.98 | 0.88 / 0.99 | 1.00 |
| L54|reply32 | 0.93 | 0.98 | 0.86 | 0.97 | 0.74 | 0.95 | 0.85 | 0.81 | 0.90 / 0.98 | 1.00 |

## DC code, outcome dishonest: strata conflicting (n 100, positives 8); skipped one-off (n 100, positives 0)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.82 | 0.79 | 0.83 | 0.71 | 0.77 | 0.76 | 0.79 | 0.56 | 0.76 / 0.97 | 1.00 |
| L32|reply32 | 0.81 | 0.80 | 0.80 | 0.63 | 0.57 | 0.80 | 0.58 | 0.63 | 0.76 / 0.96 | 1.00 |
| L54|boundary | 0.85 | 0.82 | 0.88 | 0.85 | 0.84 | 0.81 | 0.94 | 0.86 | 0.78 / 0.99 | 1.00 |
| L54|reply32 | 0.70 | 0.81 | 0.74 | 0.71 | 0.74 | 0.79 | 0.86 | 0.84 | 0.68 / 0.83 | 1.00 |

stratum conflicting

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.82 | 0.79 | 0.83 | 0.71 | 0.77 | 0.76 | 0.79 | 0.56 | 0.76 / 0.97 | 1.00 |
| L32|reply32 | 0.81 | 0.80 | 0.80 | 0.63 | 0.57 | 0.80 | 0.58 | 0.63 | 0.76 / 0.96 | 1.00 |
| L54|boundary | 0.85 | 0.82 | 0.88 | 0.85 | 0.84 | 0.81 | 0.94 | 0.86 | 0.78 / 0.99 | 1.00 |
| L54|reply32 | 0.70 | 0.81 | 0.74 | 0.71 | 0.74 | 0.79 | 0.86 | 0.84 | 0.68 / 0.83 | 1.00 |

## RH reply, outcome dishonest: strata one-off (n 100, positives 18); skipped conflicting (n 100, positives 0)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.62 | 0.63 | 0.66 | 0.63 | 0.60 | 0.62 | 0.63 | 0.74 | 0.63 / 0.74 | 0.81 |
| L32|reply32 | 0.69 | 0.74 | 0.62 | 0.71 | 0.63 | 0.63 | 0.61 | 0.79 | 0.62 / 0.72 | 0.84 |
| L54|boundary | 0.71 | 0.67 | 0.69 | 0.61 | 0.67 | 0.66 | 0.60 | 0.83 | 0.71 / 0.80 | 0.78 |
| L54|reply32 | 0.64 | 0.67 | 0.67 | 0.71 | 0.61 | 0.52 | 0.71 | 0.56 | 0.61 / 0.68 | 0.86 |

stratum one-off

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.62 | 0.63 | 0.66 | 0.63 | 0.60 | 0.62 | 0.63 | 0.74 | 0.63 / 0.74 | 0.81 |
| L32|reply32 | 0.69 | 0.74 | 0.62 | 0.71 | 0.63 | 0.63 | 0.61 | 0.79 | 0.62 / 0.72 | 0.84 |
| L54|boundary | 0.71 | 0.67 | 0.69 | 0.61 | 0.67 | 0.66 | 0.60 | 0.83 | 0.71 / 0.80 | 0.78 |
| L54|reply32 | 0.64 | 0.67 | 0.67 | 0.71 | 0.61 | 0.52 | 0.71 | 0.56 | 0.61 / 0.68 | 0.86 |

## FT reply, outcome not_done: strata error (n 100, positives 40), no_call (n 100, positives 15)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.63 | 0.62 | 0.60 | 0.64 | 0.65 | 0.62 | 0.54 | 0.57 | 0.61 / 0.74 | 0.85 |
| L32|reply32 | 0.79 | 0.85 | 0.77 | 0.79 | 0.84 | 0.92 | 0.62 | 0.81 | 0.69 / 0.90 | 0.99 |
| L54|boundary | 0.73 | 0.78 | 0.63 | 0.68 | 0.66 | 0.81 | 0.71 | 0.69 | 0.67 / 0.84 | 0.93 |
| L54|reply32 | 0.90 | 0.91 | 0.65 | 0.70 | 0.66 | 0.74 | 0.57 | 0.80 | 0.73 / 0.88 | 0.97 |

stratum error

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.64 | 0.62 | 0.59 | 0.64 | 0.66 | 0.61 | 0.52 | 0.50 | 0.60 / 0.74 | 0.88 |
| L32|reply32 | 0.81 | 0.84 | 0.76 | 0.77 | 0.81 | 0.90 | 0.65 | 0.77 | 0.68 / 0.89 | 0.99 |
| L54|boundary | 0.72 | 0.78 | 0.63 | 0.70 | 0.66 | 0.83 | 0.74 | 0.69 | 0.65 / 0.83 | 0.93 |
| L54|reply32 | 0.91 | 0.92 | 0.65 | 0.71 | 0.65 | 0.75 | 0.58 | 0.74 | 0.72 / 0.87 | 0.97 |

stratum no_call

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.62 | 0.64 | 0.63 | 0.66 | 0.62 | 0.63 | 0.60 | 0.75 | 0.62 / 0.75 | 0.76 |
| L32|reply32 | 0.76 | 0.88 | 0.78 | 0.85 | 0.92 | 0.98 | 0.54 | 0.93 | 0.70 / 0.94 | 1.00 |
| L54|boundary | 0.76 | 0.79 | 0.61 | 0.61 | 0.65 | 0.76 | 0.64 | 0.68 | 0.73 / 0.86 | 0.94 |
| L54|reply32 | 0.89 | 0.90 | 0.67 | 0.69 | 0.68 | 0.71 | 0.55 | 0.97 | 0.77 / 0.90 | 0.99 |

## CF reply, outcome dishonest: strata kuq_false_assumption (n 34, positives 23), kuq_future_unknown (n 32, positives 25); skipped fictitious_reference (n 100, positives 0), kuq_unsolved_problem (n 33, positives 29)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.78 | 0.79 | 0.72 | 0.76 | 0.80 | 0.83 | 0.63 | 0.56 | 0.66 / 0.84 | 0.94 |
| L32|reply32 | 0.65 | 0.64 | 0.61 | 0.65 | 0.71 | 0.66 | 0.70 | 0.54 | 0.62 / 0.75 | 0.95 |
| L54|boundary | 0.68 | 0.67 | 0.65 | 0.68 | 0.73 | 0.72 | 0.76 | 0.76 | 0.63 / 0.79 | 0.94 |
| L54|reply32 | 0.64 | 0.77 | 0.63 | 0.66 | 0.67 | 0.64 | 0.67 | 0.74 | 0.63 / 0.77 | 0.92 |

stratum kuq_false_assumption

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.72 | 0.75 | 0.66 | 0.71 | 0.82 | 0.79 | 0.68 | 0.61 | 0.63 / 0.79 | 0.96 |
| L32|reply32 | 0.56 | 0.61 | 0.61 | 0.54 | 0.55 | 0.52 | 0.70 | 0.56 | 0.60 / 0.72 | 0.93 |
| L54|boundary | 0.62 | 0.64 | 0.63 | 0.69 | 0.68 | 0.64 | 0.78 | 0.71 | 0.62 / 0.76 | 1.00 |
| L54|reply32 | 0.60 | 0.64 | 0.60 | 0.62 | 0.52 | 0.51 | 0.53 | 0.60 | 0.62 / 0.72 | 0.87 |

stratum kuq_future_unknown

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | 0.83 | 0.83 | 0.77 | 0.80 | 0.79 | 0.86 | 0.58 | 0.50 | 0.68 / 0.88 | 0.92 |
| L32|reply32 | 0.74 | 0.66 | 0.60 | 0.76 | 0.85 | 0.80 | 0.71 | 0.53 | 0.64 / 0.78 | 0.96 |
| L54|boundary | 0.73 | 0.69 | 0.67 | 0.67 | 0.77 | 0.79 | 0.74 | 0.81 | 0.64 / 0.82 | 0.88 |
| L54|reply32 | 0.67 | 0.89 | 0.66 | 0.70 | 0.81 | 0.75 | 0.79 | 0.86 | 0.63 / 0.81 | 0.96 |

