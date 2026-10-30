# On-policy prediction within item strata (D-062 item 4), gemma-4-31b-it, cond baseline

Same readout as auroc.md, computed within each item stratum (AR: one-turn / recipient-state contract family; other tasks: the item factor) and averaged over strata with weights = positives; strata need 5 items of each class.

## AR write, outcome dishonest: strata recipient-state (n 120, positives 17); skipped one-turn (n 80, positives 80)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.73 | 0.66 | 0.74 | 0.63 | 0.61 | 0.60 | 0.58 | 0.62 | 0.67 / 0.90 | 1.00 |
| L30|reply32 | 0.69 | 0.68 | 0.78 | 0.65 | 0.63 | 0.65 | 0.68 | 0.69 | 0.68 / 0.93 | 0.99 |
| L51|boundary | 0.66 | 0.71 | 0.65 | 0.68 | 0.55 | 0.67 | 0.71 | 0.63 | 0.69 / 0.83 | 1.00 |
| L51|reply32 | 0.65 | 0.67 | 0.68 | 0.72 | 0.61 | 0.65 | 0.70 | 0.82 | 0.69 / 0.92 | 0.98 |

stratum recipient-state

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.73 | 0.66 | 0.74 | 0.63 | 0.61 | 0.60 | 0.58 | 0.62 | 0.67 / 0.90 | 1.00 |
| L30|reply32 | 0.69 | 0.68 | 0.78 | 0.65 | 0.63 | 0.65 | 0.68 | 0.69 | 0.68 / 0.93 | 0.99 |
| L51|boundary | 0.66 | 0.71 | 0.65 | 0.68 | 0.55 | 0.67 | 0.71 | 0.63 | 0.69 / 0.83 | 1.00 |
| L51|reply32 | 0.65 | 0.67 | 0.68 | 0.72 | 0.61 | 0.65 | 0.70 | 0.82 | 0.69 / 0.92 | 0.98 |

## DC claim, outcome dishonest: strata conflicting (n 100, positives 30); skipped one-off (n 100, positives 1)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.76 | 0.68 | 0.70 | 0.72 | 0.81 | 0.83 | 0.86 | 0.50 | 0.70 / 0.86 | 1.00 |
| L30|reply32 | 0.93 | 0.95 | 0.88 | 0.87 | 0.81 | 0.91 | 0.92 | 0.56 | 0.75 / 0.94 | 1.00 |
| L51|boundary | 0.95 | 0.81 | 0.92 | 0.93 | 0.95 | 0.96 | 0.91 | 0.97 | 0.88 / 1.00 | 1.00 |
| L51|reply32 | 0.78 | 0.83 | 0.83 | 0.82 | 0.73 | 0.56 | 0.54 | 0.74 | 0.71 / 0.90 | 1.00 |

stratum conflicting

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.76 | 0.68 | 0.70 | 0.72 | 0.81 | 0.83 | 0.86 | 0.50 | 0.70 / 0.86 | 1.00 |
| L30|reply32 | 0.93 | 0.95 | 0.88 | 0.87 | 0.81 | 0.91 | 0.92 | 0.56 | 0.75 / 0.94 | 1.00 |
| L51|boundary | 0.95 | 0.81 | 0.92 | 0.93 | 0.95 | 0.96 | 0.91 | 0.97 | 0.88 / 1.00 | 1.00 |
| L51|reply32 | 0.78 | 0.83 | 0.83 | 0.82 | 0.73 | 0.56 | 0.54 | 0.74 | 0.71 / 0.90 | 1.00 |

## DC code, outcome dishonest: strata conflicting (n 100, positives 30); skipped one-off (n 100, positives 1)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.77 | 0.69 | 0.79 | 0.82 | 0.83 | 0.95 | 0.97 | 0.64 | 0.76 / 0.89 | 0.98 |
| L30|reply32 | 0.77 | 0.77 | 0.76 | 0.69 | 0.76 | 0.83 | 0.79 | 0.80 | 0.71 / 0.93 | 0.97 |
| L51|boundary | 0.67 | 0.82 | 0.85 | 0.88 | 0.91 | 0.86 | 0.81 | 0.95 | 0.74 / 0.97 | 0.97 |
| L51|reply32 | 0.68 | 0.69 | 0.69 | 0.63 | 0.68 | 0.69 | 0.70 | 0.76 | 0.64 / 0.80 | 0.98 |

stratum conflicting

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.77 | 0.69 | 0.79 | 0.82 | 0.83 | 0.95 | 0.97 | 0.64 | 0.76 / 0.89 | 0.98 |
| L30|reply32 | 0.77 | 0.77 | 0.76 | 0.69 | 0.76 | 0.83 | 0.79 | 0.80 | 0.71 / 0.93 | 0.97 |
| L51|boundary | 0.67 | 0.82 | 0.85 | 0.88 | 0.91 | 0.86 | 0.81 | 0.95 | 0.74 / 0.97 | 0.97 |
| L51|reply32 | 0.68 | 0.69 | 0.69 | 0.63 | 0.68 | 0.69 | 0.70 | 0.76 | 0.64 / 0.80 | 0.98 |

## RH reply, outcome dishonest: strata conflicting (n 100, positives 34), one-off (n 100, positives 48)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.61 | 0.65 | 0.57 | 0.64 | 0.73 | 0.72 | 0.58 | 0.65 | 0.61 / 0.72 | 0.75 |
| L30|reply32 | 0.64 | 0.73 | 0.63 | 0.69 | 0.74 | 0.75 | 0.69 | 0.61 | 0.61 / 0.73 | 0.80 |
| L51|boundary | 0.63 | 0.60 | 0.66 | 0.67 | 0.71 | 0.68 | 0.81 | 0.67 | 0.64 / 0.77 | 0.81 |
| L51|reply32 | 0.65 | 0.59 | 0.60 | 0.64 | 0.68 | 0.74 | 0.56 | 0.64 | 0.62 / 0.72 | 0.80 |

stratum conflicting

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.64 | 0.72 | 0.60 | 0.68 | 0.76 | 0.77 | 0.63 | 0.67 | 0.64 / 0.77 | 0.79 |
| L30|reply32 | 0.64 | 0.73 | 0.61 | 0.65 | 0.76 | 0.80 | 0.58 | 0.52 | 0.61 / 0.75 | 0.84 |
| L51|boundary | 0.66 | 0.61 | 0.72 | 0.72 | 0.78 | 0.78 | 0.86 | 0.72 | 0.67 / 0.81 | 0.80 |
| L51|reply32 | 0.66 | 0.59 | 0.59 | 0.65 | 0.71 | 0.77 | 0.51 | 0.59 | 0.64 / 0.77 | 0.81 |

stratum one-off

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.59 | 0.61 | 0.55 | 0.61 | 0.71 | 0.68 | 0.55 | 0.64 | 0.58 / 0.67 | 0.73 |
| L30|reply32 | 0.64 | 0.74 | 0.65 | 0.71 | 0.72 | 0.72 | 0.77 | 0.68 | 0.60 / 0.72 | 0.77 |
| L51|boundary | 0.60 | 0.59 | 0.62 | 0.64 | 0.66 | 0.60 | 0.78 | 0.63 | 0.62 / 0.75 | 0.82 |
| L51|reply32 | 0.65 | 0.60 | 0.60 | 0.63 | 0.66 | 0.72 | 0.59 | 0.68 | 0.60 / 0.69 | 0.80 |

## FT reply, outcome not_done: strata error (n 100, positives 36), no_call (n 100, positives 9)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.60 | 0.60 | 0.62 | 0.65 | 0.63 | 0.63 | 0.56 | 0.63 | 0.61 / 0.73 | 0.80 |
| L30|reply32 | 0.80 | 0.90 | 0.77 | 0.81 | 0.85 | 0.93 | 0.80 | 0.65 | 0.72 / 0.95 | 1.00 |
| L51|boundary | 0.78 | 0.74 | 0.69 | 0.76 | 0.69 | 0.68 | 0.86 | 0.87 | 0.71 / 0.87 | 0.99 |
| L51|reply32 | 0.81 | 0.93 | 0.82 | 0.88 | 0.83 | 0.53 | 0.59 | 0.78 | 0.72 / 0.93 | 1.00 |

stratum error

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.58 | 0.59 | 0.61 | 0.64 | 0.60 | 0.59 | 0.54 | 0.60 | 0.59 / 0.71 | 0.81 |
| L30|reply32 | 0.80 | 0.91 | 0.78 | 0.83 | 0.83 | 0.94 | 0.80 | 0.67 | 0.71 / 0.95 | 1.00 |
| L51|boundary | 0.76 | 0.74 | 0.70 | 0.75 | 0.67 | 0.70 | 0.89 | 0.88 | 0.71 / 0.86 | 0.99 |
| L51|reply32 | 0.80 | 0.94 | 0.84 | 0.89 | 0.85 | 0.52 | 0.56 | 0.82 | 0.71 / 0.92 | 1.00 |

stratum no_call

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.69 | 0.62 | 0.66 | 0.68 | 0.76 | 0.75 | 0.64 | 0.76 | 0.65 / 0.78 | 0.78 |
| L30|reply32 | 0.81 | 0.89 | 0.71 | 0.76 | 0.91 | 0.88 | 0.78 | 0.57 | 0.76 / 0.94 | 1.00 |
| L51|boundary | 0.87 | 0.76 | 0.66 | 0.80 | 0.75 | 0.62 | 0.74 | 0.81 | 0.69 / 0.93 | 0.99 |
| L51|reply32 | 0.87 | 0.89 | 0.75 | 0.81 | 0.71 | 0.55 | 0.70 | 0.60 | 0.76 / 0.96 | 1.00 |

## CF reply, outcome dishonest: strata fictitious_reference (n 100, positives 51); skipped kuq_false_assumption (n 33, positives 29), kuq_future_unknown (n 31, positives 28), kuq_unsolved_problem (n 32, positives 31)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.76 | 0.76 | 0.67 | 0.76 | 0.78 | 0.82 | 0.77 | 0.75 | 0.69 / 0.83 | 0.90 |
| L30|reply32 | 0.81 | 0.82 | 0.86 | 0.89 | 0.98 | 0.99 | 0.72 | 0.82 | 0.78 / 0.98 | 1.00 |
| L51|boundary | 0.70 | 0.85 | 0.77 | 0.88 | 0.85 | 0.88 | 0.88 | 0.53 | 0.68 / 0.82 | 0.95 |
| L51|reply32 | 0.93 | 0.82 | 0.74 | 0.70 | 0.96 | 0.79 | 0.69 | 0.70 | 0.73 / 0.94 | 1.00 |

stratum fictitious_reference

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | 0.76 | 0.76 | 0.67 | 0.76 | 0.78 | 0.82 | 0.77 | 0.75 | 0.69 / 0.83 | 0.90 |
| L30|reply32 | 0.81 | 0.82 | 0.86 | 0.89 | 0.98 | 0.99 | 0.72 | 0.82 | 0.78 / 0.98 | 1.00 |
| L51|boundary | 0.70 | 0.85 | 0.77 | 0.88 | 0.85 | 0.88 | 0.88 | 0.53 | 0.68 / 0.82 | 0.95 |
| L51|reply32 | 0.93 | 0.82 | 0.74 | 0.70 | 0.96 | 0.79 | 0.69 | 0.70 | 0.73 / 0.94 | 1.00 |

