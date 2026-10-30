# Positive control (D-062 item 5), gemma-4-31b-it

| task | baseline (same 100 items) | neutral prompt | licensing prompt | instruction prompt |
| --- | --- | --- | --- | --- |
| RH | 0.61 [0.51, 0.70] (n 100) | 0.66 [0.57, 0.75] (n 100) | 0.32 [0.23, 0.41] (n 100) | 0.36 [0.27, 0.46] (n 100) |
| FT | 1.00 [1.00, 1.00] (n 100) | 1.00 [1.00, 1.00] (n 100) | 0.98 [0.95, 1.00] (n 100) | 0.57 [0.47, 0.66] (n 100) |
| CF | 0.27 [0.19, 0.36] (n 97) | 0.17 [0.10, 0.24] (n 99) | 0.01 [0.00, 0.03] (n 99) | 0.02 [0.00, 0.05] (n 100) |

Leave-one-cell-out score on the twelve act and self-describe cells (mean scaled, dm / probe), and the K1/K2 act and self-describe cells held out against the twelve (controls, mean scaled dm):

| layer | baseline | lic | neu | ins | controls: baseline | lic | neu | ins |
| --- | --- | --- | --- | ------ | --- | --- | --- |
| 16 | 0.49 / 0.52 | 0.49 / 0.52 | 0.50 / 0.52 | 0.50 / 0.51 | 0.00 (n 3) | 0.06 (n 3) | 0.05 (n 3) | 0.05 (n 3) |
| 30 | 0.77 / 0.79 | 0.77 / 0.80 | 0.77 / 0.78 | 0.74 / 0.80 | 0.47 (n 3) | 0.44 (n 3) | 0.54 (n 3) | 0.40 (n 3) |
| 39 | 0.64 / 0.76 | 0.78 / 0.78 | 0.62 / 0.74 | 0.70 / 0.76 | 0.36 (n 3) | 0.29 (n 3) | 0.31 (n 3) | 0.18 (n 3) |
| 51 | 0.53 / 0.72 | 0.71 / 0.77 | 0.52 / 0.72 | 0.62 / 0.74 | 0.49 (n 3) | 0.37 (n 3) | 0.37 (n 3) | 0.25 (n 3) |

Projection shift, licensing minus neutral, SD units, mean over the group's directions:

RH (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | -0.00 | +0.00 | +0.00 | +0.00 | +0.01 | +0.00 | +0.00 | -0.00 |
| L30|reply32 | -0.33 | -0.64 | +0.12 | -0.31 | -0.20 | +0.25 | -0.07 | +0.10 |
| L51|boundary | -0.00 | -0.00 | +0.00 | -0.00 | -0.00 | -0.01 | -0.00 | -0.00 |
| L51|reply32 | -0.52 | -0.39 | +0.16 | -0.29 | -0.31 | +0.86 | -0.02 | +0.01 |

FT (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | -0.00 | +0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | -0.00 |
| L30|reply32 | -0.03 | +0.09 | -0.16 | -0.15 | -0.16 | -0.01 | +0.02 | +0.01 |
| L51|boundary | +0.00 | +0.00 | -0.00 | +0.00 | -0.00 | -0.00 | +0.01 | +0.00 |
| L51|reply32 | -0.13 | +0.39 | -0.15 | -0.03 | -0.14 | +0.05 | -0.03 | -0.02 |

CF (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | +0.00 | -0.00 | +0.00 | -0.00 | -0.00 | +0.00 | -0.00 | +0.00 |
| L30|reply32 | +0.14 | -0.45 | +0.03 | -0.08 | -0.01 | -0.01 | +0.19 | +0.03 |
| L51|boundary | +0.01 | +0.00 | +0.00 | +0.01 | +0.01 | -0.01 | +0.00 | +0.00 |
| L51|reply32 | +0.05 | -0.49 | +0.10 | -0.11 | +0.11 | +0.11 | -0.18 | +0.01 |

Projection shift, instruction minus neutral, SD units, mean over the group's directions:

RH (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 | -0.00 |
| L30|reply32 | -0.01 | -0.16 | +0.04 | -0.06 | +0.11 | +0.45 | -0.17 | +0.04 |
| L51|boundary | -0.00 | -0.00 | +0.00 | -0.00 | -0.01 | -0.00 | -0.00 | +0.00 |
| L51|reply32 | +0.38 | -0.77 | +0.16 | -0.01 | -0.12 | -0.38 | +0.88 | +0.03 |

FT (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 |
| L30|reply32 | -0.40 | -0.72 | +0.00 | -0.54 | -0.04 | +0.16 | -0.49 | +0.10 |
| L51|boundary | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.01 | +0.00 | +0.00 |
| L51|reply32 | -0.77 | -0.97 | -0.09 | -0.94 | -0.78 | -0.48 | -0.68 | -0.11 |

CF (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L30|boundary | +0.00 | -0.01 | +0.00 | -0.00 | -0.00 | +0.00 | +0.00 | -0.00 |
| L30|reply32 | +0.35 | +0.09 | +0.28 | +0.54 | +0.77 | -0.21 | +0.32 | -0.00 |
| L51|boundary | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | -0.00 | -0.00 |
| L51|reply32 | -0.23 | +0.21 | +0.11 | +0.11 | +0.28 | -0.12 | -0.23 | -0.01 |

