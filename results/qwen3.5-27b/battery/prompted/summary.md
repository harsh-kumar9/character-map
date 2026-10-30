# Positive control (D-062 item 5), qwen3.5-27b

| task | baseline (same 100 items) | neutral prompt | licensing prompt | instruction prompt |
| --- | --- | --- | --- | --- |
| RH | 0.89 [0.82, 0.95] (n 100) | 0.92 [0.86, 0.97] (n 100) | 0.87 [0.80, 0.93] (n 100) | 0.87 [0.80, 0.93] (n 100) |
| FT | 1.00 [1.00, 1.00] (n 100) | 1.00 [1.00, 1.00] (n 100) | 1.00 [1.00, 1.00] (n 100) | 0.91 [0.85, 0.96] (n 100) |
| CF | 0.60 [0.50, 0.69] (n 100) | 0.60 [0.51, 0.69] (n 98) | 0.58 [0.49, 0.68] (n 96) | 0.55 [0.45, 0.65] (n 100) |

Leave-one-cell-out score on the twelve act and self-describe cells (mean scaled, dm / probe), and the K1/K2 act and self-describe cells held out against the twelve (controls, mean scaled dm):

| layer | baseline | lic | neu | ins | controls: baseline | lic | neu | ins |
| --- | --- | --- | --- | ------ | --- | --- | --- |
| 17 | 0.71 / 0.75 | 0.72 / 0.75 | 0.71 / 0.75 | 0.72 / 0.75 | 0.50 (n 3) | 0.51 (n 3) | 0.51 (n 3) | 0.51 (n 3) |
| 32 | 0.77 / 0.84 | 0.79 / 0.85 | 0.79 / 0.85 | 0.78 / 0.86 | 0.26 (n 3) | 0.30 (n 3) | 0.38 (n 3) | 0.29 (n 3) |
| 42 | 0.76 / 0.78 | 0.75 / 0.80 | 0.77 / 0.79 | 0.74 / 0.80 | 0.20 (n 3) | 0.27 (n 3) | 0.28 (n 3) | 0.26 (n 3) |
| 54 | 0.78 / 0.78 | 0.73 / 0.76 | 0.76 / 0.78 | 0.72 / 0.76 | 0.20 (n 3) | 0.38 (n 3) | 0.31 (n 3) | 0.39 (n 3) |

Projection shift, licensing minus neutral, SD units, mean over the group's directions:

RH (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | +0.00 | +0.00 | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 |
| L32|reply32 | -0.13 | -0.29 | +0.07 | -0.11 | -0.19 | -0.13 | -0.12 | -0.01 |
| L54|boundary | -0.00 | -0.00 | -0.00 | -0.01 | -0.01 | -0.00 | +0.00 | -0.00 |
| L54|reply32 | +0.01 | -0.02 | +0.02 | +0.02 | +0.03 | -0.02 | -0.04 | -0.01 |

FT (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 |
| L32|reply32 | -0.02 | +0.01 | -0.09 | -0.10 | -0.12 | -0.07 | -0.01 | -0.02 |
| L54|boundary | -0.00 | +0.00 | +0.00 | +0.00 | -0.00 | -0.00 | +0.00 | -0.00 |
| L54|reply32 | +0.03 | +0.10 | +0.01 | +0.06 | +0.06 | -0.01 | +0.02 | -0.00 |

CF (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | +0.01 | +0.01 | -0.00 | +0.00 | +0.00 | -0.01 | +0.01 | +0.00 |
| L32|reply32 | -0.18 | -0.34 | +0.26 | +0.04 | +0.07 | +0.13 | -0.03 | -0.00 |
| L54|boundary | +0.00 | +0.00 | -0.00 | +0.00 | -0.00 | +0.00 | -0.00 | +0.00 |
| L54|reply32 | +0.08 | -0.12 | +0.03 | +0.03 | +0.11 | -0.09 | +0.06 | -0.02 |

Projection shift, instruction minus neutral, SD units, mean over the group's directions:

RH (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | -0.00 | +0.00 | -0.00 | +0.00 | +0.00 | +0.01 | +0.00 | -0.00 |
| L32|reply32 | -0.13 | -0.31 | +0.07 | -0.12 | -0.21 | -0.10 | -0.08 | +0.00 |
| L54|boundary | -0.00 | -0.00 | -0.00 | -0.01 | -0.01 | -0.01 | +0.00 | +0.00 |
| L54|reply32 | +0.03 | -0.02 | +0.02 | +0.03 | +0.03 | -0.01 | -0.05 | +0.00 |

FT (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | -0.00 |
| L32|reply32 | -0.04 | -0.20 | +0.08 | -0.02 | +0.09 | -0.18 | +0.07 | +0.00 |
| L54|boundary | -0.00 | -0.00 | +0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 |
| L54|reply32 | -0.16 | -0.22 | +0.03 | -0.15 | -0.05 | -0.19 | +0.17 | +0.02 |

CF (n 100)

| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | K1 | K2 | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L32|boundary | +0.01 | +0.02 | +0.00 | +0.01 | +0.02 | -0.00 | +0.00 | +0.00 |
| L32|reply32 | -0.04 | -0.24 | +0.24 | +0.14 | +0.25 | +0.04 | +0.06 | -0.01 |
| L54|boundary | +0.00 | -0.00 | +0.00 | +0.00 | +0.00 | -0.01 | -0.00 | +0.00 |
| L54|reply32 | +0.19 | -0.11 | +0.07 | +0.13 | +0.30 | -0.05 | +0.04 | -0.03 |

