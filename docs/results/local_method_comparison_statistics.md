# Local paired method-comparison statistics

| Comparison | Benchmark | Metric | Mean difference | Bootstrap 95% CI | Holm p | Rank-biserial | Better / worse / tied |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FM vs DDPM (primary) | CAMEO | BB RMSD | -0.007215 | [-0.149566, 0.130141] | 0.877531916 | +0.013916 | 86/77/0 (FM / DDPM) |
| FM vs DDPM (primary) | CAMEO | TM-score | +0.001182 | [-0.002197, 0.004750] | 0.630634811 | +0.090678 | 90/73/0 (FM / DDPM) |
| FM vs DDPM (primary) | PDB-date | BB RMSD | +0.010595 | [-0.030799, 0.053301] | 0.24682802 | +0.102677 | 241/200/1 (FM / DDPM) |
| FM vs DDPM (primary) | PDB-date | TM-score | +0.000811 | [-0.000220, 0.001837] | 0.476093223 | +0.077457 | 230/211/1 (FM / DDPM) |
| DDPM vs Bit (secondary) | CAMEO | BB RMSD | +0.096375 | [-0.028546, 0.229402] | 0.715561562 | +0.100554 | 84/79/0 (DDPM / Bit) |
| DDPM vs Bit (secondary) | CAMEO | TM-score | +0.002351 | [-0.000876, 0.005468] | 0.112039179 | +0.198414 | 94/69/0 (DDPM / Bit) |
| DDPM vs Bit (secondary) | PDB-date | BB RMSD | -0.004276 | [-0.040123, 0.031841] | 0.715561562 | -0.053837 | 211/230/1 (DDPM / Bit) |
| DDPM vs Bit (secondary) | PDB-date | TM-score | -0.000752 | [-0.002192, 0.000710] | 0.715561562 | -0.064775 | 215/226/1 (DDPM / Bit) |

FM-vs-DDPM primary tests are all non-significant after Holm correction. Positive FM advantage: BB=DDPM−FM, TM=FM−DDPM. DDPM-vs-Bit results are a separate secondary family; positive means DDPM better.

Finalized FM-vs-Bit CAMEO TM: Holm p=0.00879188717; significant positive paired rank shift after Holm correction, while the bootstrap CI for the mean improvement includes zero.
