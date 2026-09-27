# 4 Results

### 4.1 Local Matched Comparison

We evaluated three methods in the same local comparison: Local Bit, Local Bit with Latent Residual Flow Matching (FM), and Local Bit with Matched Residual DDPM. The local evaluation comprised 163 CAMEO targets and 442 PDB-date targets. Table 3 reports mean backbone (BB) RMSD, for which lower values indicate better structural agreement, and mean TM-score, for which higher values indicate better agreement. All three method rows refer to the local matched setting; the paper-reported results are presented separately in Table 4.

On CAMEO, Local Bit yielded a mean BB RMSD of 6.2889 Å and a mean TM-score of 0.84142. Latent Residual FM yielded 6.1997 Å and 0.84495, respectively. Matched Residual DDPM yielded 6.1925 Å and 0.84377. Thus, both residual refinements improved the CAMEO mean BB RMSD and TM-score relative to Local Bit. The Bit-relative mean changes were +0.0891596035 Å in BB RMSD and +0.003533130792 in TM-score for FM, compared with +0.0963748763 Å and +0.002351455122 for DDPM. Here and below, positive BB improvement denotes Bit RMSD minus refined RMSD, whereas positive TM improvement denotes refined TM-score minus Bit TM-score. The numerically lowest CAMEO mean BB RMSD belonged to DDPM, while the numerically highest mean TM-score belonged to FM; these descriptive rankings do not by themselves establish a statistically reliable difference between the refinements.

On PDB-date, Local Bit yielded a mean BB RMSD of 3.1138 Å and a mean TM-score of 0.91380. FM yielded 3.1074 Å and 0.91386, whereas DDPM yielded 3.1180 Å and 0.91304. The corresponding Bit-relative mean changes were +0.0063187644 Å and +0.000059240022 for FM, and −0.0042764425 Å and −0.000752195122 for DDPM. Thus, the PDB-date FM means changed only slightly relative to Bit, while neither PDB-date mean metric improved under the matched DDPM refinement. These values summarize observed performance on the evaluated targets rather than the consistency or magnitude of improvement for every individual protein.

### 4.2 Paired Statistical Comparison

The primary paired statistical analysis directly compared FM with Matched Residual DDPM on BB RMSD and TM-score in each benchmark, producing four tests. Each difference was computed on matched targets. For BB RMSD, the FM advantage was defined as DDPM minus FM; for TM-score, it was FM minus DDPM. Positive values therefore favor FM for either metric. The analysis reports a target-paired bootstrap percentile 95% confidence interval (CI) for each mean difference and a Holm-adjusted two-sided Wilcoxon signed-rank p-value across the four primary tests. The paired tests address a different question from the method means in Table 3: they assess the distribution of within-target differences between FM and DDPM.

For CAMEO BB RMSD, the mean FM advantage was −0.007215 Å (95% bootstrap CI [−0.149566, 0.130141]; Holm-adjusted p = 0.8775). The CAMEO TM-score advantage was +0.001182 (95% CI [−0.002197, 0.004750]; Holm-adjusted p = 0.6306). For PDB-date, the BB RMSD advantage was +0.010595 Å (95% CI [−0.030799, 0.053301]; Holm-adjusted p = 0.2468), and the TM-score advantage was +0.000811 (95% CI [−0.000220, 0.001837]; Holm-adjusted p = 0.4761). Each of these four bootstrap intervals included zero. The directions and magnitudes of the paired means varied across benchmark and metric, so a single method did not have a uniform numerical advantage across the primary comparisons.

The four primary paired comparisons did not detect a statistically significant difference between FM and Matched Residual DDPM after Holm correction. This non-detection is not evidence of statistical equivalence, nor does it establish that one refinement is uniformly better or worse. The separate DDPM-versus-Bit results in the paired statistical analysis are secondary to this direct FM-versus-DDPM comparison and do not change its inference.

The FM-versus-Bit analysis provides a distinct result on CAMEO TM-score: its Holm-adjusted p-value was 0.00879188717, consistent with a significant positive paired rank shift after Holm correction. In the same analysis, however, the bootstrap CI for the *mean* TM-score improvement included zero. These are different inferential summaries and should be reported together. In particular, a significant FM-versus-Bit result alongside a non-significant DDPM-versus-Bit result does not imply that FM significantly outperformed DDPM; the relevant direct FM-versus-DDPM tests were non-significant.

### 4.3 Refinement Efficiency

Table 3 reports residual-sampler model function evaluations (NFEs), not the number of iterations used for the DPLM base generation. The Local Bit output has no residual-refinement NFE. Latent Residual FM used 100 residual-model NFEs, whereas the full ancestral Matched Residual DDPM used 1000. FM therefore used one-tenth as many residual-model function evaluations as the matched DDPM baseline. The NFE comparison concerns the refinement samplers alone and is not a measurement of end-to-end wall-clock speed; it does not justify a claim that FM was ten times faster in elapsed time.

At the level of the observed structural-quality means, FM and DDPM produced nearby values in both benchmarks, although the direction of their numerical difference depended on the metric. Despite the tenfold difference in residual NFE, the four primary paired tests did not detect a significant FM-versus-DDPM structural-quality difference. These findings support the narrower conclusion that substantially lower residual-sampling NFE was observed without a *detected* loss of structural quality in the specified primary paired comparisons. They do not establish that the methods are equivalent or that a loss is impossible under other sampling conditions or evaluation populations.

### 4.4 Relation to DPLM-2.1 Paper References

Table 4 presents the DPLM-2.1 paper results solely as external reference values, separate from the local matched comparison in Table 3. The paper-reported Bit-based model had CAMEO RMSD/TM-score of 6.4028/0.8380 and PDB-date RMSD/TM-score of 3.2213/0.9043. The paper's Bit + RESDIFF row reported 6.1781/0.8428 and 3.0168/0.9076, respectively. Its Bit + FM row reported 6.1825/0.8414 and 2.8697/0.9099, and its Bit + FM + RESDIFF row reported 6.0765/0.8456 and 2.7884/0.9146. These figures are paper-reported references, not values obtained by re-evaluating the local predictions.

The paper's Bit + FM refers to its data-space FM approach, not the Latent Residual FM tested here. Likewise, the Local Matched Residual DDPM is not an exact reproduction of paper RESDIFF. In our local evaluation, Matched Residual DDPM did not show improvements of the same magnitude as the paper-reported RESDIFF reference. This does not refute the paper-reported RESDIFF findings: the local matched DDPM is not an exact RESDIFF reproduction, and the paper and local results are not a controlled head-to-head comparison. Differences in training, implementation, sampling, conditioning, and evaluation realization limit direct interpretation of the absolute values across those settings. Accordingly, the paper rows contextualize the local findings but are not used to assert statistical superiority of a local method over a paper-reported method.

<!-- Author-review checklists below; remove them from the final thesis if desired. -->

## Claims supported by the results

- On CAMEO, both local residual refinements improved mean BB RMSD and TM-score relative to Local Bit; the FM effect on PDB-date means was very small.
- Matched Residual DDPM improved both CAMEO means relative to Local Bit but did not improve either PDB-date mean metric.
- None of the four primary FM-versus-DDPM paired tests detected a statistically significant difference after Holm correction; this is not an equivalence finding.
- FM used 100 residual-model NFEs, one-tenth of the 1000 used by full ancestral Matched Residual DDPM, without a detected structural-quality loss in the primary paired comparisons.
- FM-versus-Bit CAMEO TM-score showed a significant positive paired rank shift after Holm correction, while the bootstrap CI for its mean improvement included zero.
- Table 4 consists of external paper references, not measurements from the local matched evaluation.

## Claims NOT supported by the results

- FM is statistically superior to DDPM.
- FM and DDPM are statistically equivalent.
- FM is 10x faster in wall-clock time.
- Local Matched Residual DDPM exactly reproduces RESDIFF.
- The local results refute the paper-reported RESDIFF findings.
- Paper Bit + FM and our Latent Residual FM are the same method.
