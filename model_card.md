## Model Card: Sequential Bayesian Optimisation Framework for Black-Box Functions f1–f8


### Overview

**Model name:** Sequential Pre-Commitment Bayesian Optimisation (SPCBO)
**Version:** Final — Week 9 (competition close)
**Type:** Sequential surrogate-based black-box optimisation framework
**Task:** Global maximisation of eight unknown scalar functions of varying dimensionality (2D to 9D) over the unit hypercube [0,1]^d
**Developer:** Individual student researcher, academic BBO competition
**Primary surrogate:** Gaussian Process (Matérn 2.5 kernel with ARD) with function-specific acquisition, supplemented by KNN, SVR, XGBoost, and KRR ensembles as needed
**Acquisition functions used across the competition:** UCB (function-specific β), EI, Thompson Sampling, MES (Week 7 f2), copula-EI (Week 8 f6), KRR-mean + GP-variance UCB (Week 9)


### Intended Use

**Primary tasks this approach is suitable for:**

- Sequential black-box optimisation of expensive, noiseless or low-noise scalar functions where the number of evaluations is severely constrained (under 15 total per function)
- Functions defined over bounded continuous domains with no gradient access and no known functional form
- Settings where understanding the function landscape (active subspaces, dominant dimensions, monotonic trends) is as important as finding the maximum
- Research and educational contexts requiring structured experimental reasoning: pre-specified hypotheses, falsifiable assumptions, and post-hoc reflection on why strategies succeeded or failed

**Use cases that should be avoided:**

- Functions requiring more than one query per decision cycle (this framework is strictly one query per function per week)
- High-dimensional functions above approximately 10 dimensions at these sample sizes — the surrogate models fitted here are unreliable at N < 15 with d > 9
- Settings where calibrated uncertainty is not monitored — the calibration diagnostic battery (NLL, z-score, calibration gap, CI coverage) is essential for safe use of this framework, and omitting it risks confidently selecting poor candidates
- Production decision systems where a single erroneous query carries high cost — the framework is validated for academic competition conditions with no irreversible consequences
- Functions with strong discontinuities or highly multimodal structure in low-dimensional spaces, where the GP smoothness assumption fails systematically (as observed with f3 and f7)


### Strategy Details: Evolution Across Ten Rounds

### Week 0 — Initialisation
A single oracle-supplied point was provided for each function. These served as the anchor observations. No surrogate was fitted. Best observed values established for f2 (0.611), f7 (1.365), f8 (9.598).

### Week 1 — Random Space-Filling
Uniform random sampling across [0,1]^d for each function. No acquisition function. Purpose: initialise the dataset with low-bias, space-filling coverage to enable reliable surrogate fitting in Week 2. f4 improved over Week 0 to −1.801 via random sampling alone; f5 and f8 were matched or slightly regressed, confirming that random sampling alone is inefficient once a good seed exists.

### Week 2 — Homoskedastic GP-UCB Baseline
First surrogate-based strategy. Matérn GP with RBF kernel, uniform UCB acquisition, grid search over β and length-scale. Key finding: uniform β without per-function calibration underexplored extreme or steep regions (f1–f4). The root cause was uncertainty miscalibration — GP variance underestimated the true output variability. f5 improved to 2434 and f8 to 9.835 (both held as bests for weeks afterwards), demonstrating the approach works when the function scale is compatible with the default GP prior.

### Week 3 — Feature-Guided Heteroskedastic GP (Most Productive Week)
Per-function EDA was introduced: Pearson, Spearman, Kendall, mutual information, and partial correlation analyses were used to identify dominant dimensions and set function-specific UCB β. Pseudo-heteroskedastic GPs with Matérn 2.5 kernels replaced the homoskedastic baseline. This was the single most productive week: f4 reached its all-time best of 0.071, f5 reached its all-time best of 4171 (at a point near the X3=1.0 boundary, confirming the X3 gradient), and f8's Week 2 best was confirmed as a local rather than global maximum. The pre-commitment framework (specifying hypotheses and decision rules before observing results) was used here for the first time and proved essential for distinguishing genuine improvement from noise.

### Week 4 — Ensemble Surrogates + Variance Inflation
SVR (ε-SVR with RBF kernel), KNN (k=5), and XGBoost were added to form ensembles with cross-validation-based weighting. Variance inflation (3× posterior standard deviation scaling) was introduced to address severe GP overconfidence on f4 and f7, which had produced z-scores beyond ±4 and CI failures in Week 3. The inflation corrected both functions (f4 calibration gap fell from 0.49 to 0.08; f7 z-score recovered from −4.11 to +1.01). However, ensemble averaging was found to suppress posterior variance as a side effect, replacing overconfidence on some functions with underconfidence on others. f5 suffered a critical calibration failure in this week (NLL=19.66, z=−6.30, CI FAIL, quantile=0.000) caused by an incorrectly applied log-transform in combination with CV-LCB selection.

### Week 5 — Calibration Repair and WhiteKernel Restoration
WhiteKernel (noise absorption term in the GP kernel) was restored after its removal in Week 4 caused overconfident predictions across multiple functions. A Yeo-Johnson input warp on X1 was tested for f2 but falsified: the relationship was moderately linear (Pearson r=0.424), so the warp distorted the kernel distance metric and worsened calibration (quantile moved from 0.947 to 0.978). f2 reached its all-time best of 0.683 this week, despite the warp failure, because the directional bias toward high X1 was correct. The key learning from this week: calibration quality and optimisation performance are distinct objectives that can decouple completely — a new best was found with a miscalibrated surrogate, and an improving calibration trajectory does not guarantee acquisition improvement.

### Week 6 — Conservative Baseline Restoration
TuRBO's trust region restriction was removed for all functions where it was preventing full-domain exploration. SVM masking was removed when the decision boundary proved unreliable at N≈35. β was raised to 3.5 with 40 L-BFGS-B restarts to counteract posterior variance contraction. The philosophy: stop adding complexity that has not been validated on these specific functions and restore the simplest configuration that found each function's best value.

### Week 7 — Competition-Validated Acquisition Innovation
Methods drawn from published BBO competition literature were applied: Max Value Entropy Search for f2 (SigOpt/Eriksson et al. 2021), CMA-ES for GP hyperparameter tuning on f1 (BBOB COCO 2013), TuRBO-M with three parallel trust regions for f7 (Eriksson et al. NeurIPS 2019), opposition-based sampling for f4 (GECCO 2022), and Differential Evolution for f6 (motivated by SAASBO). Most strategies produced no improvement or regression. The structural hypothesis that had motivated this week — that standard UCB/EI had been exhausted and competition-validated structural diversity would break stagnation — did not hold. Competition-winning methods optimise for general function classes; they do not necessarily transfer to specific unknown functions.

### Week 8 — Recovery via Validated Empirical Configurations
Asymmetric ARD length-scale bounds were applied to f6 (motivated by Bull 2011) to prevent the HP optimiser from collapsing to structurally incorrect hyperparameter regions — the failure mode identified from Week 7's Differential Evolution run. A dual-acquisition portfolio (UCB β=5.0 and β=8.0 plus copula-EI, motivated by Salinas et al. ICML 2020 and Srinivas et al. 2010's theoretical β schedule) was applied to f6. The Week 4 four-model ensemble (GP, SVR, KNN, XGBoost with CV-based weighting) was restored for f7, which had found the f7 best of 1.599 and had been incorrectly replaced in Week 7. TuRBO-M was removed from f7. The core principle: validated empirical configurations are more reliable than theoretically motivated alternatives that have not been tested on these specific functions.

### Week 9 — Final Structured Exploitation
KRR (Kernel Ridge Regression) was introduced as a decoupled mean estimator: KRR fitted to all observations via LOO-CV grid search over alpha and gamma, providing a stable mean surface; a fixed Matérn GP provided calibrated posterior variance independently. The acquisition score was UCB = mean_KRR(x) + β × sigma_GP(x), decoupling mean and uncertainty estimation to prevent the GP instability that had recurred under high-sample-count conditions. Candidate generation used basin-constrained Sobol sequences informed by the accumulated structural knowledge: for f7, hard filters x1<0.20, x5<0.55, x6>0.50 based on the cluster structure in the top-3 observed points; for f8, active subspace filtering retaining only x1, x3, x7 as meaningful dimensions. Minimum distance guards (δ_min) were applied to all functions to prevent near-duplicate queries.


### Performance

Primary metric across all functions: **best value observed** by competition close.

| Function | Dimensionality | All-Time Best | Week Found | Key Structural Finding |
|----------|---------------|---------------|------------|----------------------|
| f1 | 2D | ~5.5e-57 | Week 3 | Confirmed flat — no exploitable structure |
| f2 | 2D | 0.683 | Week 5 | X1 dominant (Pearson r=0.424); monotonically positive |
| f3 | 4D | −0.035 | Week 0 | Best was the initial point; KNN > GP as estimator; rough surface |
| f4 | 5D | 0.071 | Week 3 | Active subspace x2, x4 near optimum; plateau since Week 3 |
| f5 | 5D | 4171 | Week 3 | X3 dominant; log-scale required; best near x3=1.0 boundary |
| f6 | 6D | −0.565 | Week 2 | X4, X5 active subspace confirmed (partial r both p<0.01); best found early |
| f7 | 7D | 1.599 | Week 4 | Top-3 cluster at low-x1, low-x5, x6>0.70; four-model ensemble validated |
| f8 | 9D | 9.835 | Week 2 | Active subspace x1, x3, x7 confirmed by ARD + SVM-RFE + MI convergence |

Secondary metrics tracked per function per week:

- **Negative log-likelihood (NLL):** primary calibration quality indicator; values below 0 indicate well-calibrated GP uncertainty
- **Z-score:** standardised prediction error; target range ±2; values beyond ±4 indicate critical overconfidence
- **95% CI pass/fail:** binary indicator of whether the true observation fell within the GP's 95% credible interval
- **Calibration gap:** absolute difference between observed quantile position and 0.5; values below 0.15 indicate good calibration
- **Best-observed improvement per week:** the primary competition metric

Notable calibration trajectory for f7 (the most volatile function): calibration gap fell from 0.500 (Week 3, critical failure, z=−4.11) to 0.345 (Week 4) to 0.136 (Week 5) following progressive variance inflation and ensemble restoration. This three-week monotonic improvement was the clearest example of calibration repair working as intended.

Notable persistent failures: f3, f4, f5, f6 all showed stagnation of the best-observed value from Week 3 or Week 4 onward despite seven or more additional submissions. This is documented honestly rather than explained away.


### Assumptions and Limitations

**Core assumptions underlying the approach:**

1. The GP with Matérn 2.5 kernel is a reasonable surrogate for each function. This holds for smooth functions (f2, f5, f8) but is violated for rough or non-stationary functions (f3, f7), where KNN consistently outperformed GP as a point estimator.

2. Active subspace structure can be detected from small samples using EDA. This assumption held for f6 and f8 where multiple independent methods (ARD, partial correlation, MI, SVM-RFE) converged. It was partially falsified for f3, where EDA signals were inconsistent across weeks and no stable active subspace emerged.

3. Calibration diagnostics (NLL, z-score, calibration gap) provide reliable signals for model quality. This holds when N is sufficient for the GP hyperparameter landscape to be well-conditioned, but at N<15, the NLL landscape can be flat or multimodal, causing L-BFGS-B to converge to structurally incorrect hyperparameters — the failure mode identified in Week 7 for f6.

4. A pre-committed hypothesis and decision rule framework prevents retrospective rationalisation. This assumption held; the records show multiple cases where post-hoc reasoning would have produced different (and incorrect) conclusions about which component of a strategy drove an outcome.

**Key limitations:**

- Sample sizes of 9–13 per function are insufficient for reliable mutual information estimation, more than 3-fold cross-validation on higher-dimensional functions, or stable ensemble weight estimation. MI estimates in Weeks 2–4 should be treated as exploratory signals, not confirmed structural findings.
- The acquisition framework is strictly one query per function per cycle, which prevents within-week error correction. A single misdirected query (such as f5 Week 4's critical calibration failure) cannot be immediately recovered.
- Strategy conflation: each week introduced multiple simultaneous changes, making it impossible in several cases to attribute an outcome to a single component. This is an inherent limitation of the one-query-per-week structure.
- The approach assumes no observation noise by default. If the oracle outputs are noisy, variance inflation may mask genuine signal, and the WhiteKernel noise absorption term may absorb real function variation.
- Competition-validated methods from the BBO literature (MES, TuRBO-M, CMA-ES, opposition sampling, Differential Evolution) were applied in Week 7 without prior validation on these specific functions. All produced no improvement or regression. This is the clearest evidence that method transfer from general BBO competition results to specific unknown functions is unreliable.


### Ethical Considerations

**Transparency and reproducibility:**

The most important ethical feature of this framework is the pre-commitment structure. Every submission from Week 3 onward was preceded by a written specification of the directional hypothesis, falsifiable key assumptions (maximum three), and decision rules that would govern the next week's strategy regardless of the outcome. These were written before results were observed and have not been altered retrospectively. This makes the experimental record reproducible in the specific sense that the reasoning process — not just the code — is verifiable. Anyone reading the submission logs and review documents can reconstruct not only what was done but why it was expected to work and why it sometimes failed.

**Honest reporting of stagnation:**

Several functions (f3, f4, f5, f6) showed no improvement after Week 3 or Week 4 despite seven further submissions. This is reported directly in the performance table above and in the submission logs. No selective reporting of results has been applied. The all-time bests shown are the actual competition bests, including cases where the initial random or oracle-supplied point was never beaten.

**Limitation of calibration as a proxy for correctness:**

The framework uses calibration diagnostics as the primary quality signal for surrogate models. This is documented honestly as a proxy rather than a ground truth: a CI pass means the surrogate's uncertainty was appropriately sized at the one queried point, not that the landscape reconstruction is globally correct. A surrogate can be well-calibrated at the queried point and deeply wrong in unexplored regions. Users of this framework in real-world settings should not interpret calibration success as model validation.

**Suitability for real-world adaptation:**

The framework as documented is suitable for adaptation to real-world expensive-evaluation optimisation problems (materials discovery, hyperparameter tuning, engineering design) with the following conditions: the evaluation budget must be similarly constrained (10–20 total evaluations per function), the function must return a scalar output without gradient information, and the practitioner must be willing to invest time in per-function EDA between each evaluation cycle. The pre-commitment structure is strongly recommended for any real-world application to prevent the kind of retrospective rationalisation that small-sample optimisation experiments are prone to. The calibration diagnostic battery (NLL, z-score, calibration gap, CI coverage tracked together) should be treated as a minimum monitoring requirement, not an optional addition.


### Model Card Information

**Last updated:** March 2026
**Contact:** BBO Competition Student Researcher
**Framework references:** Srinivas et al. (ICML 2010) for GP-UCB theory; Bull (2011) for asymmetric length-scale bounds; Eriksson et al. (NeurIPS 2019) for TuRBO; Salinas et al. (ICML 2020) for copula acquisition; Gebru et al. (2018) for model card framework
**Related documents:** [Link to be added on submission] (dataset documentation), [Link to be added on submission] (repository overview)
**Repository:** [Link to be added on submission]
