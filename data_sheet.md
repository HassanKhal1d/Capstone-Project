# Dataset Datasheet | Black Box Optimization Capstone Project


| Category        | Details |
|-----------------|--------|
| **Competition** | Black-Box Optimisation (BBO) — Academic Module Competition |
| **Functions** | f1 (3D), f2 (3D), f3 (4D), f4 (5D), f5 (5D), f6 (6D), f7 (7D), f8 (9D) |
| **Input domain** | \[0, 1\]^d for all functions; all dimensions bounded identically |
| **Objective** | Maximise the unknown scalar output of each function |
| **Query budget** | 13 submissions across all functions; 1 query per function per week |
| **Primary use** | Surrogate modelling, Bayesian optimization, experimental design research |


### Motivation

**What task does this dataset help to solve?**  
This dataset was generated to support sequential black-box optimization across eight unknown functions, f1 through f8, of varying dimensionality (3D to 9D). All inputs are constrained to [0,1]^d. The task is to find the global maximum of each function using only the observed input-output pairs, with no access to gradients, functional form, or noise structure. The dataset records every query submitted to the competition oracle and the corresponding output returned, forming the empirical basis for surrogate model fitting, acquisition function design, and landscape structure discovery.

**Who created the dataset and who funded it?**  
The dataset was provided as part of a Bayesian Optimization competition within the ML and AI certificate programme delivered by Imperial College Business School in collaboration with Emeritus. The underlying function generator used to create the dataset was not publicly disclosed, and therefore cannot be definitively attributed to either institution. All observed data points were generated through the student’s sequential submissions during the competition, under a constrained evaluation budget, forming the final dataset used for analysis.


### Composition

**What does it contain?**  
The dataset comprises eight independently sampled black-box function datasets (f1–f8), collectively containing 271 input–output observations accumulated across thirteen weeks of sequential Bayesian optimization submissions. Each observation consists of a normalised input vector drawn from [0, 1]^d and a scalar oracle output, with dimensionality ranging from two (f1, f2) to eight (f8). All data were generated through active experimentation rather than drawn from a pre-existing corpus, meaning the datasets are incomplete samples of their respective continuous domains and carry no recommended train/test split. Instance counts at the close of Week 13 range from 22 observations (f1, f2) to 52 (f8), reflecting both the per-function submission budget and the varying difficulty of identifying productive regions. No missing values or incomplete instances are present. The datasets exhibit substantial structural heterogeneity: output ranges span from a narrow interval of roughly 0.76 units (f2) to over five orders of magnitude (f5), and the proportion of observations that lie in high-value regions varies from nearly all (f8) to fewer than five percent (f1, f4, f7). Observations within each function are sequentially dependent, as every week's sampling strategy was conditioned on all prior results, making the datasets more informative than equivalent random samples of the same size but also subject to sequential sampling bias toward regions identified as promising in earlier weeks.


| Function | Dimensions | N (Week 13) | Output Range | Best y | Active Dimensions | Primary Challenge |
|----------|-----------|-------------|-------------|-----------------|-------------------|-------------------|
| f1 | 2 | 22 | ~[−3.6×10⁻³, 6.6×10⁻⁶] | 6.6 × 10⁻⁶ | X2 only | Ultra-narrow ridge; 200 order-of-magnitude output span |
| f2 | 2 | 22 | [−0.080, 0.682] | 0.682 | X1 (dominant), X2 (secondary) | Small sample; GP hyperparameter instability |
| f3 | 3 | 27 | [−0.399, −0.026] | −0.026 | X2, X3 (separable) | High noise; flat plateau; KNN > GP throughout |
| f4 | 4 | 43 | [−32.6, 0.555] | 0.555 | All 4 (isotropic) | Extremely narrow positive plateau; catastrophic walls |
| f5 | 4 | 33 | [0.113, 8343.2] | 8343.2 | X2, X3 (dominant) | Heavy-tailed; 5-order magnitude range; monotone corner |
| f6 | 5 | 33 | [−2.757, −0.225] | −0.225 | X4, X5 (94% variance) | Broad smooth tilt; no sparse subspace despite prior belief |
| f7 | 6 | 42 | [0.003, 2.695] | 2.695 | X6 (highly active), X1, X2 | Isolated basin; weak global structure; 91% observations y < 0.7 |
| f8 | 8 | 52 | [5.541, 9.966] | 9.966 | X1, X3, X7 | Near-deterministic; dual-basin structure; inactive dimension fixing critical |



### Collection Process

**How was the data acquired?**  
QQueries were generated through a sequential Bayesian optimisation pipeline executed once per week per function. The core loop was: fit a surrogate model to all accumulated (x, y) pairs; optimise an acquisition function over [0,1]^d to select the next candidate; submit to the oracle; append the returned (x, y) pair to the dataset; update EDA. This loop ran for thirteen active weeks (Weeks 1 through 13), with Week 0 providing the initial oracle-supplied point for each function.

**Sampling Strategy**
The sampling strategy was neither purely random nor fully deterministic; it was adaptive
and hypothesis-driven, evolving materially across weeks in response to observed outcomes.
Week 1 used uniform random sampling to establish a baseline. From Week 2 onward, candidate
points were selected by optimising acquisition functions over surrogate models fitted to all
accumulated observations. The primary acquisition functions employed were:

- **Upper Confidence Bound (UCB)** — used throughout Weeks 2–8 as the default exploratory
  mechanism, with function-specific β parameters adjusted to balance exploration and exploitation
- **Expected Improvement (EI)** — adopted from Week 3 onward for functions with well-identified
  basins, particularly f4 and f5, where the goal shifted to beating a known incumbent
- **Thompson Sampling** — applied selectively in Weeks 5 and 7 to inject stochastic diversity
  and escape confirmed local maxima
- **Max Value Entropy Search (MES)** — trialled on f2 in Week 7 as a competition-backed
  alternative targeting global maximum uncertainty reduction
- **Log Expected Improvement (LogEI)** — introduced in Weeks 10–13 for near-ceiling functions
  (f5, f8) where standard EI suffered numerical instability near strong incumbents

Surrogate models evolved in parallel, progressing from single Gaussian Processes with RBF
or Matérn kernels through increasingly structured configurations:

- **Gaussian Processes with ARD kernels** — the primary global surrogate throughout, with
  asymmetric length-scale bounds introduced from Week 8 to enforce known active subspace
  structure
- **Support Vector Regressors, k-Nearest Neighbours, and XGBoost** — incorporated as ensemble
  components from Week 4 onward, weighted by cross-validated inverse MSE
- **Kernel Ridge Regression** — adopted for f3 from Week 9 as a numerically stable alternative
  to GP marginal likelihood optimisation at small sample sizes
- **Nadaraya–Watson kernel regression** — used on f2 in Week 11 when GP hyperparameter
  fitting became unreliable

Candidate generation relied on quasi-random **Sobol** and **Halton** sequences to ensure
low-discrepancy domain coverage, consistently filtered by minimum-distance guards
(δ_min ∈ [0.02, 0.07] depending on function and week) to prevent near-duplicate evaluations.
From Week 9 onward, for functions where structural knowledge was sufficient, deterministic
or geometry-first methods replaced acquisition-function-driven selection entirely:

- **Quadratic ridge extrapolation in log₁₀ space** for f1, targeting the estimated peak of
  the X2 ridge directly from three locally informative observations
- **Maximin gap-filling** for f2 and f3, placing queries at the largest unsampled gap within
  the confirmed high-value region when surrogate models were too unstable to guide selection
- **Corner-restricted GP posterior mean maximisation with per-dimension floors** for f5,
  reflecting confirmed monotone structure toward the (1,1,1,1) boundary
- **Active-subspace-fixed LogEI with inactive dimensions anchored to incumbent values** for f8,
  reducing the effective search space from eight to three dimensions

As a result, the accumulated datasets are subject to **sequential sampling bias**: regions
identified as promising early in the process are systematically over-represented relative to
a uniform random sample of the same size. Functions where the productive region was identified
early (f5, f8) have the most concentrated late-stage observations, while functions that
resisted structural identification (f3, f7) retain comparatively more dispersed coverage
across the domain.

**Time frame of data collection**  
Data were collected over fourteen discrete weeks (Week 0 through Week 13) during the Spring 2025 term of the Imperial Business School BBO Capstone Project. Each week constituted one observation per function, yielding a maximum of fourteen observations per function from the submission process. Week 0 observations were provided by the course oracle as fixed starting points and were not chosen by the student. The total active collection period spanned approximately thirteen weeks of sequential decision-making, with one candidate submitted per function per week under a strict budget constraint. No data were collected outside this structured weekly cadence, and no retrospective or batch collection took place.

### Preprocessing / Cleaning / Labelling

**Was any preprocessing or labeling performed?**  
Transformations were applied selectively and only when EDA provided statistical justification. No global preprocessing pipeline was applied uniformly. All transformations are pipeline steps applied at model-fitting time; raw (x, y) records are stored without modification.
The most important and consistently validated transformation was a log-transform of f5's outputs, applied from Week 3 onward. f5's output range spans four orders of magnitude (685 to 4171), which makes GP fitting in the original scale unreliable. The log-transform stabilised the variance and enabled calibrated uncertainty estimates. The Week 4 reflection documented the consequences of getting this wrong: the log-GP + log-SVR ensemble used in Week 4 applied the log-transform incorrectly in combination with a CV-LCB selection criterion, producing the worst calibration result of the entire competition (NLL=19.66, z=−6.30, CI FAIL, quantile=0.000 — the truth fell entirely outside the predictive distribution). This was described in the Week 4 reflection as the most dangerous failure mode, because log-space calibration errors are amplified exponentially on back-transformation.
A Yeo-Johnson input warp on X1 was tested for f2 in Week 5. The hypothesis was that X1 had a non-linear monotonic relationship with f2's outputs. The test falsified this: the warp worsened calibration (quantile moved from 0.947 to 0.978) because EDA confirmed the relationship was moderately linear (Pearson r=0.424, Spearman ρ=0.361), so warping distorted the kernel distance metric without correcting any genuine non-stationarity. The warp was removed in Week 6.
A Box-Cox output warp was applied to f3 in Week 6 to stabilise GP calibration on its weakly negative outputs, with the condition that it would be retained only if the calibration gap stayed below 0.15. No other transformations were applied across the dataset.

**What are the intended uses?**  
The dataset is intended for educational and research use in surrogate-based optimisation, specifically for comparing GP, KRR, KNN, SVR, XGBoost, and ensemble methods as point estimators and uncertainty providers under small-sample sequential conditions; for investigating how GP posterior miscalibration manifests and how it can be diagnosed using NLL, z-score, calibration gap, and CI coverage as a coherent diagnostic battery; for active subspace detection using ARD length-scales, partial correlation, mutual information, and SVM-RFE on functions with confirmed sparse structure; and for studying the pre-commitment experimental design framework itself, where hypotheses and decision rules are specified before results are observed.

**What are the inappropriate uses?**
The dataset should not be used to claim statistically validated global optima. With N approximately 9 to 13 per function, no guarantee exists that the best-observed value is near the true global maximum. For f3 and f6, there is strong evidence from calibration metrics that the surrogate models' uncertainty is too compressed for reliable exploration, meaning the dataset may be missing high-value regions entirely. The dataset should also not be used to generalise structural findings — for instance, the finding that X3 dominates f5 is specific to f5 and cannot be extrapolated to other functions. It should not be used for high-stakes production decisions, and it should not be treated as if the calibration diagnostics validate the surrogate's accuracy; a CI pass means the surrogate's uncertainty was appropriately sized at the one queried point, not that the landscape reconstruction is correct.

**Are there risks or limitations for future use?**  
The most important bias is acquisition bias: after Week 1, all queries are drawn from regions the surrogate identified as promising, creating a highly non-uniform design. There is also a confirmation bias documented in the reflections — structural hypotheses formed in early weeks (e.g. that X3 dominates f5) influenced where later queries were placed, potentially self-confirming an incomplete picture of the surface. Strategy conflation is a persistent limitation: each week introduced multiple simultaneous changes (surrogate, acquisition function, hyperparameter tuning method, candidate generation strategy), making it often impossible to determine which specific component drove an improvement or regression from a single observation.


### Distribution

**How has the dataset been distributed?**  
The dataset exists as structured Markdown submission logs and review documents stored within the project knowledge repository for this competition. The raw data can be extracted from the tables in these documents into CSV or DataFrame format. No public repository has been designated.

**What are the terms of use?**  
The dataset was generated within an academic module competition and is subject to the terms of that module. Sharing is currently limited to academic assessment contexts. Any use of the dataset should credit the competition framework from which the oracle function evaluations were obtained. The unknown functions f1 through f8 are proprietary to the competition and cannot be reverse-engineered from this dataset alone. The query points should not be submitted to any active iteration of the same competition.


### Maintenance

**Who maintains the dataset?**  
The dataset is maintained by the student researcher who generated it. It is append-only within the competition cycle: each weekly submission adds exactly one record per function and no prior records are modified. If an EDA or calibration error is discovered in a prior week, the correction is documented in the subsequent week's review rather than retroactively altering the original record, preserving the integrity of the sequential experimental log.
At competition close (Week 13), the complete set of submission logs and review documents will constitute the final archive, stored as part of the academic coursework submission. Retention follows institutional data retention policies for coursework. There is no version control system beyond the implicit week-number versioning; each week's documents are an append to the record, not a replacement of it.
