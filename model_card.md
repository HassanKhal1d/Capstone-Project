# Model Card Collection: Black-Box Optimization Functions f1–f8

**Project:** Imperial Business School BBO Capstone Competition
**Author:** Student Researcher
**Version:** Final (Week 13)
**Date:** March 2026

---

## How to Read These Model Cards

Each card documents one black-box oracle function treated as a surrogate optimisation
target. The "model" in each case is the best surrogate pipeline developed over 13 weeks
to approximate and optimise the function. Performance is reported as best observed output
value per week. All inputs are normalised to [0, 1]^d.

---

---

# Model Card: f1

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f1 |
| Input dimensions | 2 (X1, X2) |
| Output type | Scalar, near-zero positive (with rare negative values) |
| Observations at close | 22 |
| Best observed output | 6.60 x 10^-6 (Week 12) |
| Surrogate family | Gaussian Process (RBF, Matern), Copula-warped GP, quadratic polynomial |

## Purpose

f1 is a 2-dimensional black-box function with an extremely narrow productive ridge
concentrated along X2 near X2 = 0.655 at X1 = 0.684. The optimisation objective is
to maximise the scalar output. The function was treated as a test case for strategies
handling near-zero outputs spanning approximately 200 orders of magnitude.

## Intended Use

- Benchmarking surrogate behaviour on pathologically scaled outputs
- Testing output warping strategies (copula transform, log10 transform)
- Studying 1-dimensional active ridge detection in 2D spaces
- Educational demonstration of acquisition function failure under flat posterior surfaces

## Out-of-Scope Use

- Any application requiring reliable global coverage; the productive region covers
  less than 1% of the domain
- Surrogate validation studies assuming well-behaved Gaussian outputs in raw scale

## Training Data and Inputs

Inputs are 2-dimensional vectors in [0, 1]^2. No external training data was used.
All 22 observations were generated through sequential Bayesian optimisation submissions.
The dataset is heavily concentrated near X1 = 0.684, X2 in [0.655, 0.730] due to
acquisition bias from Week 9 onward.

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | 7.71 x 10^-16 | 7.71 x 10^-16 |
| 3 | GP-UCB | 5.31 x 10^-15 | 5.31 x 10^-15 |
| 9 | Copula GP + TuRBO | 1.87 x 10^-9 | 1.87 x 10^-9 |
| 10 | Log10 GP + LogEI | 2.03 x 10^-8 | 2.03 x 10^-8 |
| 11 | Quadratic ridge extrapolation | 6.50 x 10^-7 | 6.50 x 10^-7 |
| 12 | Quadratic ridge extrapolation | 6.60 x 10^-6 | 6.60 x 10^-6 |
| 13 | Quadratic ridge extrapolation | -1.22 x 10^-3 | 6.60 x 10^-6 |

**Total improvement from Week 0 to best:** approximately 8.6 billion times in output
magnitude, achieved primarily through the introduction of the Gaussian copula transform
in Week 9.

## Key Structural Findings

- The function is effectively 1-dimensional; X1 is structurally flat (ARD length scale
  hits upper bound in all GP fits from Week 9 onward)
- A single ultra-narrow ridge exists in X2; outputs span 200 orders of magnitude
- The ridge is locally concave in log10(y) space; a degree-2 polynomial fitted to the
  three most productive observations reliably estimates the peak
- Jackknife Spearman correlations: X2 dominates (rho = 0.328, std = 0.047);
  X1 negligible (rho = 0.179, std = 0.049)

## Limitations

- The true global maximum is unknown; best observed value may be far from the ceiling
- GP acquisition collapsed to a flat surface from Week 3 to Week 8 due to near-zero
  output scale; eight evaluations were effectively wasted
- Week 13 regression (y = -1.22 x 10^-3) confirms the ridge has a sharp left boundary;
  over-stepping X2 below approximately 0.640 exits the productive region entirely
- With N = 22, no statistically validated claim about global optimality is possible

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications. The function
is a synthetic oracle used in an academic competition context.

## Recommendations

- Future work should fit the quadratic exclusively to the three most recent productive
  observations and validate curvature sign before submitting
- The Gaussian copula transform is essential for any GP-based approach on this function;
  raw-scale GP fitting is not viable
- Acquisition function optimisation is not meaningful once the posterior surface is flat;
  direct structural reasoning should replace surrogate guidance at that stage

---

---

# Model Card: f2

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f2 |
| Input dimensions | 2 (X1, X2) |
| Output type | Scalar, bounded approximately [-0.08, 0.68] |
| Observations at close | 22 |
| Best observed output | 0.682 (Week 5) |
| Surrogate family | GP (RBF + WhiteKernel), Nadaraya-Watson kernel regression |

## Purpose

f2 is a 2-dimensional black-box function with a single high-value basin located in
X1 in [0.660, 0.715] and X2 in [0.820, 0.900]. The optimisation objective is to
maximise the scalar output. The function served as a test case for monotone feature
exploitation and directional acquisition filtering.

## Intended Use

- Benchmarking GP-UCB with directional candidate filtering on low-dimensional smooth
  surfaces
- Studying the effect of WhiteKernel inclusion on calibration in small-sample regimes
- Educational demonstration of X1-dominant structure detection via partial correlation
  and mutual information

## Out-of-Scope Use

- High-confidence optimality claims; N = 22 is insufficient to rule out undiscovered
  basins
- Production deployment without independent validation of the basin structure

## Training Data and Inputs

Inputs are 2-dimensional vectors in [0, 1]^2. All 22 observations were generated
through sequential Bayesian optimisation. The dataset is concentrated in the cluster-b
region (X1 in [0.66, 0.72], X2 in [0.82, 0.90]) from Week 5 onward due to acquisition
bias.

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | 0.611 | 0.611 |
| 4 | GP ensemble | 0.566 | 0.611 |
| 5 | Yeo-Johnson ARD-GP + UCB | 0.682 | 0.682 |
| 9 | ARD-GP UCB + spatial weight | 0.576 | 0.682 |
| 12 | Geometry-first maximin | 0.637 | 0.682 |
| 13 | GP posterior mean slice | 0.596 | 0.682 |

**Note:** The best observed value of 0.682 was achieved in Week 5 and was not improved
upon in any subsequent week despite eight further submissions.

## Key Structural Findings

- X1 is the dominant driver: Pearson r = 0.424, Spearman rho = 0.361, MI = 0.215
- X2 is statistically significant as a secondary driver (p = 0.049)
- A single high-value cluster exists at X1 in [0.66, 0.72], X2 in [0.82, 0.90]
- GP models consistently produced upward quantile bias (quantile near 1.0), indicating
  systematic underestimation of the true surface in the high-value region
- Nadaraya-Watson kernel regression with h = 0.04 produced more stable local predictions
  than GP marginal likelihood optimisation at N = 22

## Limitations

- The best observed value has not improved since Week 5; it is unclear whether 0.682 is
  near the true maximum or whether undiscovered territory exists in X2 below 0.820
  or above 0.900
- GP hyperparameter optimisation became unreliable in later weeks; length scale and
  noise estimates were inconsistent across restarts
- With N = 22, the unsampled gap in X2 within the active band remains unexplored

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications.

## Recommendations

- The GP posterior mean slice approach (fixing X1 at confirmed values, sweeping X2) is
  more reliable than unconstrained GP argmax at this sample size
- A dedicated geometric gap-filling query targeting X2 in [0.820, 0.843] along X1 = 0.703
  represents the highest remaining information-gain opportunity
- WhiteKernel must be retained; its removal in Week 4 degraded calibration significantly

---

---

# Model Card: f3

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f3 |
| Input dimensions | 3 (X1, X2, X3) |
| Output type | Scalar, negative, bounded approximately [-0.40, -0.025] |
| Observations at close | 27 |
| Best observed output | -0.025630 (Week 12) |
| Surrogate family | GP (Matern + WhiteKernel), KNN, KRR, ensemble |

## Purpose

f3 is a 3-dimensional black-box function with a flat interior plateau. The optimisation
objective is to maximise the scalar output (i.e. minimise the magnitude of the negative
value). The function served as a test case for noise-robust local exploitation and
surrogate selection under low signal-to-noise conditions.

## Intended Use

- Benchmarking KNN and KRR as alternatives to GP on weakly structured surfaces
- Studying fANOVA-based variance decomposition for identifying separable dimensions
- Educational demonstration of plateau behaviour and the limits of surrogate-guided
  search at small N

## Out-of-Scope Use

- Applications requiring confident identification of the global maximum; the function
  plateau makes ranking within the high-value region unreliable
- Any surrogate comparison study treating GP as a reliable mean estimator on this
  function without acknowledging its systematic instability

## Training Data and Inputs

Inputs are 3-dimensional vectors in [0, 1]^3. All 27 observations were generated
through sequential Bayesian optimisation. Late-stage observations are concentrated in
X2 in [0.77, 0.92] and X3 in [0.38, 0.52] due to structural targeting from Week 11
onward.

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | -0.03484 | -0.03484 |
| 11 | KRR exploitation | -0.02753 | -0.02753 |
| 12 | KRR + structural targeting | -0.02563 | -0.02563 |
| 13 | High-X2 plateau probe | -0.05252 | -0.02563 |

**Note:** No improvement was recorded from Week 0 through Week 10 (eleven consecutive
submissions). The first improvement occurred in Week 11 via KRR-based exploitation
within a structurally defined target region.

## Key Structural Findings

- fANOVA variance decomposition: X3 = 54.3%, X2 = 33.2%, separability = 94.4%
- X1 contributes negligible variance; mutual information is effectively zero across
  all weeks
- The high-value region is an interior plateau: conditional std in X3 in [0.40, 0.60]
  is approximately 0.009
- KNN consistently outperformed GP as a point estimator (Week 4: KNN LOO error 4.2 x
  10^-3 vs GP 6.8 x 10^-2)
- GP marginal likelihood optimisation produced pathological kernels that inverted X2
  and X3 importance in multiple weeks

## Limitations

- Eleven consecutive weeks without improvement indicate the function is near-flat in
  the high-value region and resistant to surrogate-guided search
- LOO RMSE of 0.079 exceeds the expected improvement margin of approximately 0.003,
  meaning surrogate guidance cannot reliably rank candidates within the plateau
- SIR direction was unstable throughout (bootstrap angle = 58.4 degrees at Week 13)
  and should not be used for directional guidance
- With N = 27, the true maximum may lie in unexplored X2 territory above 0.920

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications.

## Recommendations

- KRR with LOO-CV hyperparameter selection is the recommended surrogate for this
  function; GP should not be used without fixed or tightly bounded hyperparameters
- Structural targeting based on fANOVA (X2 in [0.80, 0.92], X3 in [0.38, 0.52])
  should be retained as the candidate generation constraint
- Boundary artefact rejection (excluding KRR peaks at X3 > 0.55 without data support)
  is essential to prevent wasted evaluations

---

---

# Model Card: f4

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f4 |
| Input dimensions | 4 (X1, X2, X3, X4) |
| Output type | Scalar, ranging from approximately -32.6 to +0.555 |
| Observations at close | 43 |
| Best observed output | 0.555 (Week 10) |
| Surrogate family | GP (Matern ARD), SVM, ensemble, EI acquisition |

## Purpose

f4 is a 4-dimensional black-box function with a sharply bounded positive plateau
surrounded by catastrophically negative regions. The optimisation objective is to
maximise the scalar output. The function served as a test case for trust-region
exploitation, empirical constraint encoding, and catastrophic failure avoidance.

## Intended Use

- Benchmarking trust-region Bayesian optimisation (TuRBO-style) on functions with
  narrow feasible regions surrounded by infeasible walls
- Studying empirical floor constraints derived from observed failure patterns
- Educational demonstration of the consequences of GP overconfidence and unconstrained
  acquisition in skewed landscapes

## Out-of-Scope Use

- Any strategy relying on global GP mean estimates; the output distribution is highly
  asymmetric and the plateau occupies a very small fraction of the domain
- Applications where a single catastrophic negative output would be unacceptable without
  prior feasibility screening

## Training Data and Inputs

Inputs are 4-dimensional vectors in [0, 1]^4. All 43 observations were generated
through sequential Bayesian optimisation. Late-stage observations are concentrated in
the mid-basin region (X1 in [0.35, 0.46], X2 in [0.39, 0.49], X3 in [0.34, 0.46],
X4 in [0.37, 0.48]).

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | -4.026 | -4.026 |
| 1 | Random sampling | -1.801 | -1.801 |
| 3 | GP-UCB | 0.070 | 0.070 |
| 6 | GP-UCB (full domain) | 0.347 | 0.347 |
| 10 | Neighbourhood EI + L-BFGS-B | 0.555 | 0.555 |
| 13 | VAR-GP-EI + X3 floor | 0.335 | 0.555 |

**Note:** Four consecutive improvements were achieved in Weeks 1, 3, 6, and 10.
No improvement occurred in Weeks 11, 12, or 13 despite targeted exploitation.

## Key Structural Findings

- ARD length-scale ratio was consistently approximately 1.2 across all weeks,
  indicating no sparse active subspace; all four dimensions contribute similarly
- Pairwise interaction heatmaps confirm the mid-basin constraint: all positive
  observations satisfy each dimension in approximately [0.29, 0.55]
- X3 < 0.340 is an empirically confirmed infeasible floor: all observations with
  X3 below this threshold return y < 0
- The plateau occupies approximately 2.3% of the domain volume at radius 0.10
  around the incumbent
- GP gradient at the incumbent was approximately 10^-31 in Week 12, confirming
  the surrogate has no usable directional signal near saturation

## Limitations

- The best observed value of 0.555 has not improved since Week 10; whether this
  is near the true maximum or a local plateau is unknown
- GP directional guidance failed completely in the neighbourhood of the incumbent;
  L-BFGS-B gradient optimisation produced near-zero EI in Weeks 11 and 12
- SIR bootstrap stability degraded from 31 to 43 degrees across weeks; the method
  is unreliable on this function and should be excluded
- With N = 43, the neighbourhood saturation is low (approximately 2.3%) but
  remaining improvement may be below surrogate detection thresholds

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications. The
catastrophic negative output region is a structural property of the synthetic oracle
and does not represent real-world harm.

## Recommendations

- The empirical X3 floor (X3 >= 0.340) must be enforced in all future queries;
  removing it caused the Week 9 catastrophic failure (y = -4.630)
- EI with xi = 0.0 within a tight trust region (radius 0.12) around the incumbent
  is the recommended acquisition strategy at this stage
- Heatmap-constrained mid-basin bounds should be used for all candidate generation;
  global search has not produced improvements since Week 6

---

---

# Model Card: f5

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f5 |
| Input dimensions | 4 (X1, X2, X3, X4) |
| Output type | Scalar, strictly positive, heavy-tailed, range approximately [0.11, 8343] |
| Observations at close | 33 |
| Best observed output | 8343.237 (Week 13) |
| Surrogate family | Log-GP (Matern ARD), log-SVR ensemble, EI, GP posterior mean |

## Purpose

f5 is a 4-dimensional black-box function with monotone increasing structure toward
the (1, 1, 1, 1) corner of the domain. The optimisation objective is to maximise the
scalar output. The function served as the primary test case for log-transform
stabilisation, heavy-tailed surrogate modelling, and corner-exploitation strategies.

## Intended Use

- Benchmarking log-transformed GP surrogates on functions with multi-order-magnitude
  output ranges
- Studying SIR-based directional guidance in monotone corner-dominated functions
- Educational demonstration of EI versus UCB behaviour under log-space modelling with
  heavy systematic underestimation

## Out-of-Scope Use

- Any surrogate strategy applied in the raw output scale; variance instability makes
  GP fitting in raw scale numerically unreliable
- Strategies relying on global exploration; 26 of 33 observations outside the
  [0.975, 1.0]^4 corner return y < 1090

## Training Data and Inputs

Inputs are 4-dimensional vectors in [0, 1]^4. All 33 observations were generated
through sequential Bayesian optimisation. From Week 10 onward, observations are
concentrated in the upper corner with per-dimension floors enforced (X1 >= 0.985,
X2 >= 0.999, X3 >= 0.998, X4 >= 0.970).

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | 1088.86 | 1088.86 |
| 2 | UCB BO | 2434.33 | 2434.33 |
| 3 | Log-GP EI | 4171.12 | 4171.12 |
| 7 | Log-GP EI + X3 boundary | 6575.30 | 6575.30 |
| 10 | Corner GP posterior mean | 6971.76 | 6971.76 |
| 11 | Corner GP posterior mean | 7735.18 | 7735.18 |
| 12 | Corner GP posterior mean | 8219.09 | 8219.09 |
| 13 | Corner GP posterior mean + SIR | 8343.24 | 8343.24 |

**Note:** f5 is the only function to record an improvement in every active week from
Week 7 onward. Thirteen consecutive submissions produced monotone improvement from
Week 7 to Week 13.

## Key Structural Findings

- The function is monotone increasing in all four dimensions toward the (1, 1, 1, 1)
  corner; 26 of 33 off-corner observations return y < 1090
- X3 is the dominant driver (partial r = 0.638); SIR dimension importance ranking
  is X2 > X3 > X4 > X1 with jackknife angular deviation of 0.91 degrees
- Log-GP calibration error is below 1% on the five most recent high-value corner
  observations; posterior uncertainty is approximately 0.019 in log units
- The function exhibits systematic GP underestimation in the corner: Week 10
  underestimation was 2.3%, Week 11 was 9.3%, confirming upside potential remains
- Isomap manifold analysis shows continued exponential increase toward the boundary
  with no plateau detected at Week 13

## Limitations

- The true ceiling is unknown; the function may continue increasing beyond the
  current best if corner points above all current floors exist
- GP underestimation bias in the corner means EI-guided acquisition may be overly
  conservative; posterior mean maximisation is preferred in the terminal phase
- With N = 33, the per-dimension floor constraints (X2 >= 0.9990, X3 >= 0.9980)
  leave very limited candidate volume; distance guard violations are possible

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications.

## Recommendations

- Log transform (shift = 1.0) must be applied at all times; raw-scale GP is not viable
- Per-dimension floors should be anchored to the most recent incumbent and tightened
  progressively; X4 can be relaxed slightly relative to X1, X2, X3 given its lower
  SIR loading
- GP posterior mean maximisation with SIR as a soft tiebreaker (weight = 0.02) is
  the recommended terminal-phase acquisition strategy

---

---

# Model Card: f6

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f6 |
| Input dimensions | 5 (X1, X2, X3, X4, X5) |
| Output type | Scalar, negative, bounded approximately [-2.76, -0.22] |
| Observations at close | 33 |
| Best observed output | -0.224 (Week 10) |
| Surrogate family | GP (Matern ARD + WhiteKernel), anchored kernel, EI acquisition |

## Purpose

f6 is a 5-dimensional black-box function with a smooth interior ridge governed
primarily by X4 and X5. The optimisation objective is to maximise the scalar output
(i.e. minimise the magnitude of the negative value). The function served as a test
case for active subspace identification, kernel anchoring, and acquisition portfolio
design.

## Intended Use

- Benchmarking ARD-GP on functions with confirmed low-effective-dimensional structure
- Studying the consequences of incorrect sparse-active-subspace priors (asymmetric
  length scale bounds) versus data-driven GP fitting
- Educational demonstration of fANOVA variance decomposition and XGBoost gain as
  complementary methods for identifying dominant input dimensions

## Out-of-Scope Use

- Any strategy enforcing asymmetric ARD length scale bounds without data justification;
  Week 9 revealed the function is globally smooth with approximately uniform length
  scales, not sparse-active-subspace
- Acquisition strategies relying on EI when predicted means are clustered within
  0.03 units of the incumbent; EI becomes unreliable at that margin

## Training Data and Inputs

Inputs are 5-dimensional vectors in [0, 1]^5. All 33 observations were generated
through sequential Bayesian optimisation. Late-stage observations are concentrated
in X4 in [0.68, 0.88] and X5 in [0.04, 0.14] following structural identification
from Week 9 onward.

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | -0.714 | -0.714 |
| 1 | Random sampling | -0.687 | -0.687 |
| 2 | UCB BO | -0.565 | -0.565 |
| 9 | ARD-GP EI + neighbourhood | -0.249 | -0.249 |
| 10 | Portfolio UCB + copula EI | -0.224 | -0.224 |
| 13 | Bracketing GP EI | -0.949 | -0.224 |

**Note:** The best observed value of -0.224 was achieved in Week 10 and was not
improved upon in Weeks 11, 12, or 13.

## Key Structural Findings

- fANOVA assigns 93.9% of variance to X4 and X5 combined (main effects only),
  with 6.1% interaction
- XGBoost gain independently confirms X5 (gain = 0.467) and X4 (gain = 0.354)
  as dominant dimensions
- Partial correlations: X5 (p = 1.7 x 10^-5), X4 (p = 0.012); X1, X2, X3 are
  not statistically significant
- Critical finding from Week 9: the function is NOT sparse-active-subspace in the
  GP kernel sense; all five ARD length scales are approximately uniform when fitted
  without asymmetric bounds, indicating a broad smooth gradient rather than a sharply
  localised subspace
- Quadratic bracketing of the three best observations in X4 and X5 identifies
  an interpolated peak at X4 = 0.744, X5 = 0.098

## Limitations

- The best observed value has not improved since Week 10; the Week 13 regression
  (y = -0.949) indicates the search has moved outside the productive ridge
- X3 drift was identified as the primary failure mechanism in Week 12; small
  deviations in X3 outside [0.50, 0.65] significantly degrade the GP posterior peak
- Asymmetric ARD length scale bounds (used in Weeks 3 through 8) encoded an
  incorrect structural belief and constrained the GP from fitting the true surface
- With N = 33, the interpolated X4/X5 peak at (0.744, 0.098) has not been
  directly sampled

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications.

## Recommendations

- The anchored kernel (X4 length scale = 0.603, X5 = 0.887, X2 = 10.0) validated
  in Week 12 should be used as the starting point for all future GP fits
- Candidate generation should be restricted to X4 in [0.68, 0.88], X5 in [0.02,
  0.18], X1 <= 0.70, X3 in [0.50, 0.65]
- A GP mean comparison gate (reject candidates where peak drop exceeds 0.05) should
  be applied to prevent X3 drift repeating the Week 13 regression

---

---

# Model Card: f7

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f7 |
| Input dimensions | 6 (X1, X2, X3, X4, X5, X6) |
| Output type | Scalar, positive, range approximately [0.003, 2.695] |
| Observations at close | 42 |
| Best observed output | 2.695 (Week 13) |
| Surrogate family | GP + SVR + KNN + XGBoost ensemble, CV-weighted |

## Purpose

f7 is a 6-dimensional black-box function with a single isolated high-value basin.
The optimisation objective is to maximise the scalar output. The function served as
a test case for ensemble surrogate design, multi-model CV weighting, and trust-region
exploitation in weakly structured high-dimensional spaces.

## Intended Use

- Benchmarking four-model CV-weighted ensembles (GP, SVR, KNN, XGBoost) on diffuse
  weakly structured surfaces
- Studying TDA Mapper and Isomap for basin isolation detection in 6D
- Educational demonstration of active subspace identification via GP ARD kernel
  analysis on functions with one dominant dimension

## Out-of-Scope Use

- Any single-surrogate approach (GP alone, KNN alone); all single-model strategies
  produced consistently lower performance than the four-model ensemble
- Global exploration strategies; 91% of observations yield y < 0.7, confirming
  the high-value basin occupies a small fraction of the domain

## Training Data and Inputs

Inputs are 6-dimensional vectors in [0, 1]^6. All 42 observations were generated
through sequential Bayesian optimisation. High-value observations (y > 1.35) all
satisfy X1 < 0.13 and X6 > 0.667; late-stage observations are concentrated in
this subregion.

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | 1.365 | 1.365 |
| 4 | Four-model ensemble | 1.599 | 1.599 |
| 6 | Ensemble UCB | 2.448 | 2.448 |
| 10 | Trust region ensemble | 2.558 | 2.558 |
| 11 | Trust region ensemble | 2.582 | 2.582 |
| 13 | Low-X2 boundary exploration | 2.695 | 2.695 |

**Note:** The four-model CV-weighted ensemble is the only surrogate configuration
that produced multiple consecutive improvements. TuRBO-M (Week 7) and single GP
strategies (Weeks 7, 8, 9) all produced regressions.

## Key Structural Findings

- GP ARD kernel identifies X6 as the most active dimension (length scale = 0.061)
  with X1 and X2 moderately active; X3 and X4 are effectively flat (length scale
  = 5.0)
- Active subspace analysis confirms approximately 1 to 2 effective dimensions
  near the optimum (eigenvalue ratio 5.30 vs 1.45)
- SIR dominant direction: X1, X6, X5 (stable across bootstrap resampling)
- TDA Mapper confirms the high-value basin as a topologically isolated single
  component
- Local linear coefficient for X2 is -0.680 (from top-10 observations at Week 13);
  this gradient was unexploited until Week 13 and produced the final improvement

## Limitations

- The best observed value has been improving but the true ceiling is unknown; the
  Week 13 point at X2 = 0.222 is the first observation below the prior cluster
  minimum of X2 = 0.249
- TuRBO-M produced a catastrophic regression in Week 7 (y = 0.013) by leaving the
  high-value basin; multi-trust-region strategies should not be used on this function
  without prior basin confirmation
- Isomap embedding is unstable at N = 42 in 6D and should be used only for
  visualisation, not for candidate generation
- The negative X2 gradient direction has been validated by only one observation;
  further confirmation is needed

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications.

## Recommendations

- The four-model ensemble (GP + SVR + KNN + XGBoost) with 5-fold CV inverse-MSE
  weighting is the only validated configuration and should be retained
- Trust region width should be contracted progressively (0.20 in Week 10, 0.15 in
  Week 11, 0.12 in Week 12) following TuRBO contraction rules
- The confirmed structural constraints (X1 < 0.13, X6 > 0.667) should be enforced
  as hard filters on all candidate generation

---

---

# Model Card: f8

## Model Overview

| Field | Detail |
|-------|--------|
| Function | f8 |
| Input dimensions | 8 (X1, X2, X3, X4, X5, X6, X7, X8) |
| Output type | Scalar, positive, range approximately [5.54, 9.97] |
| Observations at close | 52 |
| Best observed output | 9.966 (Week 13) |
| Surrogate family | ARD-GP (Matern 2.5) + KNN ensemble, LogEI, active subspace fixing |

## Purpose

f8 is an 8-dimensional near-deterministic black-box function with a sparse active
subspace. The optimisation objective is to maximise the scalar output. The function
served as the primary test case for active subspace identification, inactive dimension
fixing, and LogEI in high-dimensional spaces with near-ceiling behaviour.

## Intended Use

- Benchmarking ARD-GP with asymmetric length scale bounds on functions with confirmed
  sparse active subspaces
- Studying the effect of fixing inactive dimensions at incumbent values (SAASBO-style)
  on convergence near the ceiling
- Educational demonstration of dual-basin structure detection using TDA Mapper and
  the risk of cluster-crossing in near-deterministic functions

## Out-of-Scope Use

- Any strategy that allows inactive dimensions (X2, X4, X5, X6, X8) to vary freely;
  Week 9 demonstrated that corner collapse on inactive dimensions caused a significant
  regression (y = 9.772 vs incumbent 9.949)
- Global exploration strategies; the active subspace occupies a small region near
  X1 in [0.16, 0.24], X3 in [0.08, 0.21], X7 in [0.13, 0.27]

## Training Data and Inputs

Inputs are 8-dimensional vectors in [0, 1]^8. All 52 observations were generated
through sequential Bayesian optimisation. The three highest-value observations (rows
42, 48, 49) share identical values for all five inactive dimensions (X2 = 0.210,
X4 = 0.198, X5 = 0.814, X6 = 0.388, X8 = 0.664), which defines the cluster A basin.

## Performance

| Week | Method | Output (y) | Best Observed |
|------|--------|-----------|---------------|
| 0 | Initial Output | 9.598 | 9.598 |
| 2 | UCB BO | 9.835 | 9.835 |
| 3 | GP-UCB ARD | 9.949 | 9.949 |
| 10 | LogEI active subspace | 9.940 | 9.949 |
| 11 | LogEI active subspace | 9.943 | 9.949 |
| 13 | Partial correlation probe | 9.966 | 9.966 |

**Note:** The Week 3 best of 9.949 was not improved for nine consecutive weeks
(Weeks 4 through 12). The final improvement to 9.966 in Week 13 was achieved by
probing slightly below the incumbent in both X1 and X3 simultaneously within the
cluster A inactive dimension configuration.

## Key Structural Findings

- Three active dimensions confirmed across all methods: X1 (ARD length scale = 0.20
  to 0.30), X3 (= 0.20 to 0.30), X7 (= 0.60); five inactive dimensions (X2, X4,
  X5, X6, X8 hit upper bounds >= 10)
- The function is near-deterministic: WhiteKernel noise estimate = 10^-8 throughout
- TDA Mapper confirms two distinct high-value basins: cluster A (rows 42, 48, 49;
  y >= 9.940) and cluster B (rows 41, 43, 47; y = 9.77 to 9.86)
- GP 1D marginal probes show posterior mean peaks at X1 = 0.16 to 0.18 and X3 =
  0.17 to 0.20, slightly below the Week 12 incumbent at X1 = 0.191, X3 = 0.203
- Inactive dimension values are identical across the three cluster A observations;
  this is the defining structural feature of the basin

## Limitations

- The best observed value of 9.966 was achieved in the final week; whether further
  improvement is possible below X1 = 0.163 and X3 = 0.178 is unknown
- Week 12 cluster B targeting (y = 9.849) confirmed cluster B is inferior to cluster
  A; no further cluster B exploration is warranted
- Nine consecutive weeks without improvement from Week 3 to Week 12 indicate the
  incumbent plateau is very flat and the improvement margin is small
- LogEI is essential near the incumbent; standard EI suffers numerical instability
  when improvement margins are below approximately 0.02 units

## Ethical Considerations

No human subjects, sensitive data, or real-world deployment implications. The
near-deterministic nature of the function means surrogate uncertainty is structural
rather than aleatoric and should not be interpreted as genuine measurement noise.

## Recommendations

- Inactive dimensions must be fixed at incumbent cluster A values
  (X2 = 0.210, X4 = 0.198, X5 = 0.814, X6 = 0.388, X8 = 0.664) for all
  future queries
- Active dimension search should be restricted to X1 in [0.14, 0.22], X3 in
  [0.15, 0.22], X7 in [0.13, 0.27]
- LogEI with xi = 0.0 is the recommended acquisition function; L-BFGS-B warm
  starts from the top 25 LogEI candidates should be used to avoid local optima
  in the 3-dimensional active subspace

---

*End of Model Card Collection*

*All model cards reflect the state of knowledge at the close of Week 13.
No further updates are planned. These cards should be read alongside the
full submission logs (Week_2_Submission_Log.md through Week_13_Submission_Log.md)
and the inputs_13.txt and outputs_13.txt data files.*
