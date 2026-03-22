# Week 8 — Validated Anchor Recovery + Structured Rollback from Competition Method Failures

This document outlines the Week 8 submission plan. The strategy shifts away from the
competition-method experimentation of Week 7 and returns to empirically validated
configurations on functions where gradient-free HP tuning produced catastrophic regressions,
while retaining competition-backed approaches only where Week 7 evidence supports them.

The central diagnosis from Week 7 is that Differential Evolution and CMA-ES produced
hyperparameter basins that were structurally incorrect — uniform or near-uniform length scales
on functions with confirmed sparse active subspaces. Multi-trust-region exploration added
complexity without structural justification on f7. The response is not to abandon structured
methods wholesale, but to rebuild from the configurations that demonstrably worked.

Emphasis:
- Return to empirically validated hyperparameter configurations on f6, f7, and f4
- Replace gradient-free HP tuning with multi-start L-BFGS-B seeded from validated anchors
- Srinivas et al. (2010) theoretically justified beta schedule applied to stagnated functions
- Bull (2011) asymmetric length scale bounds replacing unconstrained ARD optimisation
- Squirrel NeurIPS 2020 acquisition portfolio replacing single-acquisition strategies on f6
- Accept confirmed plateaus on f1, f3 and allocate minimal budget accordingly

---

## Meta-Strategy

### Validated Anchor Recovery + Theoretically Grounded Exploration Pressure

**Primary surrogate**
- ARD-GP per function with per-dimension length scales
- Multi-start L-BFGS-B with validated anchor seeds replacing CMA-ES and Differential Evolution
  on all functions where gradient-free tuning failed in Week 7
- Asymmetric length scale bounds enforced per Bull (2011) for all functions with confirmed
  sparse active subspaces: active dimensions capped short, inactive dimensions forced long

**Hyperparameter tuning policy**
- Multi-start L-BFGS-B with anchor seeding for f6, f7, f4 — gradient-free methods removed
  after producing structurally incorrect HP basins in Week 7
- Week 2 validated configuration (β=5.0, ℓ=0.2) explicitly seeded as one restart on f6
- Multi-start L-BFGS-B retained for f5 and f8 — confirmed smooth likelihoods
- No CMA-ES or Differential Evolution on any function this week

**Warping policy**
- Log transform retained for f5 — validated across six weeks, calibration stable
- Box-Cox retained for f3 — calibration stable, acquisition is the failure not the warp
- No other warping introduced

**Residual correction**
- MLP absent from all functions without exception
- No reintroduction — confirmed structurally unreliable at N between 20 and 50

**Acquisition**
- Dual UCB portfolio (β=5.0 primary, β=8.0 secondary) for f6 — Srinivas et al. (2010)
  justifies higher β as N grows and GP variance contracts
- Copula-transformed EI as third portfolio component for f6 — Salinas et al. (ICML 2020)
- EI concentrated near x3 boundary for f5 — validated as the only approach that found
  the Week 3 best
- UCB β=2.0 with X1-directional filter for f2 — restored from Week 5 successful configuration
- Low-β UCB for f1 — confirmed flat surface, exploration pressure unnecessary
- Thompson Sampling removed from all functions — produced stagnation in prior weeks

**Candidate generation**
- Full-domain Sobol with asymmetric-bounds ARD-GP for f6 — no dimension restrictions,
  no fixed coordinates, no trust regions
- Sobol with soft directional weighting for f7 (X1 low, X5 low, X6 high)
- Full-domain Sobol for f8 with ARD-informed active subspace filtering
- Full-domain Sobol for f4 — all structured acquisition exhausted, revert to exploration
- Low-X3/high-X2 candidate concentration for f3 — final structured attempt before plateau
  acceptance

**Ensemble policy**
- f7: restore Week 4 four-model ensemble (GP, SVR, KNN, XGB) with CV-based weighting —
  TuRBO-M removed; Week 4 ensemble found the best observed and is the only validated
  configuration for this function
- f8: retain 0.7 GP + 0.3 KNN — empirically validated across three consecutive weeks
- All other functions: single GP surrogate, no ensemble

---

## Learning Objectives

1. Does restoring the Week 2 hyperparameter anchor (β=5.0, ℓ=0.2) spirit via multi-start
   L-BFGS-B with asymmetric bounds and a Srinivas-scaled β beat the f6 best of −0.5649
   that has stood unbeaten for six consecutive submissions?

2. Does the Squirrel NeurIPS 2020 dual-acquisition portfolio (UCB β=5.0 + UCB β=8.0 +
   copula EI) produce more corner-directed candidate selection than any single acquisition
   used in Weeks 3–7?

3. Does restoring the Week 4 four-model ensemble on f7 recover acquisition quality lost
   when TuRBO-M replaced it — does the ensemble weight distribution match Week 4?

4. Does acknowledging confirmed plateaus on f1 and f3 and minimising budget allocation to
   them free enough experimental clarity to focus on functions still improvable?

5. Does Bull (2011) asymmetric length scale enforcement prevent the HP optimiser from
   collapsing to the uniform-length-scale basin that caused Week 7 catastrophic regressions?

---

## Structural Hypothesis for Week 8

- Week 7's gradient-free methods failed because they were solving the wrong subproblem.
  CMA-ES and Differential Evolution are designed to escape multimodal landscapes, but
  the GP log marginal likelihood for f6 is not pathologically multimodal — it has a clear
  basin near short active-dimension length scales. The failure was not the search algorithm;
  it was unconstrained HP bounds that allowed convergence to structurally incorrect regions.
  Asymmetric bounds (Bull 2011) restore the constraint that prevents this.

- The Week 2 configuration (β=5.0, ℓ=0.2 uniform) found f6's best observed by accident
  at N=15. At N≈35, GP posterior variance has contracted. The Srinivas et al. (2010)
  theoretical schedule implies β_theory≈22 at this sample size. A portfolio of β=5.0
  and β=8.0 respects the Week 2 validation while applying theoretically justified upward
  pressure appropriate for the larger dataset.

- The f7 four-model ensemble is the only configuration that found y=1.598 in Week 4.
  TuRBO-M's parallel trust regions addressed a different failure mode — multimodal basins
  in 7D — but there is no confirmed evidence of multimodality in f7. The four-model
  ensemble with CV-based weighting is the correct structural response.

Week 8 prioritises recovery from Week 7 method failures over continued experimentation.
The core insight is that validated empirical configurations are more reliable than
theoretically motivated alternatives that have not been validated on these specific functions.

---

## Function-Specific Strategy Summary

| Function | Week 8 Direction | Research Reference | Key Change from Week 7 |
|---|---|---|---|
| f1 | Low-β UCB + full-domain Sobol, minimal budget | — | CMA-ES removed; plateau acknowledged; single low-cost query |
| f2 | RBF+WhiteKernel GP + UCB β=2.0 + X1>0.70 filter | Week 5 validated config | MES removed; restore Week 5 acquisition that found current best |
| f3 | KNN-mean GP-uncertainty EI, low-X3/high-X2 | — | GA tuning removed; final structured attempt before plateau |
| f4 | Full-domain Sobol random exploration | — | Opposition sampling and CMA-ES removed; exhausted structured methods |
| f5 | Log-GP + EI + x3≥0.90 strip + global component | SMAC, Hutter et al. 2011 | Retain Week 7 strategy; add 20% global Sobol pool for diversity |
| f6 | RBF-ARD GP + dual UCB β=5.0/8.0 + copula EI | Srinivas 2010, Bull 2011, Squirrel NeurIPS 2020, Salinas ICML 2020 | DE removed; L-BFGS-B with W2 anchor seed; asymmetric ℓ bounds; acquisition portfolio |
| f7 | Week 4 four-model ensemble + CV weights + directional Sobol | TuRBO, Eriksson et al. NeurIPS 2019 | TuRBO-M removed; restore GP+SVR+KNN+XGB with CV weighting |
| f8 | ARD-GP 0.7/0.3 ensemble + full-domain Sobol + active subspace filter | NeurIPS 2020 Bayesmark, NVIDIA Research | Exclusion zone removed; ARD-confirmed active dims guide Sobol bias |

---

## Competition Research Backing by Function

### f2 — Week 5 Validated Configuration
The Week 5 RBF + WhiteKernel GP with UCB β=2.0 and X1-directional filtering is the only
configuration that has beaten the Week 0 random baseline for f2. MES in Week 7 addressed
a different problem — reducing uncertainty about the global maximum location — which is
appropriate when the surface is unknown. At N≈20 with confirmed X1 dominance, the surface
is sufficiently understood. EI and MES require reliable mean estimates; UCB's exploration
term compensates for systematic underestimation by pushing acquisition above the mean.

### f5 — SMAC (Hutter et al. 2011)
Sequential Model-Based Algorithm Configuration established log-transformed GP with EI and
concentrated candidate generation near dominant boundary values as the standard for
heavy-tailed expensive optimisation. f5's dominant X3 gradient and four-order-of-magnitude
output range match the conditions SMAC was designed for. Week 8 retains this and adds a
20% global Sobol component to probe whether the Week 3 maximum is the global optimum or
a local one — a question that five consecutive local strategies cannot answer.

### f6 — Srinivas et al. (2010) + Bull (2011) + Squirrel NeurIPS 2020 + Salinas et al. (ICML 2020)
Four distinct academic contributions are integrated for f6.

Srinivas et al. (2010) ICML Theorem 1 derives β_t = 2·log(d_eff·t²·π²/6δ). With d_eff=3
(X4, X5, X2 confirmed active), N≈35, δ=0.1, this gives β_theory≈22. The portfolio uses
β=5.0 (Week 2 validation) and β=8.0 (conservative upward shift), both well below the
theoretical ceiling but restoring the corner-directed pressure that found the Week 2 best.

Bull (2011) JRSS-B proves that for sparse active subspace functions, optimal length scales
are asymmetric: active dimensions short, inactive dimensions long asymptotically. Active
dim bounds [0.03, 0.45], inactive dim bounds [0.80, 20.0] enforce this structurally and
prevent the HP optimiser from finding the uniform-ℓ basin that caused the Week 7 collapse.

The Squirrel team (Lindauer et al., NeurIPS 2020 BBO Challenge 3rd place) used a portfolio
of GP+EI, GP+UCB, and GP+PI with copula transformations. Portfolio acquisition hedges
against any single acquisition's failure mode. Week 8 applies this with weights
0.50 (UCB β=5.0), 0.30 (UCB β=8.0), 0.20 (copula EI).

Salinas et al. (ICML 2020) copula rank transformation maps raw y values to N(0,1) quantiles
via the empirical CDF. The f6 active subspace creates heavy-tailed y distribution — most
points return near −0.8 to −1.8 while the best cluster is narrow. Copula normalisation
prevents the GP likelihood being dominated by extreme low observations and produces reliable
EI variance estimates.

### f7 — Week 4 Four-Model Ensemble
The Week 4 GP + SVR + KNN + XGB ensemble with CV-based weighting is the only configuration
that found y=1.598 on f7. TuRBO-M (Eriksson et al. NeurIPS 2019) is designed for functions
with confirmed multimodal basins requiring parallel exploration. There is no confirmed
evidence of multimodality in f7's 7D space — the function has a diffuse but unimodal
structure where the four-model ensemble's non-parametric diversity (KNN, XGB) is the
correct response to the absence of GP stationarity assumptions in high dimensions.

### f8 — NVIDIA Research NeurIPS 2020 Bayesmark Framework
The 0.7 GP + 0.3 KNN ensemble with ARD-confirmed active subspace (X1, X3, X7) has
produced the best calibration f8 has achieved (Gap=0.134, Z=−0.34) and correctly
identified the local maximum. Full-domain Sobol replaces the exclusion-zone approach
from Week 7. If the Week 7 exclusion-zone query returned a new best, Week 8 exploits
the new region; if not, the local maximum at Week 3 is the global optimum and Week 8
uses the ARD-confirmed active subspace to bias a final targeted search in the low
X1/low X3/low X7 corner with tighter bounds.

## Function-Specific Strategies

# f1 — Sobol Sampling with Multi-Start L-BFGS-B Log-Transformed Heteroscedastic GP (HPT)

## Objective of Submission

Design an aggressive exploration strategy for a very flat, near-zero, expensive black-box function with:

- 2 inputs  
- 1 output  

The approach:

- Applies a log transform (with positive shift) to expand tiny output differences  
- Fits a Gaussian Process (GP) surrogate in log space  
- Uses Upper Confidence Bound (UCB) to select the next query point  


## Three Key Assumptions

1. Outputs are non-negative or extremely close to zero, so adding a small positive shift before log transformation is safe.  
2. After log transformation, outputs become approximately Gaussian and show meaningful variation.  
3. The function is smooth enough in 2D input space for a Matern GP to be a reasonable surrogate once transformed.


## Research Backing

This strategy draws from heteroscedastic Bayesian optimisation and output warping methods:

- Log transforms stretch tiny values into a numerically stable range.  
- L-BFGS-B is standard for GP hyperparameter optimisation via log marginal likelihood.  
- UCB balances mean and uncertainty in a principled way.

### Supporting Papers

- Cowen-Rivers, A., et al. (2022). *HEBO: Heteroscedastic Evolutionary Bayesian Optimisation*. JMLR 23(57).  
- De Dios Santacruz, C., et al. (2023). *Bayesian Optimisation with Heteroscedastic Noise*. Machine Learning.  
- Rasmussen, C.E., & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.


## Explorative Principle

### Core Problem

Raw outputs are extremely small (e.g., 1e-40). In raw space:

- GP cannot distinguish 1e-15 from 1e-80.  
- Everything appears numerically zero relative to noise.  
- Uncertainty becomes meaningless.

### Log Transformation

Apply:

y_log = log(y + ε)

This maps tiny values into a usable negative range (e.g., -20 to -5), allowing the GP to:

- Detect real variation  
- Learn meaningful structure  
- Produce calibrated uncertainty  

### Acquisition in Log Space

For each candidate:

UCB(x) = μ_log(x) + β · σ_log(x)

This encourages sampling where:

- The model is uncertain  
- The transformed output may be higher  

This is appropriate for extremely flat functions where discovering *any* slightly larger region is valuable.


## Why This Strategy Is Ideal for f1

- Raw GP fails due to near-zero outputs.  
- Log transformation restores numerical resolution.  
- GP can model structure once values are expanded.  
- UCB focuses on uncertain and potentially higher regions.  
- Sobol sampling ensures domain coverage.  
- Distance guard prevents redundant evaluations.  

This is critical when evaluations are expensive and the function is nearly flat.


## Tech Stack

- `numpy` — numerical operations  
- `scipy` — Sobol quasi-random sequence generation  
- `scikit-learn` — GP regression with Matern + WhiteKernel  


## Hyperparameters and Recommended Settings

### Output Transform

- `epsilon = 1e-12`  
  Ensures log(0) is avoided and maps zero outputs to ≈ -27.6.

### Gaussian Process

- Kernel: `Matern(ν=2.5) + WhiteKernel`  
- Length scale bounds: `[0.05, 10.0]` for both dimensions  
- Noise bounds: `[1e-6, 5.0]`  
- `n_restarts_optimizer = 20`

### Candidate Generation

- `n_sobol = 3000`  
- `min_dist_guard = 0.05`

### Acquisition

- `ucb_beta = 2.0`  
  Moderately aggressive exploration pressure.


## Hyperparameter Tuning Method

- Multi-start L-BFGS-B  
- Optimises GP log marginal likelihood in log-transformed space  
- Uses exact gradients  
- Multiple restarts reduce local optimum risk  


## Entire Flow of the Strategy

1. Collect existing data: `X_train_1`, `y_train_1`.  
2. Apply log transform with shift: `y_log = log(y + ε)`.  
3. Fit GP (Matern + WhiteKernel) using multi-start L-BFGS-B.  
4. Generate Sobol candidates across `[0,1]^2`.  
5. Apply minimum distance guard.  
6. Compute GP predictive mean and standard deviation in log space.  
7. Compute UCB score: `μ + βσ`.  
8. Select candidate with highest UCB.  
9. Evaluate expensive function and append data.


## Hypothesis Framework

### Core Assumptions

- Log-transformed outputs are approximately Gaussian.  
- Function is smooth enough for Matern GP.  
- High GP uncertainty corresponds to informative regions.

### If Assumptions Hold

- GP fits transformed data well.  
- Uncertainty estimates are realistic.  
- UCB spreads sampling across domain.  
- Best observed value improves or flatness is confirmed.

### If Assumptions Break

- Non-smooth function → GP misestimates uncertainty.  
- Log transform insufficient → distorted modelling.  
- Strong input-dependent noise → WhiteKernel inadequate.


## Critical Analysis

### Strengths

- Directly solves near-zero numerical collapse.  
- Principled probabilistic modelling.  
- UCB is simple and interpretable.  
- Sobol + distance guard ensures coverage.  
- Efficient gradient-based hyperparameter tuning.

### Weaknesses

- Depends on log transform validity.  
- GP struggles with discontinuities or strong non-stationarity.  
- More complex than simple non-parametric methods.

### Potential Failure Modes

- Sharp jumps or strong non-stationarity.  
- Significant negative outputs.  
- Extremely small evaluation budget.  
## Summary

This strategy prioritises geometric exploration and hyperparameter robustness over gradient exploitation.  

For a near-zero, expensive 2D black-box function, scrambled Halton sampling combined with CMA-ES stabilised Gaussian Processes provides a principled and computationally controlled approach to exploration.

---

## f2 — Yeo–Johnson Warped ARD Gaussian Process with UCB

## Objective

Design an aggressive but principled exploration strategy for an expensive 2D black-box function using:

- Yeo–Johnson input warping  
- Anisotropic (ARD) Gaussian Process  
- Upper Confidence Bound (UCB) acquisition  

Applied to `x_train_2` and `y_train_2`.

## Core Assumptions

1. **Structural dominance of x1**  
   The first input dimension contributes more strongly to variation in the objective than x2.

2. **Warping improves stationarity**  
   Yeo–Johnson transformation reduces skew in x1 and improves Gaussian Process fit.

3. **Moderate smoothness**  
   The function is sufficiently smooth in the transformed space for a Matérn 2.5 kernel to model accurately.

## Input Warping

- Apply Yeo–Johnson transformation to `x1` only  
- Leave `x2` untransformed  

This stabilizes the dominant dimension without distorting the weaker one.

## Surrogate Model

Anisotropic Gaussian Process with:

- Matérn 2.5 kernel  
- Separate length scales per dimension (ARD)  
- WhiteKernel for noise estimation  

Hyperparameters are optimized using multi-start L-BFGS-B on the log marginal likelihood.

## Acquisition Strategy

Upper Confidence Bound (UCB):

UCB(x) = μ(x) + β σ(x)

Where:

- μ(x) is the GP predictive mean  
- σ(x) is the GP predictive standard deviation  
- β = 2.0  

This balances exploitation and exploration.

## Candidate Generation

1. Generate 5000 Sobol sequence points over the 2D domain  
2. Apply directional filter: keep candidates with `x1 >= 0.70`  
3. Apply minimum distance guard: remove candidates within 0.05 of existing samples  

This maintains coverage while concentrating search in promising regions.

## Hyperparameters

- Warping: Yeo–Johnson on x1  
- Kernel: Matérn 2.5 + WhiteKernel  
- Length scale bounds (x1): [0.05, 3.0]  
- Length scale bounds (x2): [0.20, 10.0]  
- Noise bounds: [1e-6, 0.5]  
- GP restarts: 20  
- Sobol candidates: 5000  
- x1 threshold: 0.70  
- Minimum distance guard: 0.05  
- UCB beta: 2.0  

## Optimization Flow

1. Start with observed data (`x_train_2`, `y_train_2`)  
2. Generate Sobol candidates  
3. Apply directional filter  
4. Apply minimum distance guard  
5. Fit Yeo–Johnson on observed x1  
6. Transform training and candidate x1 values  
7. Fit ARD Matérn GP with multi-start L-BFGS-B  
8. Compute predictive mean and variance  
9. Evaluate UCB  
10. Select highest-UCB candidate for evaluation  

## Expected Outcomes

If assumptions hold:

- Shorter learned length scale for x1 than x2  
- UCB selects high-x1 candidates  
- Best observed value improves or stabilizes near optimum  

If assumptions fail:

- Directional filter may exclude optimal region  
- Warping may distort structure  
- GP may misestimate anisotropy  

## Rationale for Expensive Evaluations

Because evaluations are costly:

- Sobol ensures structured global coverage  
- Directional filtering reduces wasted evaluations  
- ARD GP provides calibrated uncertainty  
- UCB selects informative and high-potential points  

This maximizes information gain per evaluation relative to naive random or grid-based search.

---

## f3 — Voronoi Space Filling with Multi-Bandwidth KNN Ensemble and GP Expected Improvement

## Objective

Design an exploration strategy that searches the full four-dimensional space without directional bias, while using simple and reliable models to guide improvement over the current best value in `y_train_3`.

## Core Assumptions

1. The function has shifting important dimensions, so no fixed directional bias is safe.
2. KNN is a more reliable point predictor than a Gaussian Process for this function.
3. The primary failure so far has been poor spatial coverage, not poor modelling.


## Research Foundation

Voronoi-based maximin space filling is a well-established method for exploring expensive black-box functions when structural knowledge is limited.  
Multi-bandwidth KNN ensembles are appropriate when functions exhibit both local and global patterns.  
Expected Improvement (EI) is a classical acquisition function for surpassing a known best value.

### Supporting Literature

- Mak & Joseph (2018) — Minimax and maximin projection designs  
- Johnson, Moore & Ylvisaker (1990) — Minimax and maximin distance designs  
- Cover & Hart (1967) — Nearest neighbor methods  
- Hastie, Tibshirani & Friedman (2009) — The Elements of Statistical Learning  
- Mockus et al. (1978) — Bayesian methods for extremum seeking  


## Explorative Principle

The strategy combines three signals:

1. **Voronoi Maximin Score**  
   Measures distance from existing samples.  
   Points in large empty regions receive high scores.

2. **Multi-Bandwidth KNN Ensemble**  
   Predicts candidate values using multiple neighborhood sizes.

3. **Gaussian Process Expected Improvement**  
   Provides calibrated uncertainty to estimate probability of improving the current best.

The selected point must be:
- Far from existing samples  
- Predicted promising by KNN  
- Likely to improve the best observed value  


## Why This Strategy Fits f3

The function represented by `x_train_3` and `y_train_3` shows inconsistent dimensional importance.  
Earlier strategies that restricted search regions performed poorly.

Observations:

- KNN has been the most reliable point predictor  
- GP uncertainty remains useful for improvement estimation  
- Coverage has been insufficient  

This hybrid approach ensures unbiased exploration while retaining model guidance.


## Hyperparameters

### Transform

- Box-Cox shift: `-min(y) + 0.01`
- Lambda learned from data

### KNN Ensemble

- k = 3, 5 
  Captures local, medium, and global structure

### Gaussian Process

- Kernel: Matérn 2.5 + WhiteKernel  
- Length scale bounds: [0.01, 10.0]  
- Noise bounds: [1e-8, 1.0]  
- Restarts: 10  

### Candidate Generation

- Sobol points: 12000  
- Voronoi top percentage: 40%  
- Minimum distance guard: 0.05  

### Expected Improvement

- ξ (xi): 0.001  

### Hybrid Weights

- Space filling: 0.40  
- KNN prediction: 0.35  
- Expected Improvement: 0.25  


## Optimization Flow

1. Shift and transform `y_train_3` using Box-Cox  
2. Generate 12000 Sobol candidates  
3. Apply minimum distance guard  
4. Compute Voronoi maximin score (in reduced 3D subspace)  
5. Keep top 20% most isolated candidates  
6. Fit multi-bandwidth KNN ensemble using leave-one-out weighting  
7. Fit GP on Box-Cox transformed target  
8. Compute three normalized scores for each candidate:
   - Space filling score  
   - KNN predicted mean  
   - Expected Improvement  
9. Combine scores using fixed hybrid weights  
10. Select candidate with highest hybrid score  

## Expected Outcomes

If assumptions hold:

- Selected points lie in unexplored regions  
- New high-value regions are discovered  
- Exploration remains unbiased across dimensions  

If assumptions fail:

- KNN may mislead acquisition  
- Voronoi score may dominate excessively  
- GP uncertainty may be unreliable under incorrect transformation  


## Strengths

- Strong global coverage  
- No directional bias  
- Robust small-data models  
- Balanced exploration and exploitation  


## Weaknesses

- More complex than single-model approaches  
- Fixed hybrid weights may not be optimal  
- Voronoi score may dominate when dataset is very small  


## Failure Modes

- Strong global structure not captured by KNN  
- Highly irregular or chaotic function behavior  
- Excessive noise degrading EI reliability  
- Important structure located in regions appearing unpromising to local predictors  


This strategy prioritizes unbiased global coverage while retaining simple, stable predictive models to guide improvement in an expensive four-dimensional optimization setting.tured b## f4 — Sobol Sampling with ARD-GP + L-BFGS-B Hyperparameter Tuning (4D)

## Objective

Use a simple but strong Bayesian Optimization strategy to select the next evaluation point for an expensive black-box function represented by `x_train_4` and `y_train_4`.

The method uses:

- A single ARD Gaussian Process with a Matérn 2.5 kernel  
- Upper Confidence Bound (UCB) acquisition  
- Aggressive exploration with multiple restarts  
- Sobol sampling as global backup  


## Core Assumptions

1. All input dimensions in `x_train_4` are informative.
2. The function is smooth but not perfectly smooth.
3. The main risk is getting stuck in local regions, so strong exploration and many restarts are required.


## Research Foundation

Gaussian Processes with Matérn kernels are standard for modelling smooth black-box functions.  
GP-UCB has theoretical regret guarantees when the exploration parameter is sufficiently large.  
Multiple L-BFGS-B restarts are widely used in Bayesian Optimization systems to avoid poor local optima.

### Supporting Literature

- Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning  
- Srinivas et al. (2010) — Gaussian Process Optimization in the Bandit Setting  
- Brochu, Cora & de Freitas (2010) — Tutorial on Bayesian Optimization  


## Explorative Principle

The Gaussian Process builds a probabilistic model mapping `x_train_4` to `y_train_4`.

For any candidate point, it provides:

- Predictive mean  
- Predictive standard deviation (uncertainty)

The Upper Confidence Bound acquisition is defined as:

UCB(x) = μ(x) + β σ(x)

Where:

- μ(x) is the predicted mean  
- σ(x) is predictive uncertainty  
- β controls exploration strength  

A high β encourages exploration of uncertain but potentially high-value regions.

Multiple acquisition restarts ensure thorough search of the 4D acquisition surface.


## Why This Strategy Fits f4

Earlier complex ensembles and opposition-based strategies risk overfitting or instability.

A single clean ARD-GP:

- Is easier to interpret and debug  
- Uses all past data efficiently  
- Provides calibrated uncertainty  

High β encourages exploration when the optimum has not yet been located.

The combination of:

- Gradient-based acquisition search  
- Sobol candidate pool  

Provides both local refinement and global coverage.


## Hyperparameters

### Model

- Kernel: Matérn 2.5 + WhiteKernel  
- Length scale bounds (each dimension): [0.05, 10.0]  
- Noise bounds: [1e-6, 0.5]  
- GP restarts: 20  

### Acquisition

- UCB beta: 3.5–4.0  
- Acquisition restarts: 40  

### Candidate Backup

- Sobol pool size: 10000  
- Minimum distance guard: 0.05  


## Hyperparameter Tuning

GP hyperparameters are tuned by maximizing the log marginal likelihood using L-BFGS-B with multiple random restarts.

The acquisition function is optimized using L-BFGS-B from many random initial points in the unit hypercube.


## Optimization Flow

1. Fit ARD Gaussian Process (Matérn 2.5) to `x_train_4`, `y_train_4`
2. Define UCB acquisition
3. Run multiple L-BFGS-B optimizations of negative UCB from random starts
4. Generate Sobol candidate pool
5. Compute UCB for Sobol candidates
6. Apply minimum distance guard
7. Compare:
   - Best gradient-based candidate  
   - Best Sobol candidate  
8. Select the candidate with highest valid UCB
9. Evaluate the black-box function at that point


## Expected Outcomes

If assumptions hold:

- GP uncertainty estimates are meaningful  
- UCB balances exploration and exploitation  
- Best observed value improves over iterations  

If assumptions break:

- GP may misestimate uncertainty under high noise  
- Irrelevant dimensions may waste model capacity  
- UCB may explore inefficiently  


## Strengths

- Simple and interpretable  
- Well-studied theoretical foundation  
- Clear exploration–exploitation control via β  
- Strong acquisition optimization through restarts  


## Weaknesses

- Sensitive to kernel choice and hyperparameters  
- Computationally heavier due to restarts  
- Single GP may be too rigid for non-stationary structure  


## Failure Modes

- Sharp discontinuities  
- Heavy observational noise  
- Strong local irregularities  
- Highly non-stationary behavior not captured by Matérn kernel  

---
s or poor GP fit.  
- Very few evaluations make CMA-ES cost-dominant.  
- True optimum in a very small region not covered by Sobol + opposition sampling.

---

# f5 — Log-Transformed Gaussian Process Expected Improvement Concentrated Near x3 Boundary

## Objective of Submission

The objective is to propose the next query point for a 5-dimensional expensive black-box function using:

- A log-transformed Gaussian Process model  
- An Expected Improvement (EI) acquisition function  
- Candidate generation concentrated near the upper boundary of x3  

The strategy aggressively searches for improvements near the dominant dimension (x3) while maintaining limited global coverage.


## Three Key Assumptions

1. The function output has a large dynamic range and becomes more Gaussian-like after a log transform.

2. One input dimension (x3) is a dominant monotonic driver of the output, particularly near its upper boundary.

3. The Gaussian Process log marginal likelihood in the log-transformed space is smooth and well-conditioned, making multi-start L-BFGS-B effective.


## Research Backing

- Hutter, F., Hoos, H. H., and Leyton-Brown, K. (2011). *Sequential Model-Based Algorithm Configuration (SMAC).*  

- Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.  


## Explorative Principle

The exploration strategy combines three ideas tailored to a heavy-tailed, x3-dominated function:

1. Log transform for stabilising heavy-tailed outputs  
2. ARD Gaussian Process with gradient-based hyperparameter tuning  
3. Expected Improvement concentrated near the x3 upper boundary  


## Log Transform for Heavy-Tailed Outputs

When outputs span a large range, the distribution is often skewed and heavy-tailed.

Applying a log transform:

- Compresses large values  
- Expands smaller values  
- Reduces skewness  
- Produces a distribution closer to Gaussian  

Since Gaussian Processes assume approximately Gaussian noise, modelling log(y) instead of y results in:

- Smoother likelihood surface  
- Improved conditioning  
- More stable hyperparameter optimisation  


## ARD Gaussian Process with Multi-Start L-BFGS-B

An ARD Matern kernel assigns a separate length scale to each input dimension.

Design choices:

- x3 receives tighter bounds encouraging shorter length scales  
- Other dimensions receive wider bounds  

This reflects the assumption that x3 is dominant.

Hyperparameter optimisation:

- Multi-start L-BFGS-B  
- Exact gradients  
- 25 restarts to avoid poor local optima  

This is efficient when the log-marginal likelihood is smooth and well-conditioned.


## Expected Improvement Concentrated Near x3 Boundary

Expected Improvement (EI) measures the expected gain over the current best value.

Candidate generation:

- 6000 Sobol points in strip: x3 ∈ [0.90, 1.0]  
- 2000 Sobol points globally in [0,1]^5  

Rationale:

- Concentrates search near region most likely to produce improvement  
- Maintains global candidates as insurance  

EI is computed in log space relative to the best observed log value.


## Why This Strategy Is Ideal for f5

Properties:

- 5-dimensional input space  
- Large output dynamic range  
- Strong monotonic dependence on x3  

Implications:

- Log transform stabilises modelling  
- ARD allows focus on dominant dimension  
- Candidate concentration improves efficiency  
- Global pool prevents total bias  

This is more efficient than uniform exploration under expensive evaluation constraints.


## Tech Stack

- numpy  
- scipy (Sobol generator and optimisation backend)  
- scikit-learn (GaussianProcessRegressor and kernels)  


## Hyperparameters and Recommended Settings

### Candidate Generation

- Strip candidates: 6000  
- Global candidates: 2000  
- x3 strip lower bound: 0.90  
- Minimum distance guard: 0.04  


### Acquisition

- Expected Improvement xi: 0.001  

Small xi focuses on improvement over current best rather than broad exploration.


### Hyperparameter Optimisation

- L-BFGS-B restarts: 25  


### Kernel Bounds

- x3 length scale bounds: [0.01, 3.0]  
- Other dimensions length scale bounds: [0.05, 15.0]  
- Noise bounds: [1e-8, 0.01]  

These reflect:

- Tight control on dominant dimension  
- Flexibility for weaker dimensions  
- Near-deterministic behaviour after log transform  


## Hyperparameter Tuning Method

Multi-start L-BFGS-B maximises the GP log marginal likelihood in log-output space.

Advantages:

- Uses exact gradients  
- Efficient under smooth likelihood  
- Multiple restarts reduce local optimum risk  

This is appropriate because the log transform improves conditioning.


## Full Strategy Workflow

1. Collect existing data: `X_train_5`, `y_train_5`.  
2. Apply log transform to outputs.  
3. Define asymmetric length scale bounds (tight for x3).  
4. Fit ARD Matern GP using multi-start L-BFGS-B.  
5. Generate 6000 Sobol candidates with x3 ∈ [0.90, 1.0].  
6. Generate 2000 global Sobol candidates in [0,1]^5.  
7. Combine candidate pools.  
8. Apply minimum distance guard.  
9. Predict log-mean and log-standard deviation.  
10. Compute Expected Improvement in log space.  
11. Select candidate with highest EI.  
12. Evaluate black-box function and update dataset.  


## Hypothesis Framework

### Core Assumptions

- Log transform stabilises output distribution.  
- x3 dominates output behaviour.  
- Log-space likelihood is smooth.  

### If Assumptions Hold

- Stable GP hyperparameters  
- Shorter length scale for x3  
- EI focuses on high-value strip candidates  
- Efficient discovery of improved values  

### If Assumptions Break

- Log transform fails to stabilise distribution  
- x3 is not dominant  
- Important regions lie outside strip  
- Likelihood remains multimodal  


## Critical Analysis

### Strengths

- Handles heavy-tailed outputs  
- Focused search near dominant boundary  
- ARD captures anisotropic smoothness  
- Uses principled EI acquisition  

### Weaknesses

- Strong reliance on x3 dominance assumption  
- Strip concentration may under-explore other regions  
- Gradient-based optimisation may struggle if likelihood is irregular  


## Failure Modes

- Dominant dimension misidentified  
- True optimum lies away from x3 boundary  
- Output remains non-Gaussian after log transform  
- Likelihood highly multimodal despite restarts  


## Summary

This strategy combines:

- Log-transformed ARD Gaussian Process  
- Multi-start L-BFGS-B hyperparameter tuning  
- Expected Improvement acquisition  
- Concentrated candidate generation near x3 upper boundary  

It is designed for e
## f6 — RBF ARD Gaussian Process with Portfolio UCB and Copula EI (5D)

## Objective of Submission

The objective is to propose the next query point for an expensive 5-dimensional black-box function using `x_train_6` and `y_train_6`.  

The strategy uses a Gaussian Process with an RBF kernel and a portfolio of acquisition functions to aggressively explore promising regions while protecting against model uncertainty.


## Key Assumptions

1. Only a subset of input dimensions are truly important; the remaining dimensions are mostly inactive.
2. The response surface is smooth enough that an RBF kernel can capture the structure.
3. A high exploration parameter in Upper Confidence Bound (UCB) is necessary to push the search into corners far from existing data.

## Research Backing

The UCB rule with a theoretically motivated beta schedule is derived from regret bounds for Gaussian Process bandits. Asymmetric length scales across dimensions are supported by active subspace and sparsity theory. Acquisition portfolios that combine multiple criteria have been shown to outperform single strategies on difficult black-box tasks.

### Supporting Academic Work

- Srinivas et al., *Gaussian Process Optimization in the Bandit Setting*, ICML 2010  
- Bull, *Convergence Rates of Efficient Global Optimization Algorithms*, JRSS-B 2011  
- Lindauer et al. (Squirrel Team), NeurIPS 2020 Black-Box Optimization Challenge Report  
- Salinas et al., *Quantile Regression for Bayesian Optimization with Copulas*, ICML 2020  


## Explorative Principle and Function-Specific Rationale

The Gaussian Process models the function as a smooth surface over the 5D input space.

An ARD RBF kernel assigns a separate length scale to each dimension:

- Active dimensions → short length scales  
- Inactive dimensions → long length scales  

This allows the model to preserve high uncertainty along important directions while flattening irrelevant ones.

Exploration is driven by three acquisition components:

1. **Primary UCB** with moderate beta to balance mean and uncertainty.
2. **Secondary UCB** with higher beta to aggressively explore high-uncertainty regions, especially near corners.
3. **Copula-based Expected Improvement (EI)** using a rank-transformed version of `y` to stabilise heavy-tailed or skewed responses.

Each acquisition score is normalised to `[0, 1]` and combined into a single weighted portfolio score. The selected point performs well across all three criteria.


## Black-Box Optimization Competition Context

Similar portfolio-based Gaussian Process strategies were used in the NeurIPS 2020 Black-Box Optimization Challenge.

Top-performing teams, including the Squirrel Team, combined UCB-style exploration with EI-like criteria and careful kernel design to achieve strong performance in high-dimensional settings.


## Why This Strategy Is Appropriate for `x_train_6` and `y_train_6`

We assume:

- Some dimensions are clearly active.
- Others contribute mostly noise.

A single isotropic length scale would blur this structure and incorrectly reduce uncertainty in important corners.

By using:

- Short length scales for active dimensions
- Long length scales for inactive ones

the model maintains high uncertainty in meaningful directions. The high-beta UCB component then drives exploration into these regions. The copula EI component adds robustness when the raw `y` distribution is skewed or heavy-tailed.


## Justification for Expensive Evaluations

Each function evaluation is costly.

The portfolio strategy:

- Avoids oversampling well-understood regions
- Targets points that are both uncertain and likely to improve the best observed value
- Increases information gain per evaluation

This maximises return on a limited evaluation budget.


## Tech Stack

- `numpy` for numerical arrays  
- `scipy` for Sobol sequences and normal distribution functions  
- `scikit-learn` for Gaussian Process regression and RBF kernels  


## Hyperparameters and Settings

### Core Hyperparameters

- Active and inactive dimension indices  
- Initial length scales for active and inactive dimensions  
- Length scale bounds for active and inactive dimensions  
- Noise level and bounds  
- Primary and secondary beta values for UCB  
- Portfolio weights  
- Expected Improvement `xi`  
- Number of GP restarts  
- Number of Sobol candidates  
- Minimum distance guard  


### Recommended Initial Values

**Active dimensions**
- Initial length scale: `0.15`
- Bounds: `(0.03, 0.45)`

**Inactive dimensions**
- Initial length scale: `3.0`
- Bounds: `(0.8, 20.0)`

**Noise level**
- Initial: `1e-4`
- Bounds: `(1e-8, 0.005)`

**UCB parameters**
- `beta_primary = 5.0`
- `beta_secondary = 8.0`

**Portfolio weights**
- 0.5 → UCB (beta 5)
- 0.3 → UCB (beta 8)
- 0.2 → Copula EI

**EI parameter**
- `xi = 0.001`

**Optimization settings**
- `n_restarts = 20`
- `n_sobol = 20000`
- `min_dist = 0.04`


## Hyperparameter Tuning Method

Gaussian Process hyperparameters are tuned by maximising the log marginal likelihood using L-BFGS-B with multiple random restarts. This reduces sensitivity to local optima and is standard in GP literature.


## Entire Strategy Flow

1. Fit ARD RBF Gaussian Process to `x_train_6`, `y_train_6`.
2. Generate a Sobol candidate pool in the 5D unit cube.
3. Filter candidates using a minimum distance constraint.
4. Compute GP mean and standard deviation on raw `y`.
5. Transform `y_train_6` via rank-to-normal (copula transform).
6. Fit second GP on transformed targets.
7. Compute:
   - UCB (beta_primary)
   - UCB (beta_secondary)
   - Copula-based EI
8. Normalise all acquisition scores to `[0, 1]`.
9. Combine via weighted portfolio.
10. Select candidate with highest combined score.


## Hypothesis Framework

### Core Assumptions

- Smooth function
- Sparse active subspace
- Need for strong exploration in corners

### If Assumptions Hold

- GP uncertainty is well-calibrated
- Portfolio drives search into promising active subspace corners
- Best observed value improves progressively

### If Assumptions Break

- RBF may misrepresent discontinuities
- Incorrect active dimension identification harms performance
- Heteroscedastic noise may distort uncertainty
- Model may become overconfident


## Critical Analysis

### Strengths

- Encodes structural prior via asymmetric length scales
- Reduces dependence on a single acquisition heuristic
- Strong exploration pressure through high-beta UCB
- Robustness to skewed response via copula EI

### Weaknesses

- Requires prior analysis to identify active dimensions
- Portfolio weights are heuristic
- Computationally expensive (GP restarts + large candidate pool)

### Failure Modes

- Sharp discontinuities
- Changing active subspace across domain
- Too few observations to estimate ARD reliably
- Candidate pool misses true optimum

---

## f7 — Four-Model CV-Weighted Ensemble with Soft Directional Sobol UCB (7D)

## Objective of Submission

The objective is to propose the next query point for an expensive 7-dimensional black-box function using `x_train_7` and `y_train_7`.

The strategy uses a four-model ensemble:

- Gaussian Process (GP)
- Support Vector Regression (SVR)
- k-Nearest Neighbors (kNN)
- Gradient Boosted Trees (XGBoost) or Random Forest (fallback)

Model predictions are combined using leave-one-out cross-validation (LOOCV) weighted averaging. Exploration is driven by a Gaussian Process Upper Confidence Bound (UCB) acquisition, modulated by soft directional weighting and evaluated over a Sobol candidate pool.


## Key Assumptions

1. The response surface is moderately smooth but not fully captured by a single model class; a diverse ensemble is more reliable.
2. Some dimensions exhibit directional effects that can be encoded as soft preferences rather than hard constraints.
3. Leave-one-out cross-validation mean squared error (LOOCV MSE) is a reasonable proxy for model reliability at small sample sizes.

## Research Backing

- Ensemble methods reduce variance and improve robustness on noisy or diffuse surfaces.
- Tree-based models capture nonlinear interactions that kernel methods may miss.
- SVR with RBF kernel and epsilon-insensitive loss is robust to outliers.
- Cross-validation-based weighting provides a principled way to combine models when information criteria are unstable at small sample sizes.

### Supporting Literature

- Breiman (1996), *Bagging Predictors*
- Friedman (2001), *Greedy Function Approximation*
- Cortes & Vapnik (1995), *Support Vector Networks*
- De Groot & Schuurmans (2004), *Model Selection via Cross Validation*


## Explorative Principle and Function-Specific Rationale

The ensemble combines four complementary inductive biases:

- **Gaussian Process**: Smooth probabilistic model with uncertainty estimates.
- **SVR (RBF)**: Robust nonlinear regression resistant to outliers.
- **kNN**: Captures very local structure.
- **XGBoost / Random Forest**: Captures nonlinear interactions and piecewise behavior.

Each model is evaluated via LOOCV MSE. Weights are assigned as:

weight_i ∝ 1 / MSE_i  
(normalized to sum to 1)

The ensemble mean prediction is:

μ_ensemble(x) = Σ w_i μ_i(x)

Exploration is driven by:

UCB(x) = μ_ensemble(x) + β · σ_GP(x)

where σ_GP is the Gaussian Process standard deviation (inflated).

This UCB score is modulated by **soft directional weights**, increasing the score for desirable trends (e.g., high x6, low x1, low x5) without excluding other regions.

The acquisition is evaluated over a Sobol candidate pool to ensure space-filling coverage.


## Competition Context

Similar ensemble and directional acquisition strategies have been used in NeurIPS black-box optimization challenges, where teams combined Gaussian Processes, tree-based models, and kernel methods with custom acquisition functions.

Top-performing teams demonstrated that ensemble surrogates with tailored acquisition functions can outperform pure Gaussian Process methods in high-dimensional noisy settings.


## Why This Strategy Is Appropriate for `x_train_7` and `y_train_7`

We assume a seven-dimensional function with mixed structure:

- Some dimensions may have monotonic or directional effects.
- Others may interact in complex nonlinear ways.

A single Gaussian Process may underfit tree-like structure.  
A single tree model may be too rough.  

The four-model ensemble leverages complementary strengths.

Soft directional weighting encodes prior knowledge while preserving exploration, which is critical when evaluations are expensive.


## Justification for Expensive Evaluations

Each evaluation is costly.

The strategy ensures that new points are:

- Informative (via GP uncertainty)
- Likely to improve the objective (via ensemble mean)
- Biased toward historically strong directions (via soft weighting)

This increases the expected value of each evaluation under limited budgets.


## Tech Stack

- `numpy` — vectorised operations
- `scipy.stats.qmc` — Sobol sequence generation
- `scikit-learn` — GP, SVR, kNN, Random Forest, LOOCV, MSE
- `xgboost` — Gradient Boosted Trees (optional)


## Hyperparameters and Settings

### Gaussian Process

- Kernel: Matérn 2.5 + White noise
- Initial length scales: 1.0 (each dimension)
- Length scale bounds: (0.05, 10.0)
- Noise initial: 1e-3
- Noise bounds: (1e-6, 1.0)
- Restarts: 10
- Variance inflation factor: 3.0

### SVR

- C: 10.0
- epsilon: 0.1
- Kernel: RBF
- gamma: "scale"

### kNN

- k: 5
- Weighting: distance-based

### XGBoost (preferred)

- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1

### Random Forest (fallback)

- n_estimators: 100
- max_depth: 5

### Acquisition

- beta: 2.0
- Sobol candidates: 8000
- Minimum distance guard: 0.05

### Directional Coefficients (example)

- +0.5 for positive driver dimension
- −0.3 for negative driver dimensions


## Hyperparameter Tuning

- GP hyperparameters: optimized via log marginal likelihood using L-BFGS-B with multiple restarts.
- Other models: fixed hyperparameters based on literature and prior validation.
- Ensemble weights: computed dynamically via LOOCV MSE.


## Entire Strategy Flow

1. Fit GP, SVR, kNN, and XGBoost/Random Forest to `x_train_7`, `y_train_7`.
2. Compute LOOCV MSE for each model.
3. Convert MSE to normalized inverse-error weights.
4. Generate Sobol candidate pool in 7D unit cube.
5. Apply minimum distance guard.
6. Compute predictions from each model at all candidates.
7. Form ensemble mean as weighted sum.
8. Compute GP standard deviation and inflate by variance factor.
9. Compute soft directional multiplier for each candidate.
10. Compute acquisition:

   score(x) = directional_weight(x) · [ μ_ensemble(x) + β · σ_GP(x) ]

11. Select candidate with highest score.


## Hypothesis Framework

### Core Assumptions

- Models capture complementary structure.
- LOOCV MSE reflects reliability.
- Directional effects are stable across the domain.

### If Assumptions Hold

- Ensemble mean outperforms any single model.
- UCB identifies promising and uncertain regions.
- Directional weighting accelerates improvement.

### If Assumptions Break

- Directional priors may mislead search.
- LOOCV may not fully detect overfitting.
- GP uncertainty may be miscalibrated.
- Ensemble may share common bias.


## Critical Analysis

### Strengths

- Reduces risk of single-model misspecification.
- Data-adaptive model weighting.
- Preserves exploration while incorporating prior structure.
- Explicit uncertainty-driven exploration term.

### Weaknesses

- Computationally heavier than single-model methods.
- Requires prior analysis for directional coefficients.
- Ensemble may still inherit shared bias.

### Failure Modes

- Extremely noisy or discontinuous functions.
- Too few observations to fit four models reliably.
- Incorrect directional assumptions.
- Candidate pool fails to cover optimal regions.

---

## f8 — Heteroscedastic ARD GP + kNN + Extra Trees + Gradient Boosting with Micro-HPO and High-Beta UCB (8D)

## Objective of Submission

To propose the next query point for `f8` by aggressively exploring the 8-dimensional input space using a four-model ensemble:

- ARD Matérn 2.5 Gaussian Process (heteroscedastic-friendly)
- k-Nearest Neighbors
- Extra Trees
- Gradient Boosting

The ensemble is tuned via micro hyperparameter optimization (micro-HPO), and exploration is driven by a high-beta Upper Confidence Bound (UCB) acquisition function.


## Key Assumptions

1. The function is moderately smooth in an active subspace, so an ARD Matérn 2.5 Gaussian Process can model global trends and provide meaningful uncertainty estimates.

2. The function also contains non-smooth interactions or sharp transitions that tree ensembles (Extra Trees and Gradient Boosting) capture better than GP or kNN.

3. Micro hyperparameter optimization over small, controlled grids improves model calibration without overfitting at approximately 46 observations.


## Research Backing

### Supporting Academic Work

- Srinivas et al. (2010), *Gaussian Process Optimization in the Bandit Setting*  
  Justifies high-beta GP-UCB for sublinear regret and principled exploration.

- Cowen-Rivers et al. (2022), *HEBO*  
  Demonstrates robustness of ARD and heteroscedastic GP models under heterogeneous variance and outliers.

- Breiman (2001), *Random Forests*  
- Geurts et al. (2006), *Extremely Randomized Trees*  
  Show tree ensembles capture nonlinear interactions and non-smooth structure.

- Friedman (2001), *Greedy Function Approximation*  
  Introduces gradient boosting for structured nonlinear modeling.

- De Groot & Schuurmans (2004), *Model Selection via Cross Validation*  
  Supports cross-validation-based model selection and weighting in small data regimes.


## Explorative Principle and Function-Specific Rationale

The acquisition principle is Upper Confidence Bound:

UCB(x) = μ_ensemble(x) + β · σ_GP_inflated(x)

Where:

- μ_ensemble(x) = weighted average of predictions from GP, kNN, Extra Trees, and Gradient Boosting
- σ_GP_inflated(x) = GP predictive standard deviation scaled by a variance inflation factor
- β = high exploration parameter

### Why This Is Rational for f8

For an 8D function with:

- ~3 strong active dimensions
- Heterogeneous variance
- Occasional outliers

The model roles are:

- **GP (ARD Matérn 2.5)**: captures smooth global structure and provides uncertainty.
- **kNN**: captures local neighborhoods.
- **Extra Trees**: models non-smooth interactions and sharp changes.
- **Gradient Boosting**: models structured nonlinear trends.

Micro-HPO ensures each model is reasonably calibrated without overfitting.  
High-beta UCB forces global exploration in under-sampled regions.


## Competition Context

NeurIPS Black-Box Optimization challenges and related benchmarks frequently use:

- GP-UCB
- Heteroscedastic or ARD Gaussian Processes
- Tree ensembles
- Model ensembles

HEBO-style GP models combined with tree surrogates have been used by top-performing teams to improve robustness and exploration efficiency.


## Why This Strategy Is Appropriate for f8

Each evaluation of `f8` is expensive.

Key structural characteristics:

- Strong active dimensions
- Heterogeneous variance
- Outliers
- Mixed smooth and non-smooth behavior

An ARD GP with flexible noise can:

- Focus on important dimensions
- Absorb outliers
- Provide calibrated uncertainty

Tree ensembles capture interaction-heavy and non-smooth regions.  
kNN stabilizes predictions near observed data.

High-beta UCB ensures under-sampled regions are explored aggressively, which is critical in 8D with limited budget.


## Tech Stack

- `numpy` — numerical operations
- `scipy.stats.qmc` — Sobol sequence generation
- `scikit-learn`:
  - `GaussianProcessRegressor` (Matérn + WhiteKernel)
  - `KNeighborsRegressor`
  - `ExtraTreesRegressor`
  - `GradientBoostingRegressor`
  - `KFold` / `LeaveOneOut` for CV


## Hyperparameters and Settings

### Gaussian Process

- Kernel: Matérn (ν = 2.5) with ARD
- Active dimension bounds: [0.05, 3.0]
- Inactive dimension bounds: [0.1, 20.0]
- Noise initial: 0.1
- Noise bounds: [1e-4, 1.0]
- n_restarts: 20–25
- Variance inflation factor: 3.0

### kNN (micro-HPO grid)

- k ∈ {3, 5, 7}

### Extra Trees (micro-HPO grid)

- n_estimators ∈ {200, 300}
- max_depth ∈ {3, 4, 5}
- max_features ∈ {"auto", 0.7}

### Gradient Boosting (micro-HPO grid)

- n_estimators ∈ {100, 150}
- max_depth ∈ {2, 3}
- learning_rate ∈ {0.05, 0.1}

### Acquisition

- β = 5.0
- n_sobol = 15000
- min_dist_guard = 0.04

### Ensemble Weights

- Derived from inverse cross-validation MSE
- No manual tuning


## Hyperparameter Tuning Method (Micro-HPO)

1. Define small discrete grids for kNN, Extra Trees, and Gradient Boosting.
2. For each configuration:
   - Evaluate using K-fold CV MSE (e.g., 5-fold).
3. Select best configuration per model.
4. Refit selected configuration on full data.
5. Compute final CV MSE.
6. Convert inverse MSE to normalized ensemble weights.

This keeps tuning safe and computationally manageable at n ≈ 46.


## Entire Strategy Flow

1. Start with `x_train_8`, `y_train_8`.
2. Run micro-HPO for kNN, Extra Trees, and Gradient Boosting.
3. Fit ARD Matérn GP with wide noise bounds.
4. Fit tuned kNN, Extra Trees, and Gradient Boosting.
5. Compute CV MSE for all models.
6. Convert inverse MSE to ensemble weights.
7. Generate Sobol candidate set in 8D unit cube.
8. Apply minimum distance guard.
9. For each candidate:
   - Compute GP mean and sigma
   - Compute kNN, Extra Trees, GB predictions
   - Form ensemble mean
10. Inflate GP sigma.
11. Compute:

    UCB(x) = μ_ensemble(x) + β · σ_GP_inflated(x)

12. Select candidate with highest UCB.
13. Print diagnostics:
    - Coordinates
    - Distance to current best
    - Ensemble mean
    - GP uncertainty
    - Final UCB score


## Hypothesis Framework

### Core Assumptions

- The four models capture complementary structure.
- Micro-HPO improves calibration without overfitting.
- High-beta UCB efficiently explores 8D space.

### If Assumptions Hold

- Ensemble predictions are robust and accurate.
- Exploration focuses on promising but uncertain regions.
- Best observed value improves steadily.
- Acquisition remains stable across iterations.

### If Assumptions Break

- Miscalibrated GP uncertainty may misguide UCB.
- Tree models may overfit despite micro-HPO.
- Extremely discontinuous or noise-dominated functions may defeat all surrogates.
- Exploration may become inefficient.


## Critical Analysis

### Strengths

- Combines complementary modeling paradigms.
- Micro-HPO safely improves model quality in small-data regime.
- High-beta UCB aggressively explores under-sampled regions.
- ARD + wide noise bounds adapt to active dimensions and heteroscedasticity.
- No hard exclusion zones; allows revisiting promising areas.

### Weaknesses

- Higher computational and implementation complexity.
- Cross-validation overhead increases with n.
- Ensemble behavior is less interpretable than a single surrogate.

### Failure Modes

- Extremely noisy or discontinuous functions.
- Very narrow optimum region rarely sampled even with large candidate sets.
- Too few observations to reliably support four models and tuning.
moderately smooth, near-deterministic high-dimensional optimisation where local refinement has stagnated and global uncertainty must be leveraged efficiently.
