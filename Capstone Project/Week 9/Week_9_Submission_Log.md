## Function-Specific Strategies


## f1 — TuRBO-1 Local Bayesian Optimisation with Copula-Warped Gaussian Process

## Objective of Submission

To propose the next query point for Function 1 using a **local Bayesian optimisation** strategy that concentrates all evaluations inside a shrinking trust region around the current best point.

The surrogate model is stabilised using a **Gaussian copula transform** so that extremely small outputs become numerically learnable by a Gaussian Process.


## Key Assumptions

1. Function 1 has a single dominant local peak near the current best observation, and refining this peak is more valuable than restarting global exploration.

2. Raw outputs are numerically pathological, but their **rank order** preserves the underlying structure. A rank-based copula transform can therefore recover a smooth signal.

3. A Matérn Gaussian Process with ARD length scales can approximate the local shape of the peak once outputs are properly warped.


## Research Backing

### Supporting Academic Work

- Eriksson et al. (2019), *Scalable Global Optimisation via Local Bayesian Optimisation (TuRBO)*  
  Introduces trust-region Bayesian optimisation with adaptive shrinking and expansion.

- Salinas et al. (2020), *Copula-Based Bayesian Optimisation*  
  Shows that Gaussian copula warping stabilises GPs on heavy-tailed or badly scaled objectives.

- Jones, Schonlau, Welch (1998), *Efficient Global Optimisation*  
  Introduces Expected Improvement as a principled acquisition function.


## Explorative Principle and Function-Specific Rationale

The exploration principle is **local Expected Improvement (EI)** inside a trust region.

Instead of exploring the full domain `[0,1]^2`, we:

- Restrict all candidate points to a small box centered on the current best point.
- Model copula-warped outputs with a Gaussian Process.
- Compute EI with `xi = 0`.

EI(x) rewards only candidates predicted to exceed the current best in transformed space.

### Why This Is Rational for f1

- Observed outputs range roughly from `1e-248` to `5e-3`.
- Best values are around `5e-15`.
- Global exploration repeatedly evaluates regions effectively equal to zero.
- The two best points are very close in input space, suggesting a narrow local peak.

Therefore:

- Concentrating the budget inside a trust region is rational.
- Copula warping removes extreme scale issues.
- Pure improvement-driven EI is appropriate when the peak is already bracketed.


## Competition Context

TuRBO-style local Bayesian optimisation was widely used in the NeurIPS 2020 Black-Box Optimisation Challenge.

High-ranking teams combined global search with local trust-region refinement to efficiently optimise expensive black-box functions.


## Why This Strategy Is Ideal for f1

Function 1 exhibits:

- Extreme output scale pathologies
- A narrow, high-value peak
- Strong evidence that the optimum lies near current best

This strategy:

- Avoids wasting evaluations on globally poor regions
- Focuses exclusively on resolving the local peak
- Stabilises GP modelling through copula transformation
- Uses EI to directly target improvement

When evaluations are expensive, such concentrated refinement is statistically efficient.

## Tech Stack

- `numpy` — numerical operations and sampling
- `scipy.stats` — normal CDF and inverse CDF
- `scikit-learn`:
  - `GaussianProcessRegressor`
  - `Matern` + `WhiteKernel`
  - `StandardScaler`


## Hyperparameters and Settings

### Trust Region

- Centre: current best `x`
- Length: 0.10
- Expand factor: 1.5
- Shrink factor: 0.75
- min_length: 0.005
- max_length: 0.30

### Candidate Generation

- n_candidates: 5000
- min_dist_guard: 0.02

### Gaussian Process

- Kernel: Matérn ν = 2.5 (ARD)
- Length scale bounds: (0.001, 2.0)
- White noise bounds: (1e-10, 1.0)
- n_restarts_optimizer: 25

### Acquisition

- Expected Improvement
- xi = 0.0


## Hyperparameter Rationale

- Short lower length-scale bound (0.001) allows modeling a very narrow spike.
- Wide noise bounds prevent forced interpolation and allow residual absorption.
- Multiple optimizer restarts reduce risk of poor marginal likelihood solutions.
- xi = 0 enforces pure exploitation inside a region already encoding locality.
- Trust region length 0.10 captures best observations while excluding most of the domain.

These settings follow standard TuRBO and copula-EI practices.


## Entire Strategy Flow

1. Start with `x_train_1`, `y_train_1`.
2. Identify current best observation.
3. Apply Gaussian copula transform:
   - Rank-transform outputs
   - Map ranks to standard normal
4. Standardise transformed outputs (zero mean, unit variance).
5. Fit Matérn GP with white noise on transformed targets.
6. Define trust region centered on best `x`, clipped to `[0,1]^2`.
7. Generate uniform random candidates within trust region.
8. Apply minimum distance guard.
9. Compute Expected Improvement (xi = 0) at each candidate.
10. Select candidate with highest EI.
11. After evaluation:
    - Expand trust region if improvement occurs.
    - Shrink trust region otherwise.


## Hypothesis Framework

### Core Assumptions

- The optimum lies near the current best.
- Rank order preserves meaningful signal.
- Local GP is sufficient within trust region.

### If Assumptions Hold

- Points remain near true peak.
- Best value steadily improves.
- Trust region stabilises at appropriate scale.

### If Assumptions Break

- Local search may converge to suboptimal peak.
- Copula transform may fail if ranks are noisy.
- Extremely narrow spike may still be smoothed by GP.
- No mechanism exists for global restart.


## Critical Analysis

### Strengths

- Highly sample-efficient when optimum is locally bracketed.
- Copula transform resolves extreme scaling issues.
- EI directly optimises improvement.
- Adaptive trust region balances refinement and robustness.

### Weaknesses

- Purely local; cannot discover distant superior optima.
- GP may still oversmooth very sharp spikes.
- Trust-region hyperparameters are fixed and not meta-optimised.

### Failure Modes

- Initial best far from global optimum.
- Multiple sharp peaks inside trust region.
- Too few observations for reliable GP fitting.

---

## f2 — ARD-GP UCB with Tight Active-Region Candidate Concentration


## Objective of Submission

Propose the next query point for Function 2 using an ARD Gaussian Process with Upper Confidence Bound (UCB) acquisition and a soft Gaussian spatial weighting that concentrates candidates in the empirically active region while still allowing aggressive exploration of unmapped parts of the space.


## 3 Key Assumptions

1. Function 2 has a structured landscape where `x1` is the dominant driver and high values cluster around `x1 ≈ 0.7`.

2. An RBF + WhiteKernel Gaussian Process with separate length scales for `x1` and `x2` can provide reasonably calibrated mean and uncertainty estimates.

3. The most promising unexplored area lies in the `x2` gap between `0.55` and `0.80` within the active `x1` band around `0.7`.


## Research Backing

### Academic Papers Supporting the Strategy

- **Srinivas et al., 2010 — Gaussian Process Optimization in the Bandit Setting (ICML)**  
  Introduces GP-UCB and proves regret bounds when the surrogate is well calibrated, showing that UCB with an appropriate β systematically explores high-uncertainty regions.

- **Internal Week 5 Review of f2**  
  The RBF + WhiteKernel GP with UCB and β = 2.0 was the only configuration that clearly improved over baseline, making it the empirically validated surrogate for this function.

- **Bergstra et al., 2011 — Algorithms for Hyper-Parameter Optimization (NeurIPS)**  
  Supports focusing search in promising regions while still allowing exploration via probabilistic weighting rather than hard constraints.


## Explorative Principle

The explorative principle is GP-UCB combined with a soft spatial prior.

The Gaussian Process provides:

- Posterior mean `μ(x)`
- Posterior standard deviation `σ(x)`

UCB is computed as:

UCB(x) = μ(x) + β σ(x)


Points with both high predicted value and high uncertainty are preferred.

To encode knowledge that the best cluster lies near `x1 ≈ 0.7` and `x2 ≈ 0.75`, a Gaussian weight centered on this active region is multiplied with the normalized UCB score.

The weight:

- Is high near the cluster
- Decays smoothly away
- Encourages aggressive exploration within and around the active region
- Still allows movement outward if UCB is sufficiently large

## Black Box Optimization Competition

### Competition Context

NeurIPS Bayesian Optimization and Bandit Benchmarks, where GP-UCB was evaluated as a principled exploration strategy.

### Contribution Context

GP-UCB style methods were introduced and evaluated by Srinivas et al., demonstrating strong theoretical and empirical performance on expensive black-box optimization tasks. Similar GP-based UCB strategies have been used by high-performing teams in black-box optimization challenges.


## Why This Strategy Is Ideal for Function 2

### Justification Based on Expensive Evaluations

- Earlier weeks already performed global exploration.
- MES-based pure exploration in Week 7 returned a low value (~0.006), indicating exploration alone is not the bottleneck.
- The top three observations form a clear cluster:
  - `x1 ∈ [0.67, 0.73]`
  - `x2 ≈ 0.56, 0.85, 0.93`

This suggests a structured active region rather than a flat landscape.

Since evaluations are expensive, uniform sampling is inefficient. Instead, candidate concentration near the active region while exploring the unmapped `x2` gap (`0.55–0.80`) is more rational.

GP-UCB with β = 2.0 already demonstrated strong behavior. Adding a soft Gaussian weight refines search direction without discarding theoretical exploration guarantees.


## Tech Stack

### Libraries and Frameworks

- `numpy` for numerical operations
- `scipy.stats.qmc` for Sobol quasi-random candidate generation
- `scipy.spatial.distance` for minimum-distance filtering
- `scikit-learn`:
  - `GaussianProcessRegressor`
  - `RBF` kernel
  - `WhiteKernel`


## Hyperparameters and Settings

### List of Hyperparameters

- RBF kernel length scales and bounds for `x1` and `x2`
- WhiteKernel noise level and bounds
- Number of GP optimizer restarts
- Sobol candidate count
- Minimum distance threshold
- Gaussian spatial weight center and standard deviations
- UCB β parameter


## Recommended Initial Values and Rationale

### Kernel Length Scales

length_scale = [0.5, 1.0]
bounds = [(0.05, 3.0), (0.10, 5.0)]
Shorter length scale on `x1` reflects its stronger, steeper effect. Longer on `x2` reflects smoother variation. Matches Week 5 configuration.

### WhiteKernel Noise Level

noise_level = 1e-3
bounds = (1e-6, 0.3)
Reduces local optima risk in marginal likelihood optimization.

### Sobol Candidates

8192 candidates (m = 13)
Dense 2D coverage while remaining computationally manageable.

### Minimum Distance Threshold

0.05
Prevents near-duplicate evaluations.

### Gaussian Spatial Weight

center = (0.699, 0.780)
sigma_x1 = 0.12
sigma_x2 = 0.15
Center is the centroid of top observations. Sigmas cover the active region and unmapped gap without being overly restrictive.

### UCB Parameter

β = 2.0
Empirically validated in Week 5.


## Hyperparameter Tuning Method

Hyperparameters are selected based on prior empirical success and standard GP-UCB practice.

Future iterations could adapt:

- β based on improvement frequency
- Gaussian sigmas based on candidate dispersion

For this submission, parameters are fixed for stability and reproducibility.


## Entire Flow of the Strategy

1. Start from `x_train_2` and `y_train_2`.
2. Fit GP with RBF + WhiteKernel (ARD enabled).
3. Generate Sobol candidates in `[0,1]^2`.
4. Apply minimum distance filtering (`> 0.05`).
5. Compute posterior `μ(x)` and `σ(x)`.
6. Compute UCB: `μ + βσ`.
7. Compute Gaussian spatial weight.
8. Normalize UCB to `[0,1]`.
9. Multiply normalized UCB by spatial weight.
10. Select highest combined score as next query point.
11. Optionally inspect top candidates and coverage of unmapped `x2` gap.


## Hypothesis Framework

### Core Assumptions

- The cluster near `x1 ≈ 0.7` is genuine.
- The GP provides calibrated uncertainty.
- The unmapped `x2` gap is promising.

### If Assumptions Hold

- Selected candidate lies in active region but unexplored `x2`.
- High probability of improvement.
- Future iterations remain focused but diverse.

### If Assumptions Break

- May converge to suboptimal local maximum.
- Miscalibrated GP leads to poor exploration balance.
- Unmapped gap may produce low values before correction.


## Critical Analysis

### Strengths

- Theoretically grounded acquisition (GP-UCB).
- Soft prior preserves flexibility.
- Minimum distance guard avoids redundancy.
- Conceptually simple and interpretable.

### Weaknesses

- Dependent on GP kernel adequacy.
- Spatial weight introduces bias.
- Fixed hyperparameters may not be stage-optimal.

### Failure Modes

- True global maximum lies far from active region.
- Strong nonstationarity not captured by RBF.
- Data too noisy for stable GP learning.

---

## f3 — Kernel Ridge Mean with Fixed-Hyperparameter GP Uncertainty


## Objective of Submission

Propose the next query point for Function 3 by aggressively exploring promising regions through a decoupled surrogate:

- Kernel Ridge Regression (KRR) with an RBF kernel for stable mean estimation  
- A fixed-hyperparameter Gaussian Process (GP) for calibrated uncertainty  
- Upper Confidence Bound (UCB) acquisition to combine mean and uncertainty  

The goal is to avoid unstable marginal likelihood optimisation while preserving principled exploration.


## 3 Key Assumptions

1. The mean shape of f3 can be captured by a smooth nonparametric model such as Kernel Ridge Regression with an RBF kernel, without requiring full Bayesian treatment.

2. Uncertainty around that mean can be reasonably modelled by a Gaussian Process whose hyperparameters are fixed from exploratory data analysis (EDA), avoiding instability at small sample sizes.

3. The global optimum lies within structurally plausible regions identified by EDA:
   - `x2` moderate to high  
   - `x3` low to moderate  
   Regions far from this structure are unlikely to be optimal.


## Research Backing

### Academic Support

- **Duxiaoman Team Report — NeurIPS 2020 Black Box Optimization Challenge**  
  Used nonparametric surrogates for the mean (e.g., gradient boosted trees) and a Gaussian Process on residuals for uncertainty, effectively decoupling mean and variance modelling.

- **Srinivas et al., 2010 — Gaussian Process Optimization in the Bandit Setting (ICML)**  
  Motivates UCB acquisition to balance exploitation and exploration.

- **Eriksson et al., 2019 — Scalable Global Optimization via Local Bayesian Optimization (NeurIPS)**  
  Supports focusing exploration within structurally promising regions while maintaining some global coverage.


## Explorative Principle

This strategy separates:

- Function value prediction (mean)
- Uncertainty estimation (variance)

### Mean Model — Kernel Ridge Regression

- RBF kernel
- Hyperparameters (`alpha`, `gamma`) tuned via exact Leave-One-Out Cross Validation (LOO-CV)
- Provides a smooth, stable surface without marginal likelihood optimisation

### Uncertainty Model — Fixed-Hyperparameter GP

- Matérn kernel + WhiteKernel
- Length scales and noise fixed from EDA
- Provides calibrated uncertainty `σ(x)` without unstable optimisation

### Acquisition Function

Upper Confidence Bound:
UCB(x) = mean_KRR(x) + β * sigma_GP(x)

This encourages sampling points that either:

- Have high predicted value  
- Have high uncertainty  

Exploration is concentrated in structurally plausible regions, particularly:

- Around known high-value basins  
- In low-`x3` regions  
- Within constrained Sobol exploration  

---

## Black Box Optimization Competition Context

### Competition

NeurIPS 2020 Black Box Optimization Challenge and related Bayesian optimization benchmarks.

### Inspiration

The Duxiaoman team used decoupled mean and uncertainty modelling, inspiring the use of:

- Kernel Ridge Regression for mean  
- Fixed GP for uncertainty  


## Why This Strategy Is Ideal for Function 3

### Justification Based on Expensive Evaluations

- Full GP hyperparameter optimisation previously produced pathological kernels that inverted feature importance (`x2` and `x3`), wasting evaluations.
- Kernel Ridge Regression with LOO-CV is numerically stable at small `n`.
- Fixing GP hyperparameters from EDA avoids unstable marginal likelihood landscapes.
- UCB ensures exploration remains aggressive where both mean and uncertainty are promising.

This improves efficiency when evaluations are expensive.


## Tech Stack

### Libraries Used

- `numpy` for numerical operations
- `scipy.stats.qmc` for Sobol sequence generation
- `scikit-learn`:
  - `KernelRidge`
  - `LeaveOneOut`
  - `mean_squared_error`
  - `GaussianProcessRegressor`
  - `Matern`
  - `WhiteKernel`


## Hyperparameters and Settings

### Kernel Ridge Regression

- `alpha ∈ [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]`
- `gamma ∈ [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]`

Rationale:
- Covers weak to strong regularisation.
- Spans wide to narrow RBF bandwidths.
- Selected via LOO-CV.


### Gaussian Process (Fixed)

- Length scales:
  - `x1 = 5.0` (irrelevant → long scale)
  - `x2 = 0.4`
  - `x3 = 0.35`
- Noise level: `0.012`
- Tight bounds around these values

Derived from EDA structure.


### Structural Candidate Regions

**Basin A**  
- `x2 ∈ [0.45, 0.75]`
- `x3 ∈ [0.05, 0.48]`

**Basin B**  
- `x2 ∈ [0.75, 0.98]`
- `x3 ∈ [0.30, 0.58]`

**Basin C**  
- `x2 ∈ [0.60, 0.85]`
- `x3 ∈ [0.02, 0.15]`


### Sobol Arm

- `2^10 = 1024` points
- Constrained to:
  - `x2 ≥ 0.35`
  - `x3 ≤ 0.60`

### Additional Parameters

- `delta_min = 0.05` (fallback `0.03`)
- `β = 2.5`

β selected for moderately strong exploration in small-data regimes.


## Hyperparameter Tuning Method

- KRR hyperparameters tuned via exact Leave-One-Out Cross Validation.
- GP hyperparameters fixed or tightly bounded around EDA-informed values.
- β selected based on GP-UCB theory and prior experience.

This stabilizes both mean and uncertainty estimation.


## Entire Flow of the Strategy

1. Start with `x_train_3` and `y_train_3`.
2. Perform LOO-CV grid search over `alpha` and `gamma`.
3. Fit final KRR model with optimal hyperparameters.
4. Define GP with fixed Matérn kernel and noise.
5. Fit GP to training data.
6. Generate structured candidates in Basins A, B, C.
7. Generate constrained Sobol candidates.
8. Combine candidate sets.
9. Apply minimum distance filter (`delta_min`).
10. Compute:
    - `mean_KRR(x)`
    - `sigma_GP(x)`
11. Compute:
    ```
    UCB(x) = mean + β * sigma
    ```
12. Select candidate with highest UCB.
13. Return candidate and diagnostics.


## Hypothesis Framework

### Core Assumptions

- KRR mean surface approximates true function locally.
- Fixed GP provides calibrated uncertainty.
- Structural constraints include all plausible high-value regions.

### If Assumptions Hold

- Selected point lies in a promising basin with remaining uncertainty.
- Best value improves or plateau structure is confirmed.
- Exploration remains focused and efficient.

### If Assumptions Break

- Poor mean approximation biases UCB.
- Miscalibrated sigma causes over/under exploration.
- True optimum lies outside constrained regions.


## Critical Analysis

### Strengths

- Decouples mean and uncertainty to avoid GP instability.
- Uses LOO-CV for robust hyperparameter tuning.
- Incorporates structural knowledge from EDA.
- Maintains Sobol arm for partial global coverage.

### Weaknesses

- Dependent on correctness of EDA-based constraints.
- KRR provides no intrinsic uncertainty.
- Requires manual basin design.

### Failure Modes

- Strong nonstationarity or discontinuities.
- True optimum excluded by structural constraints.
- Noise level misestimated, leading to miscalibrated UCB.

---

## f4 — Local 2D ARD-GP Expected Improvement with Fixed Inactive Dimensions


## Objective of Submission

Propose a single Week 9 query for Function 4 that aggressively explores the most promising local region by:

- Fitting a Gaussian Process only in the active subspace
- Fixing inactive dimensions at their empirically optimal values
- Maximising Expected Improvement (EI)
- Respecting a strict evaluation budget

The goal is to concentrate all modelling capacity and search effort where improvement is realistically achievable.


## 3 Key Assumptions

1. `x1` and `x3` are locally inactive around the current best and can be safely fixed at their best observed values.

2. `x2` and `x4` form the active subspace controlling most variation in f4 near the optimum.

3. The true optimum lies within a local trust region around the current best, not in the global extremes.


## Research Backing

This strategy combines:

- Local Bayesian optimisation (trust-region modelling)
- Expected Improvement (EI)
- Active subspace / ARD dimensionality reduction

### Key References

- **Eriksson et al. (2019)** — *Scalable Global Optimization via Local Bayesian Optimization (TuRBO)*, NeurIPS  
- **Jones, Schonlau, Welch (1998)** — *Efficient Global Optimization*, Journal of Global Optimization  
- **Eriksson et al. (2021)** — *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO)*  
- **Bull (2011)** — *Convergence Rates of Efficient Global Optimization Algorithms*, JRSS-B  

These works support local trust regions, EI-based acquisition, and dimensionality reduction in expensive black-box settings.


## Explorative Principle

For f4:

- Most of the domain is extremely negative.
- Good values lie on a small central plateau.
- The current best is already on that plateau.

Global exploration is inefficient. Exploration must be:

- Local
- Directional
- Dimension-aware

### Strategy

1. Restrict modelling to top local observations.
2. Use ARD diagnostics to identify active dimensions.
3. Fix inactive dimensions (`x1`, `x3`) at best values.
4. Explore only the `x2–x4` plane.
5. Apply Expected Improvement within a trust region.

Expected Improvement is high only when:

- The predicted mean is competitive.
- The uncertainty is large enough to beat the current best.

This ensures focused but aggressive exploration.


## Black Box Optimization Competition Context

### Competition

NeurIPS 2020 Black Box Optimization Challenge and related benchmarks.

### Team Strategies

Top teams (e.g., Duxiaoman, NVIDIA RAPIDS, Optuna developers) used:

- Local GP models
- Trust regions
- Dimensionality reduction
- Structured search spaces


## Why This Strategy Is Ideal for f4

### Structural Observations

- Central plateau with moderate to positive values.
- Steep negative walls outside.
- Current best already lies on the plateau.
- ARD shows `x1` and `x3` are flat locally.
- `x2` and `x4` drive variation.

Thus:

- Fix `x1`, `x3`.
- Explore only `x2`, `x4`.
- Use EI to avoid chasing pure uncertainty.

This is precisely the scenario TuRBO-style methods target.


## Justification Under Expensive Evaluations

To minimise wasted evaluations:

- Fit GP only on top local observations.
- Restrict search to 2D active subspace.
- Enforce trust region.
- Apply minimum distance guard.
- Use EI (near-zero where improvement unlikely).

Each evaluation becomes maximally informative.


## Tech Stack

- `numpy`
- `scipy` (Sobol, L-BFGS-B)
- `scipy.spatial`
- `scikit-learn`
  - `GaussianProcessRegressor`
  - `Matern`
  - `WhiteKernel`


## Hyperparameters and Settings

### Core Hyperparameters

- `n_local`
- Trust region half-width
- Kernel type and ν
- Length scale bounds (2D)
- Noise level and bounds
- Sobol candidate count
- Minimum distance threshold
- L-BFGS-B restarts
- EI jitter parameter `xi`


## Recommended Initial Values

### Local Dataset

n_local = 12

Enough for stable GP, focused on plateau.

### Trust Region

half_width = 0.18

Covers plateau without reaching steep walls.

### Kernel

Matern ν = 2.5

Smooth but flexible; standard in BO practice.

### Length Scale Bounds

[0.01, 5.0] for x2 and x4

Allows sharp or smooth behaviour.

### Noise Level

initial = 1e-4
bounds = [1e-8, 0.1]

Near-deterministic but not rigid.

### Candidates

30,000 Sobol points (2D)

Dense coverage of small trust region.

### Distance Guard

min_dist = 0.025

Avoids duplicates while allowing fine local moves.


### Optimisation Restarts

50 L-BFGS-B restarts

Reliable EI maximisation in 2D.


### EI Jitter

xi = 0.001

Prevents EI collapse on flat surfaces.


## Hyperparameter Tuning Method

- GP hyperparameters optimised via marginal likelihood **only in 2D**, improving stability.
- Structural hyperparameters based on:
  - Prior runs
  - EDA
  - TuRBO-style heuristics

Manual adjustment possible if diagnostics indicate failure.


## Entire Flow of the Strategy

1. Identify current best point.
2. Select top `n_local` observations.
3. Fix `x1` and `x3` at best values.
4. Project data onto (`x2`, `x4`) subspace.
5. Normalize local `y` values.
6. Fit 2D GP with ARD.
7. Define 2D trust region (±0.18).
8. Generate Sobol candidates in region.
9. Reconstruct full 4D points.
10. Apply minimum distance filter.
11. Compute Expected Improvement.
12. Record best Sobol candidate.
13. Run multi-start L-BFGS-B to maximise EI.
14. Compare Sobol and gradient solutions.
15. Perform boundary diagnostics.
16. Verify GP mean and variance at selected point.
17. Output final 4D query.


## Hypothesis Framework

### Core Assumptions

- Plateau contains true optimum.
- `x1`, `x3` locally inactive.
- `x2`, `x4` capture variation.
- EI correlates with improvement probability.

### If Assumptions Hold

- GP learns stable 2D length scales.
- Selected point keeps `x1`, `x3` fixed.
- `x2`, `x4` lie within trust region interior.
- Predicted mean > 1.0.
- EI top candidates show diversity.

### If Assumptions Break

- Fixing `x1`, `x3` blocks access to optimum.
- EI concentrates at trust boundaries.
- 2D GP shows unstable scales.
- Plateau curvature misestimated.

## Critical Analysis

### Strengths

- Reduces dimensionality → improves GP stability.
- Avoids contamination from extreme negative regions.
- EI balances exploration and exploitation.
- Distance and boundary guards prevent pathology.

### Weaknesses

- Depends on correctness of ARD diagnosis.
- Trust region assumption may be wrong.
- Heuristic structural parameters.
- Still relies on GP marginal likelihood optimisation.

## Failure Modes

- Better plateau exists outside trust region.
- `x1` or `x3` become active outside local region.
- Noise higher than assumed.
- Budget too small to sample active subspace adequately.

---
  
## f5 — Log-GP Expected Improvement with Corner-Restricted Candidate Set (Heavy-Tail Strategy)


## Objective of Submission

To propose a principled, competition-grade strategy for choosing the next query for **f5** using:

- A **log-transformed Gaussian Process**
- **Expected Improvement (EI)** as the acquisition function
- A **corner-restricted candidate set** concentrated in the high-value region

This directly addresses the prior failure mode where UCB selected points with very low predicted mean simply because uncertainty was large.


## 3 Key Assumptions

### 1. Monotone Corner Structure

The function increases as all four inputs move toward 1.0.  
The optimal region lies near the **[0.85, 1.0]^4** corner.

### 2. Heavy-Tailed Positive Outputs

The objective is strictly positive and spans several orders of magnitude.  
Log-transforming stabilizes variance and smooths multiplicative effects.

### 3. Low Observational Noise

Evaluations are effectively deterministic.  
A small but non-zero noise level prevents pathological interpolation.


## Research Backing

This strategy is inspired by:

- **SMAC-style log-GP modelling**
- **Expected Improvement**
- GP bandit exploration theory
- Boundary-focused optimisation for monotone Lipschitz functions

### Key Academic References

- Hutter, Hoos, Leyton-Brown (2011) — Sequential Model-Based Algorithm Configuration (SMAC)
- Srinivas et al. (2010) — Gaussian Process Optimization in the Bandit Setting
- Malherbe & Vayatis (2017) — Global Optimization of Lipschitz Functions

These works justify:
- Log-surrogate modelling for heavy-tailed objectives
- EI over aggressive UCB
- Concentrating search near boundary regions containing the best point


## Explorative Principle (Function-Specific)

For **f5**:

- Outputs span 4+ orders of magnitude.
- All strong values occur when all inputs approach 1.0.

### Why Log-GP?

If \( y \) is multiplicative in structure, then:

\[
\log(y)
\]

is smoother and closer to additive.  
This improves GP fit stability with limited data.

### Why Expected Improvement (EI)?

UCB with large beta can select:

- Very low mean
- Very high sigma
- Unrealistic improvement candidates

EI instead computes:

\[
\mathbb{E}[\max(0, f(x) - f_{best})]
\]

If the predicted mean is far below the best, EI ≈ 0 — even if uncertainty is high.

This anchors exploration to **realistic improvement zones**.

### Why Restrict to the Corner?

Since all observed good points lie near the upper boundary:

- Sampling globally wastes evaluations.
- Restricting to [0.85, 1.0]^4 is structurally aligned.
- Supported by monotone Lipschitz regret theory.


## Black Box Optimization Competition Context

### Competition

NeurIPS 2020 Black Box Optimization Challenge  
Bayesmark-style algorithm configuration benchmarks

### Winning Approaches

Top-performing teams (e.g., SMAC-based methods, Squirrel AI) used:

- Log-GP surrogates
- EI or log-EI acquisitions
- Candidate generation concentrated in promising regions


## Why This Strategy Is Ideal for f5

- 4+ order-of-magnitude output range → log transform appropriate
- Best region clearly near upper corner
- Previous UCB (β=6) failure showed sigma dominance
- EI prevents sigma-only selection
- Corner restriction concentrates computational budget

This is precisely the regime where SMAC-style methods excel.


## Justification Under Expensive Evaluations

Because f5 evaluations are costly:

- Global sampling is unacceptable
- Log transform reduces sample complexity
- Corner restriction avoids low-probability regions
- EI ensures improvement-oriented selection

Every query must have a realistic chance of beating the current best.


## Tech Stack

- `numpy`
- `scipy.stats.qmc.Sobol`
- `scikit-learn`:
  - `GaussianProcessRegressor`
  - `Matern`
  - `WhiteKernel`
- `scipy.stats.norm` (for EI)


## Hyperparameters and Settings

### Structural Hyperparameters

| Parameter | Value | Rationale |
|------------|--------|-----------|
| `corner_lower` | 0.85 | Encloses high-value region while allowing variation |
| `n_corner` | 7000 | Dense 4D Sobol coverage |
| `min_dist` | 0.03 | Prevents duplicate querying |
| `log_shift` | 1.0 | Ensures positivity and stabilizes scale |
| `xi` | 0.01 | Prevents EI collapse |


### GP Hyperparameters

| Parameter | Value | Rationale |
|------------|--------|-----------|
| Matern ν | 2.5 | Smooth but flexible |
| Length scale bounds | [0.01, 5.0] | Allows steep gradients |
| Noise bounds | [1e-6, 1e-2] | Prevents zero-noise instability |
| `n_restarts_optimizer` | 30 | Stable marginal likelihood optimisation |


## Hyperparameter Tuning Method

- Structural parameters chosen via:
  - Empirical EDA
  - Literature defaults
  - Theoretical monotone-boundary guidance
- GP hyperparameters optimized via log marginal likelihood with restarts

No automated meta-optimisation due to small dataset size.


## Entire Flow of the Strategy

1. Load `X_train_5`, `y_train_5`
2. Apply log transform: y_log = log(y + log_shift) 
3. Generate 7000 Sobol candidates in [0.85, 1.0]^4
4. Apply minimum distance filter (0.03)
5. Fit ARD Matern GP on `(X_train_5, y_log)`
6. Compute predictive mean and std
7. Compute EI using current best log-value
8. Select candidate with highest EI
9. Report predicted mean (inverse-transformed for interpretability)
10. Return selected 4D point


## Hypothesis Framework

### Core Assumptions

- Optimum lies in upper corner
- Log surface is smooth
- Noise is small but non-zero


### If Assumptions Hold

- GP models log surface well
- EI peaks near slight improvements
- Selected candidate lies near but not identical to best
- Progressive improvement without wasted evaluations


### If Assumptions Break

- Optimum outside corner → stagnation
- Log surface not smooth → misestimated sigma
- Noise underestimated → overfitting
- Distance guard too strict → no viable candidates

Fallback would require expanding the candidate region.


## Critical Analysis

### Strengths

- Log transform handles heavy tails robustly
- EI prevents uncertainty-only exploitation
- Corner restriction maximizes sample efficiency
- Structurally aligned with empirical observations


### Weaknesses

- Strong assumption about corner location
- GP hyperparameters still sensitive to small data
- EI can become overly conservative if model overconfident


## When It May Fail

- Secondary optimum outside the corner
- Log transform insufficient to stabilize surface
- Poor marginal likelihood convergence
- Excessively strict distance threshold


## Strategic Summary

This is a **boundary-focused, log-stabilized, improvement-anchored Bayesian optimisation strategy** designed specifically for:

- Heavy-tailed objectives
- Monotone corner structure
- Expensive evaluations
- Small data regimes

It directly corrects the prior UCB failure mode and aligns with SMAC-style competition-grade practice.

---

## f6 — ARD-GP Sequential Neighbourhood Densification with Interior Peak Search


## Objective of Submission

Aggressively refine the interior peak of an expensive five-dimensional black-box function by:

- Concentrating samples in the empirically supported high-value region
- Fitting a Gaussian Process surrogate
- Using Expected Improvement (EI) to propose the next query

The goal is targeted local refinement rather than global exploration.


## 3 Key Assumptions (Pre-Run)

1. **Interior Peak**  
   The maximum lies in an interior neighbourhood of the domain, particularly in the \( x_4 \)–\( x_5 \) region, not at the corners.

2. **Sparse Active Subspace (Hypothesised)**  
   \( x_4 \) and \( x_5 \) were assumed strongly active;  
   \( x_1 \), \( x_2 \) largely inactive;  
   \( x_3 \) weakly active.

3. **Local Smoothness**  
   The response surface is smooth enough near the best point for a Matérn GP to model local curvature.


## Explorative Principle

The strategy was **sequential neighbourhood densification**:

- Identify the current best point.
- Restrict search to a plausible improvement neighbourhood.
- Fit an ARD GP surrogate.
- Compute Expected Improvement.
- Select the candidate with highest EI.

The neighbourhood restriction was:

- \( x_4 \ge 0.70 \)
- \( x_5 \le 0.28 \)
- Excluding the known bad corner \( x_4 \approx 1.0, x_5 \approx 0.0 \)

This avoided wasting evaluations in already disproven regions.


## What the GP Length Scales Revealed (Key Learning)

This run produced a critical structural insight:

> **The function is not sparse-active-subspace in the GP sense.**

The learned length scales show:

- All five dimensions contribute similarly to the kernel distance.
- No dimension collapses to extremely short scale.
- No dimension explodes to extremely long scale.

### Interpretation

- The strong effect of \( x_4 \) and \( x_5 \) reflects a **broad smooth gradient**, not a sharply localized active subspace.
- The GP did not support the hypothesis that \( x_4 \) and \( x_5 \) should have artificially short length scales.
- The asymmetric bounds were **fighting the marginal likelihood**, not guiding it.


## Correct Structural Understanding

The function appears to be:

- Smooth in all five dimensions
- Approximately uniform in scale
- Tilted in the direction:
  - Increasing \( x_4 \)
  - Decreasing \( x_5 \)

The neighbourhood restriction is still correct — but **because of directional tilt**, not sparse dimensionality.

This distinction matters.


## Submitted Point
x1 = 0.553475
x2 = 0.343241
x3 = 0.535219
x4 = 0.771873
x5 = 0.140689

- Predicted mean: **-0.5235**
- Predicted standard deviation: **0.107**
- EI score: **0.060**

### Interpretation

- Predicted mean is above current best.
- Uncertainty is realistic (not inflated).
- EI is positive and meaningful.
- Candidate lies in the intended interior region.

This is the strongest evidence-based selection available from this run.


## Why EI Worked Here

EI selected a point that:

- Is near the current best
- Slightly improves predicted mean
- Maintains non-trivial uncertainty

It avoided:
- Corner exploration
- Overconfident repetition
- Pure uncertainty chasing


## Structural Diagnostics from This Run

1. The function is **not sparse-active-subspace**.
2. Length scale asymmetry was an incorrect prior.
3. The surface is globally smooth.
4. The dominant structure is directional tilt, not sharp curvature.

This is the most important learning from Week 9.


## What to Fix for Week 10

### 1. Remove Asymmetric Length Scale Bounds

Do not bias specific dimensions.

Allow the GP to optimise freely over all dimensions.

The previous bounds encoded a false structural belief.


### 2. Widen the Neighbourhood Filter

Current filter reduced 10,000 Sobol candidates to 749  
→ 92.5% reduction in candidate diversity.

New proposal:

- \( x_4 \ge 0.65 \)
- \( x_5 \le 0.32 \)

This keeps directional bias while increasing diversity in 5D.


### 3. Reconsider EI vs UCB

Current EI behaviour:

- Predicted means clustered near -0.52 to -0.55
- Very small improvement margin
- EI highly sensitive to small mean errors

Because the function is broadly smooth:

- EI may collapse if mean slightly misestimated
- UCB with \( \beta = 3.0 \) could be more robust

UCB would:

- Still push toward high \( x_4 \), low \( x_5 \)
- Be less fragile to mean calibration error
- Maintain exploration pressure

This is a legitimate Week 10 experiment:
**EI vs UCB under uniform length scales.**


## Revised Structural Understanding

The correct model of f6 is:

- Broadly smooth in 5D
- No true sparse axis-aligned active subspace
- Dominant global tilt:
  - Increase \( x_4 \)
  - Decrease \( x_5 \)

Search restriction is still justified —
but because of geometry, not sparsity.


## Strategic Summary

This run achieved two outcomes:

1. Proposed a defensible next evaluation using EI.
2. Revealed the true kernel geometry of the function.

The key learning:

> Do not force sparse-active-subspace structure when the marginal likelihood does not support it.

Week 10 should:

- Remove asymmetric bounds
- Widen directional neighbourhood
- Compare EI vs UCB (β = 3.0)

This shifts the strategy from **dimension biasing** to **geometry-aware smooth optimisation**.

---

## f7 — Constrained GP Posterior Variance Maximisation with Hard Dimension Filtering


## Objective of Submission

Select a single final query for **f7** that:

- Lies inside the empirically validated high-value region  
- Maximises Gaussian Process posterior variance  
- Has the highest probability of revealing a new peak or clarifying the basin structure  

Because this is the final expensive evaluation, the goal is **maximum information gain inside the plausible optimum region**, not global exploration.


## 3 Key Assumptions

### 1. Structural Region of the Optimum

High-value points consistently satisfy:

- \( x_1 \) small  
- \( x_5 \) small  
- \( x_6 \) large  

Thus the global maximum lies within that constrained subregion.


### 2. GP Posterior Variance Is Meaningful

A Matérn ARD GP with moderate smoothness:

- Represents local structure adequately  
- Produces calibrated posterior variance  
- Can be trusted for uncertainty-driven exploration  


### 3. Global Exploration Is Wasteful

Regions outside the validated subspace:

- Are repeatedly low-value  
- Offer negligible upside  
- Should not consume the final evaluation  


## Explorative Principle

This strategy is a **constrained pure-variance maximisation** rule.

Conceptually:

- GP-UCB with very large β → dominated by variance
- Here: ignore mean entirely
- Select the point with largest posterior standard deviation
- But only inside the validated structural region

Hard filters applied:

- \( x_1 < 0.20 \)  
- \( x_5 < 0.55 \)  
- \( x_6 > 0.50 \)  

Within that region:

\[
x^* = \arg\max \sigma(x)
\]

This aggressively explores the least-understood portion of the optimum basin.


## GP Kernel Diagnostics (Critical Output)

Learned ARD length scales:
[0.458, 0.285, 5.0, 5.0, 5.0, 0.0614]

Interpretation:

| Dimension | Length Scale | Interpretation |
|------------|-------------|----------------|
| X1 | 0.458 | Active |
| X2 | 0.285 | Active |
| X3 | 5.0 | Inactive |
| X4 | 5.0 | Inactive |
| X5 | 5.0 | Inactive (surprising) |
| X6 | 0.061 | Highly active |

### Important Structural Insight

- **X6 is the dominant active dimension**
- X1 and X2 show moderate activity
- X3 and X4 are flat locally
- X5 appears flat to the GP despite negative empirical correlation

This means:

> The hard filter on X5 is necessary — the GP is not penalising high X5 itself.

The filtering step is therefore structurally justified.


## Selected Point (Week 9 Query)
x1 = 0.020
x2 = 0.998
x3 = 0.127
x4 = 0.788
x5 = 0.383
x6 = 0.520

### Structural Validation

- X1 = 0.020 → strongly compliant
- X5 = 0.383 → comfortably within constraint
- X6 = 0.520 → above threshold
- Distance from nearest training point: **0.61**

This is a **genuinely novel** point.


## Comparison to Top 3 Observations

| Point | X1 | X2 | X3 | X4 | X5 | X6 | y |
|-------|----|----|----|----|----|----|----|
| Best | 0.070 | 0.320 | 0.577 | 0.211 | 0.386 | 0.707 | 2.448 |
| 2nd | 0.097 | 0.335 | 0.370 | 0.382 | 0.465 | 0.751 | 1.599 |
| 3rd | 0.058 | 0.492 | 0.247 | 0.218 | 0.420 | 0.731 | 1.365 |
| Week 9 | 0.020 | 0.998 | 0.127 | 0.788 | 0.383 | 0.520 | ? |

### Observations

Matches validated structure in:
- X1 (low)
- X5 (low)
- X6 (above threshold)

Differs substantially in:
- X2 (very high)
- X3 (lower)
- X4 (much higher)
- X6 (lower than top-3 cluster)

This is deliberate: it explores a different combination inside the subregion.


## Risk Assessment

Potential weaknesses:

- X6 = 0.520 is below the top-3 cluster (> 0.70)
- X2 and X4 diverge strongly from top patterns

Possible outcomes:

- Moderate value (structural confirmation)
- Breakthrough if unconstrained dimensions contain additional structure

Given one remaining evaluation, exploration within region > duplication near best.


## Verdict

Submit this point.

Rationale:

- All structural constraints satisfied
- Maximal posterior variance in validated region
- Distance guard respected
- Kernel diagnostics align with empirical structure
- Pre-commitment rule honoured

This is the highest-information final query.


## Post-Submission Pre-Commitment

Record before observing result:

### If y > 2.448

- X2, X3, X4 contain additional structure.
- X6 > 0.50 is sufficient.
- Top-3 cluster at X6 > 0.70 was not a hard requirement.


### If 1.0 < y < 2.448

- Region confirmed.
- X6 needs to be higher.
- Future rule: X6 > 0.70.


### If y < 1.0

Likely cause:

- X2 = 0.998 or
- X4 = 0.788

Add future structural constraints:

- X2 < 0.60
- X4 < 0.45


## Strategic Summary

This submission:

- Trusts structural evidence
- Distrusts GP mean (due to bias)
- Trusts GP variance
- Maximises information inside validated basin
- Avoids global waste

It is a principled, constrained uncertainty maximisation strategy appropriate for a final, expensive evaluation.

---

## f8 — GTBO-Inspired Axis-Aligned Active Subspace Exploitation with ARD-GP and LogEI


## Objective of Submission

Design a **focused exploitation strategy** for **f8** that:

- Concentrates almost all search effort inside the empirically confirmed active subspace  
- Uses a Gaussian Process (ARD kernel) to model the function  
- Maximizes **log expected improvement (logEI)** to select the next query  
- Avoids wasted evaluations in dimensions that are inactive  

Goal: refine the interior of the known high-value region to find a point better than the current best.


## 3 Key Assumptions

1. **Active Subspace Dimensions:** Only **x1, x3, x7** meaningfully affect the function at current sample size; all others are largely inactive.

2. **Interior High-Value Region:** The local maximum is not fully resolved; unsampled interior points within the active subspace may improve the current best.

3. **GP Guidance is Sufficient:** A single well-calibrated ARD GP with logEI can direct effective exploitation inside the subspace.


## Explorative Principle

**Key idea:** Focus on exploitation inside the known high-value subspace rather than global wandering.

Mechanism:

- ARD GP uses **asymmetric length scales**:

  - x1, x3: short bounds (highly active)  
  - x7: intermediate bounds (weakly active)  
  - all others: long bounds (effectively inactive)  

- Generate **Sobol candidates**:

  - 70% in the high-value subspace (x1, x3, x7)  
  - 30% globally for safety  

- Apply **minimum distance guard** (0.05) to avoid duplicates  
- Compute **logEI** for each candidate  
- Take top 20 by logEI as **warm starts**  
- Locally optimise logEI with **L-BFGS-B**  
- Submit the **best refined point**

This approach aggressively exploits the interior of the known good region without wasting budget on inactive dimensions.


## Tech Stack

- **numpy**: array and distance computations  
- **scipy**:

  - `stats.qmc.sobol` for candidate generation  
  - `optimize.minimize` (L-BFGS-B) for local refinement  
  - `stats.norm` for PDF/CDF in logEI  

- **scikit-learn**:

  - `GaussianProcessRegressor`  
  - `Matern`, `WhiteKernel`, `ConstantKernel`


## Hyperparameters

| Component | Settings | Rationale |
|-----------|----------|-----------|
| Kernel | Matérn ν=2.5 + WhiteKernel + ConstantKernel | Smooth but flexible function approximation |
| ARD length scales | x1: [0.02,0.30], x3: [0.02,0.30], x7: [0.05,0.60], others: [0.80,20] | Force active dims to have short scales, inactive dims to be flat |
| Noise | [1e-8,0.5], alpha=1e-6 | Near-deterministic function; allow minimal jitter |
| GP optimizer | 29 restarts + hand-crafted initial | Multi-start ensures robust marginal likelihood optimization |
| Sobol candidates | 5000 total, 70% in high-value subspace | Exploit subspace while retaining small chance of global discovery |
| Distance guard | 0.05 | Prevent near-duplicate queries |
| Acquisition | logEI | More stable than EI near a high incumbent |
| Local optimizer | L-BFGS-B, max 200 iterations | Gradient-based refinement of logEI |


## Week 9 Kernel Validation

Learned kernel:
0.763**2 * Matern(ls=[0.3, 1.85, 0.3, 1.68, 2.48, 20, 0.6, 20], nu=2.5) + WhiteKernel(noise=1e-8)

**Interpretation:**

| Dim | Length scale | Status |
|-----|-------------|--------|
| X1 | 0.30 | Active, correctly short |
| X2 | 1.85 | Moderately inactive |
| X3 | 0.30 | Active, correctly short |
| X4 | 1.68 | Moderately inactive |
| X5 | 2.48 | Moderately inactive |
| X6 | 20 | Inactive |
| X7 | 0.60 | Weakly active |
| X8 | 20 | Inactive |

Notes:

- Noise ~ 1e-8 confirms near-deterministic function  
- X1 and X3 hitting upper bounds → could allow shorter scales (week 10: consider [0.02,0.20] and [0.02,0.20] for more flexibility)  
- Inactive dims correctly pinned to long length scales


## Week 9 Query Point
x1: 0.241028, x2: 0.000000, x3: 0.149637, x4: 0.000000
x5: 1.000000, x6: 0.187244, x7: 0.264684, x8: 1.000000


**Validation:**

- X1, X3, X7: inside high-value subspace ✅  
- X2, X4, X5, X8: boundary values due to inactivity (length scales long)  
- Minimum distance guard respected

Notes:

- Inactive dims corner-collapse is expected; does not affect exploitation  
- Could optionally fix inactive dims to current best values to match GTBO/SAASBO recommendations


## Prediction Assessment

| Metric | Value |
|--------|-------|
| GP predicted mean | 10.0583 |
| GP predicted std | 0.2129 |
| Current best | 9.9487 |
| EI | 0.1508 |

Interpretation:

- Mean above current best → credible improvement target  
- EI is significant (0.15), much higher than trivial (<0.01)  
- Confirms interior point exploitation rather than a trivial corner


## Pre-Commitment Guidance for Week 10

| Observed y | Interpretation | Week 10 Action |
|------------|---------------|----------------|
| y > 9.948 | New best, strategy validated | Tighten active dim bounds: x1 [0,0.30], x3 [0,0.20], x7 [0,0.45]; rerun logEI locally |
| 9.7 < y < 9.948 | Promising, not new best | Retain strategy; tighten active upper bounds to 0.20; constrain inactive dims to interior [0.1,0.9] |
| y < 9.5 | GP mean off; inactive corners may be pathological | Switch to global Sobol with soft directional weighting on x1,x3,x7; skip L-BFGS-B |

Additional note:

- Consider fixing inactive dims X4=0.198, X8=0.663 to match the current best, avoiding corner-collapse and matching GTBO recommendations


## Strategic Summary

- Exploits **confirmed active subspace**  
- ARD GP with asymmetric bounds separates active vs inactive dims  
- **LogEI** targets promising interior points  
- L-BFGS-B refines candidates in high-value subspace  
- Inactive dimensions corner-collapse is acceptable but can be fixed if needed  
- Well-suited for expensive evaluation budgets and nearly saturated high-value regions