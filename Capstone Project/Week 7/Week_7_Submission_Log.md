# Week 7 — Aggressive Global Exploration + Competition-Backed Acquisition Innovation

This document outlines the Week 7 submission plan. The strategy shifts away from the
return-to-baseline philosophy of Week 6 and adopts competition-validated acquisition
methods, gradient-free hyperparameter tuning, and structured multi-basin exploration
across all functions that have stagnated for three or more consecutive weeks.

Emphasis:
- Competition-backed acquisition methods replacing standard UCB and EI where justified
- Gradient-free hyperparameter optimisation (CMA-ES, Differential Evolution) for
  functions with multimodal likelihood surfaces
- Multi-trust-region exploration (TuRBO-M style) for high-dimensional stagnated functions
- Opposition-based and exclusion-zone candidate generation to escape confirmed local basins
- Retain validated calibration frameworks; do not regress on calibration improvements
  achieved in Weeks 5 and 6

---

## Meta-Strategy

### Competition-Validated Methods + Structured Multi-Basin Exploration

**Primary surrogate**
- ARD-GP per function with per-dimension length scales
- Gradient-free hyperparameter tuning (CMA-ES or Differential Evolution) where the
  log marginal likelihood is multimodal or poorly conditioned
- Multi-start L-BFGS-B retained only where Week 6 confirmed smooth likelihood surfaces

**Hyperparameter tuning policy**
- CMA-ES for functions with flat or multimodal likelihood surfaces (f1, f4, f7)
- Differential Evolution for functions with sparse active subspaces (f6)
- Multi-start L-BFGS-B retained for functions with confirmed smooth likelihoods (f5, f8)
- Gradient-based methods removed where they failed to escape local optima in prior weeks

**Warping policy**
- Log transform retained for f5 — validated across six weeks
- Box-Cox retained for f3 — calibration stable for three consecutive weeks
- No other warping introduced

**Residual correction**
- MLP removed from all functions without exception
- No reintroduction under any framing — structurally unreliable at N between 20 and 50

**Acquisition**
- Max Value Entropy Search (MES) proxy for f2 — targets global maximum uncertainty
  reduction rather than local exploitation
- EI concentrated near x3 boundary for f5 — validated by Week 3 best result
- UCB with β = 3.0 to β = 4.0 for stagnated functions requiring aggressive exploration
- Thompson Sampling retained for f8 active subspace search
- Pareto front selection removed — produced acquisition stagnation on f7 in Week 5

**Candidate generation**
- Opposition-based sampling for f4 — probes mirrored regions relative to current best
- Full-domain Sobol with exclusion zone for f8 — prevents resampling confirmed local basin
- Scrambled Halton for f1 — low-discrepancy coverage of near-flat 2D surface
- Sobol with soft directional weighting for f6 and f7

**Ensemble policy**
- f7: restore Week 4 four-model ensemble (GP, SVR, KNN, XGB) with CV-based weighting
  and soft directional acquisition weighting — validated as best-result configuration
- f8: retain 0.7 GP + 0.3 KNN — empirically validated on Week 5 novel point
- All other functions: single GP surrogate, no ensemble

---

## Learning Objectives

1. Do competition-validated acquisition methods (MES, CMA-ES, TuRBO-M, opposition
   sampling) produce measurable improvement over standard UCB and EI on functions that
   have stagnated for three or more weeks?

2. Does gradient-free hyperparameter tuning (CMA-ES, Differential Evolution) recover
   better GP configurations than multi-start L-BFGS-B on functions with flat or
   multimodal likelihood surfaces?

3. Does multi-trust-region exploration in TuRBO-M style identify distinct basins in f7's
   7D space that single-region Thompson Sampling has failed to locate?

4. Does exclusion-zone Sobol exploration in f8 confirm or falsify the hypothesis that
   the Week 3 best is a local rather than global maximum?

5. Does opposition-based sampling in f4 escape the plateau that has resisted three
   consecutive GP-UCB queries?

---

## Structural Hypothesis for Week 7

- Standard acquisition methods (UCB, EI, Thompson Sampling) have been exhausted on
  stagnated functions. Competition-winning methods introduce structural diversity that
  random restarts of the same acquisition cannot.
- Gradient-free hyperparameter tuning is more reliable than multi-start L-BFGS-B when
  the GP likelihood surface is flat or multimodal — the condition that applies to f1,
  f4, and f7 given their low signal-to-noise ratios and diffuse structure.
- Multi-basin exploration is required for f7. A single trust region centred on the
  current best has been queried from multiple angles across four weeks without
  improvement. Three parallel trust regions exploring the best, worst, and most distal
  points simultaneously is a structural change, not a parameter change.
- Exclusion-zone global exploration for f8 is the correct test of whether the Week 3
  best is a local or global maximum. One query outside the confirmed neighbourhood
  provides more information than another local refinement.

Week 7 prioritises structural diversity in acquisition and hyperparameter tuning over
incremental parameter adjustment. Calibration frameworks validated in prior weeks are
retained unchanged. Experimentation is targeted at the acquisition and search components
that have demonstrably stalled.

---

## Function-Specific Strategy Summary

| Function | Week 7 Direction | Competition Reference | Key Change from Week 6 |
|---|---|---|---|
| f1 | Scrambled Halton + CMA-ES GP | BBOB 2013 COCO, IPOP-CMA-ES | Replace L-BFGS-B with CMA-ES; Halton candidates replace Sobol |
| f2 | Max Value Entropy Search + MES proxy | SigOpt 2021, Eriksson et al. | Replace UCB with MES; boundary candidates added |
| f3 | KNN-guided EI + GA hyperparameter tuning | NeurIPS 2020 Optuna Track, Duxiaoman | GA replaces grid search for KNN k and feature weights |
| f4 | Opposition-based ARD-GP UCB + CMA-ES | GECCO 2022, Awad et al. | Opposition sampling added; CMA-ES replaces L-BFGS-B |
| f5 | Log-GP EI concentrated near x3 boundary | SMAC, Hutter et al. 2011 | Strip candidates 90% near x3 = 1.0; asymmetric length-scale bounds |
| f6 | Differential Evolution ARD-GP UCB | SAASBO, Eriksson and Jankowiak 2021 | Differential Evolution replaces L-BFGS-B for hyperparameters |
| f7 | TuRBO-M three trust regions + CMA-ES ensemble | TuRBO-M, Eriksson et al. NeurIPS 2019 | Three parallel trust regions; CMA-ES tunes ensemble weights |
| f8 | Full-domain Sobol UCB + exclusion zone + ARD | NeurIPS 2020 Bayesmark, NVIDIA Research | Exclusion zone 0.25; 15000 Sobol candidates; 0.7/0.3 GP-KNN retained |

---

## Competition Research Backing by Function

### f1 — BBOB 2013 COCO Workshop
Tanabe and Fukunaga demonstrated IPOP-CMA-ES restart strategy as the top-performing
method on noiseless benchmark functions, particularly on flat and poorly conditioned
surfaces where gradient-based methods fail due to vanishing derivatives. CMA-ES in log-
hyperparameter space is directly applicable to f1's near-zero output range.

### f2 — SigOpt Open Source BBO Challenge 2021
Eriksson et al. at Meta AI Research used Max Value Entropy Search as a core component
of their winning strategy. MES targets reduction of uncertainty about the global maximum
location rather than local exploitation — appropriate for f2 where four consecutive
weeks have produced upward quantile bias, indicating the ensemble is systematically
underestimating.

### f3 — NeurIPS 2020 BBO Challenge, Optuna Track
Duxiaoman Financial AI Team used non-parametric surrogates with gradient-free
hyperparameter search for functions with unknown smoothness. GA tuning of KNN k and
feature weights is directly motivated by their approach — the mixed integer-continuous
hyperparameter space of KNN is non-differentiable and requires gradient-free search.

### f4 — GECCO 2022 Real-Parameter BBO Workshop
Awad et al. used differential evolution with opposition-based population initialisation
as their winning strategy. Opposition-based sampling doubles coverage by probing 1 - x
for each candidate x, directly addressing f4's distributed negative gradient structure
where the optimum may be in the mirror region of the current best.

### f5 — SMAC, Hutter et al. 2011
Sequential Model-Based Algorithm Configuration established the combination of log-
transformed GP with EI and concentrated candidate generation as the standard for
heavy-tailed expensive optimisation. f5's four-order-of-magnitude output range and
dominant x3 gradient are exactly the conditions SMAC was designed for.

### f6 — SAASBO, Eriksson and Jankowiak UAI 2021
Sparse Axis-Aligned Subspace Bayesian Optimisation demonstrated that ARD kernels with
global hyperparameter search outperform gradient-based tuning on functions with sparse
active subspaces. f6's confirmed two-dimensional active subspace (X4, X5) with long
length scales on X1-X3 is precisely the target use case.

### f7 — TuRBO-M, Eriksson et al. NeurIPS 2019
Scalable Global Optimisation via Local Bayesian Optimisation demonstrated that multiple
parallel trust regions outperform single-region BO on diffuse high-dimensional functions
with multiple basins. f7's 7D structure with weak partial correlations and four
consecutive weeks without improvement is the canonical case for TuRBO-M application.

### f8 — NeurIPS 2020 Bayesmark Track, NVIDIA Research
The winning approach used advanced Bayesian optimisation with explicit exclusion of
exhausted regions and ARD-GP ensemble modelling. f8's confirmed local maximum at the
Week 3 best point and three consecutive weeks without improvement match the stagnation
profile that motivated NVIDIA's exclusion-zone strategy.

---

## Function-Specific Strategies

# f1 — Scrambled Halton Space Filling with CMA-ES Tuned Gaussian Process


## Objective of Submission

The objective is to propose a new query point for function f1 using an exploration strategy that is robust to flat, near-constant objective surfaces.

The goal is to explore new regions of the 2D domain efficiently while maintaining a stable surrogate model, even when the function provides almost no gradient information.


## Three Key Assumptions

1. The function is extremely flat, with outputs close to zero and no meaningful gradient structure.  
2. Each function evaluation is expensive, so repeated sampling near existing points must be avoided.  
3. Gaussian Process hyperparameters can be stabilised using CMA-ES even when the likelihood surface is nearly flat.  


## Research Backing

### Randomised Quasi-Monte Carlo

- Niederreiter, H. (1992). *Randomized quasi-Monte Carlo methods and low discrepancy sequences for numerical integration.*

Supports the use of low-discrepancy sequences such as Halton for uniform domain coverage.

### CMA-ES

- Hansen, N. and Ostermeier, A. (2001). *Completely derandomised self-adaptation in evolution strategies (CMA-ES).*  
- Hansen, N. (2016). *The CMA Evolution Strategy: A tutorial.*

Establishes CMA-ES as a robust gradient-free optimiser for ill-conditioned problems.


## Black Box Optimisation Context

**Competition:** BBOB 2013 COCO Workshop on Noiseless Function Benchmarking  

**Winning Strategy:** Tanabe and Fukunaga — IPOP-CMA-ES restart strategy  

CMA-ES has repeatedly demonstrated strong performance in continuous black-box optimisation, particularly in poorly conditioned landscapes.


## Explorative Principle

This strategy combines two complementary mechanisms:

### 1. Scrambled Halton Sequence (Input Space Exploration)

- Generates candidate points covering the 2D unit square evenly.  
- Each new point tends to fall in an underexplored region.  
- Ideal when the function is nearly flat and no gradient signal exists.  

### 2. CMA-ES for GP Hyperparameter Tuning

When outputs are extremely small and similar:

- GP log marginal likelihood becomes nearly flat.  
- Gradient-based optimisers fail due to vanishing derivatives.  

CMA-ES:

- Operates in log-hyperparameter space.  
- Uses covariance matrix adaptation.  
- Remains effective on flat or poorly conditioned likelihood surfaces.  

Together:

- Halton drives geometric exploration.  
- CMA-ES stabilises the surrogate model.  


## Why This Strategy Is Appropriate for f1

Observed outputs across six weeks range approximately from:

1e-57 to 5e-15

This behaves like a near-zero surface with minimal exploitable structure.

Implications:

- Gradient-following strategies are ineffective.  
- Random sampling wastes evaluations.  
- GP hyperparameters are difficult to tune via gradient descent.  

The combined Halton + CMA-ES approach:

- Ensures uniform exploration of the 2D domain.  
- Prevents sampling near existing points.  
- Avoids degenerate GP hyperparameters.  


## Tech Stack

- numpy  
- scipy  
- scikit-learn  
- cma (Python package for CMA-ES)  


## Hyperparameters and Recommended Settings

### CMA-ES

- sigma0: 0.3  
  Moderate initial step size in log space  

- maxiter: 200  
  Sufficient iterations for covariance adaptation  

- tolx: 1e-6  
  Stops when hyperparameters stabilise  

- restarts: 3  
  Improves robustness  


### Gaussian Process

- Kernel: Matern + WhiteKernel  
- Length-scale range: log-uniform from 1e-3 to 20  
  Covers short and long correlations  

- Noise range: log-uniform from 1e-10 to 1e-2  
  Prevents numerical instability  


### Exploration Parameters

- Halton candidates: 2000  
  Dense 2D coverage  

- Minimum distance guard: 0.05  
  Prevents redundant evaluations  

- UCB beta: 0.05  
  Keeps surrogate influence minimal  


## Hyperparameter Tuning Method

CMA-ES maximises the Gaussian Process log marginal likelihood.

Properties:

- Gradient-free  
- Robust on flat likelihood surfaces  
- Searches log-hyperparameter space  
- Self-adapting covariance matrix  


## Full Strategy Workflow

1. Collect existing f1 data.  
2. Define GP with Matern kernel + White noise.  
3. Express log marginal likelihood in log-hyperparameter space.  
4. Run CMA-ES to minimise negative log marginal likelihood.  
5. Fit GP using optimal hyperparameters.  
6. Generate 2000 scrambled Halton candidates in [0,1]^2.  
7. Filter candidates closer than 0.05 to existing points.  
8. Predict GP mean and standard deviation.  
9. Compute UCB acquisition with beta = 0.05.  
10. Select candidate with highest acquisition value.  
11. Submit point and update dataset.  


## Hypothesis Framework

### Core Assumptions

- The function is nearly flat.  
- Evaluations are expensive.  
- CMA-ES stabilises GP hyperparameters.  

### If Assumptions Hold

- Halton points cover the domain evenly.  
- Distance guard prevents redundancy.  
- CMA-ES finds stable hyperparameters.  
- Exploration proceeds efficiently.  

### If Assumptions Break

- Strong gradients may require larger beta.  
- GP may misrepresent the function.  
- CMA-ES may struggle if likelihood surface is irregular.  


## Critical Analysis

### Strengths

- Robust to flat objective functions.  
- Principled space-filling design.  
- CMA-ES handles ill-conditioned likelihoods.  
- Avoids redundant evaluations.  

### Weaknesses

- More complex than simple random search.  
- CMA-ES adds computational overhead.  
- GP may be unnecessary if the function is perfectly flat.  

### Failure Modes

- Sharp discontinuities.  
- Very limited computational budget.  
- High-dimensional spaces where Halton loses efficiency.  


## Summary

This strategy prioritises geometric exploration and hyperparameter robustness over gradient exploitation.  

For a near-zero, expensive 2D black-box function, scrambled Halton sampling combined with CMA-ES stabilised Gaussian Processes provides a principled and computationally controlled approach to exploration.

---

# f2 — Max Value Entropy Search with Multi-Start L-BFGS-B Gaussian Process Hyperparameter Tuning

## Objective of Submission

Design an aggressive, principled exploration strategy for a 2-dimensional expensive black-box function using:

- Gaussian Process (GP) surrogate  
- Max Value Entropy Search (MES) style exploration  
- Multi-start L-BFGS-B gradient-based GP hyperparameter tuning  

This approach prioritizes reducing uncertainty about the global maximum over local exploitation.


## Three Key Assumptions

1. The function is smooth enough in both inputs to be reasonably modelled by a Matern GP.  
2. The 2D input space is bounded in `[0, 1]^2`, making quasi-random Sobol sequences effective for coverage.  
3. The global maximum is not yet well-localized; reducing uncertainty about its location is more valuable than purely exploiting high observed values.


## Research Backing

- **Gaussian Process Regression:** Hyperparameters optimised via L-BFGS-B on log marginal likelihood.  
- **Max Value Entropy Search:** Chooses points that most reduce uncertainty about the global maximum, approximated here via posterior variance.

### Supporting Papers

- Rasmussen, C.E., & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.  
- Wang, Z., & Jegelka, S. (2017). *Max Value Entropy Search for Efficient Bayesian Optimization*. NeurIPS.


## Explorative Principle

- **Gaussian Process:** Models the unknown 2D function, providing predictive mean and standard deviation for candidate points.  
- **MES Approximation:** High GP variance points indicate regions that maximally reduce uncertainty about the global maximum.  
- **Candidate Selection:** Generate Sobol points and boundary candidates, remove those too close to existing observations, then select the point with the highest standard deviation.  
- **Rationale:** Aggressive exploration ensures each expensive evaluation provides maximal information about the function landscape.


## Black Box Optimization Competition

- **Competition:** SigOpt Open Source Black Box Optimization Challenge 2021  
- **Winning Team:** Eriksson et al., Meta AI Research


## Why This Strategy Is Ideal for f2

- GP effectively models smooth surfaces in 2D with relatively few observations.  
- MES targets uncertainty reduction, focusing on localizing the global maximum quickly.  
- L-BFGS-B efficiently tunes GP hyperparameters in low dimensions.  
- Sobol plus boundary candidate design ensures edges and corners are explored.  
- Minimum distance guard prevents redundant evaluations.


## Tech Stack

- `numpy` — numerical operations  
- `scipy` — Sobol sequence generation  
- `scikit-learn` — GP regression with Matern kernel and white noise


## Hyperparameters and Recommended Settings

| Hyperparameter | Recommended Value | Rationale |
|----------------|-----------------|-----------|
| n_sobol | 5000 | Dense coverage of 2D unit square, low discrepancy in low dimensions |
| n_boundary | 1000 | Probe edges to counter interior oversampling |
| boundary_w | 0.05 | 5% strip width near edges, balances coverage |
| min_dist_guard | 0.04 | Avoid redundant points near existing observations |
| n_restarts | 20 | Multi-start L-BFGS-B for robust hyperparameter optimisation |
| length_scale_bounds | [(0.02, 5.0), (0.05, 10.0)] | Asymmetric bounds allow flexibility; first dimension possibly more influential |
| noise_level_bounds | (1e-8, 0.5) | Handles nearly noise-free to moderately noisy observations |

**Hyperparameter Tuning Method:** Multi-start L-BFGS-B on GP log marginal likelihood. Exact gradients from scikit-learn ensure efficient convergence; multiple restarts reduce the risk of local optima.


## Entire Flow of the Strategy

1. Collect existing training data: `X_train_2` (n × 2) and `y_train_2` (n × 1).  
2. Generate Sobol candidate points over `[0, 1]^2`.  
3. Generate additional boundary candidates within narrow strips along each edge.  
4. Combine interior and boundary candidates; clip to `[0, 1]^2`.  
5. Apply minimum distance guard: discard candidates too close to existing points.  
6. Fit a GP with Matern kernel and white noise, tuning hyperparameters via multi-start L-BFGS-B.  
7. Compute GP predictive mean and standard deviation for remaining candidates.  
8. Use standard deviation as MES proxy; higher σ indicates high potential information gain.  
9. Select the candidate with highest σ as the next query point.  
10. Evaluate the black-box function at this point; append to `X_train_2` and `y_train_2`.  
11. Repeat until evaluation budget is exhausted.


## Hypothesis Framework

### Core Assumptions

- Matern GP adequately models the 2D function.  
- GP hyperparameters can be reliably learned via gradient-based optimisation.  
- Regions with high GP variance correspond to areas that reduce uncertainty about the global maximum.

### Expected if Assumptions Hold

- GP fits observed data with low residuals and realistic uncertainty.  
- Selected points spread across domain, including underexplored areas.  
- Global maximum localized quickly; best observed value improves efficiently.

### Expected if Assumptions Break

- Highly non-smooth or discontinuous functions → GP misestimates uncertainty.  
- High or heteroscedastic noise → GP with white noise term may misrepresent uncertainty.  
- Sharp peaks in narrow regions → MES proxy may miss maxima due to GP smoothing.


## Critical Analysis

### Strengths

- Data-efficient in low dimensions; ideal for expensive evaluations.  
- Probabilistic model provides mean and uncertainty.  
- MES focuses on reducing global maximum uncertainty.  
- Gradient-based hyperparameter tuning is computationally efficient.

### Weaknesses

- GP assumptions may be violated, degrading performance.  
- MES proxy ignores some posterior shape information.  
- Strongly exploration-focused; lacks explicit exploitation control.

### Potential Failure Modes

- Functions with discontinuities or sharp spikes not captured by smooth GP.  
- Heavy-tailed or input-dependent noise → GP white noise inadequate.  
- True optimum outside `[0,1]^2` domain → strategy fails.

---

# f3 — KNN-Guided Expected Improvement with Genetic Algorithm Hyperparameter Tuning

## Objective of Submission

Design an exploration-focused strategy for a 4-dimensional expensive black-box function using:

- KNN surrogate for the mean  
- Gaussian Process (GP) for uncertainty  
- Genetic Algorithm (GA) for tuning KNN hyperparameters  

This strategy targets functions with weak structure and non-monotonic behaviour where standard GP models may ggle.

---

## Three Key Assumptions

1. The function is 4-dimensional, weakly structured, and non-monotonic.  
2. KNN approximates local behaviour better than GP when smoothness is unknown.  
3. Optimal KNN hyperparameters (k and feature weights) are non-differentiable and require a gradifree search.

---

## Research Backing

- Candelieri et al. (2021). *Non-parametric Black Box Optimisation with Kernel Regression.*  
- Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimisation and Machine Learning.*  
- Cover, T., & Hart, P. (1967). *Nearest Neighbor Pattern Classification.* IEEE Transactions on Information Theory.  
- Zhang, T., & Zhou, Z. (2007). *Machine Learning Based on Kernel Ridge Region.* Neurocomputing.

---

## Explorative Principle

Separate modelling of the mean and uncertainty:

- **KNN Mean:** Captures local behaviour in weakly structured, non-monotonic functions.  
- **GP Uncertainty:** Provides reliable σ estimates even when the mean is biased.  
- **Expected Improvement (EI):** Combines KNN mean and GP uncertainty to select points likely to improve over the current best while exploring uncertain regions.  
- **GA Tuning:** Finds optimal k and feature weightthout assuming differentiability.

---

## Black Box Optimization Competition

- **Competition:** NeurIPS 2020 Black Box Optimisation Challenge, Optuna Track  
- **Winning Team:** Duxiaoman Financial AI Team — used non-p
tric surrogates for unknown smoothness.

---

## Why This Strategy Is Ideal for f3

- KNN uses nearby observations, robust for irregular surfaces.  
- GA finds good k and feature weights for mixed integer-continuous hyperparameters.  
- GP guides exploration via uncertainty.  
- Minimum distance guard prevents redundant evaluations.  Efficient under expensive evaluation constraints.

---
Tech Stack

- `numpy`  
- `scipy`  
- `scikit-learn`  

---

## Hyperparameters and Recommended Settings

### Genetic Algorithm

- Population size: 30 — enough diversity for 5D hyperparameter space (k + 4 weights)  
- Generations: 40 — ~1200 evaluations, sufficient for small datasets  
- Crossover rate: 0.8 — encourages mixing good solutions  
- Mutation rate: 0.15 — maintains diversity  

### KNN

- k range: `[2, min(10, n_obs - 1)]` — small k for local behaviour  
- Feature weights per dimension: `[0.1, 3.0]` — moderate up/down weighting  

### Candidates

- Sobol points: 8000 — dense coverage of 4D unit cube  
- Minimum distance guard: 0.05 — avoids sampling near existing points  

### Expected Improvement

- ξ parameter: 0.01 — small margin for meaningful ivement  

### GP Uncertainty

- Kernel bounds for σ: default scikit-learn GP bounds

---

## Hyperparameter Tuning Method

- GA minimises leave-one-out cross-validation MSE of weighted KNN (k + feature weights)  
- Gradient-free, suitable for mixed integer-continusearch space  
- Uses tournament selection, uniform crossover, Gaussian mutation, and elitism

---

## Entire Flow of the Strategy

1. Collect existing data `X_train_3`, `y_train_3`.  
2. Run GA to optimise KNN hyperparameters (k and feature weights) using leave-one-out CV MSE.  
3. Fit weighted KNN on `X_train_3` with learned feature weights.  
4. Fit GP on `X_train_3` and `y_train_3` to model uncertainty only.  
5. Generate 8000 Sobol candidates in 4D unit cube.  
6. Apply minimum distance guard to filter candidates.  
7. Apply feature weights to candidates and predict KNN mean.  
8. Compute GP σ at each candidate.  
9. Com EI using KNN mean, GP σ, and current best y.  
10. Select candidate with highest EI as the next query point.

---

## Hypothesis Framework

### Core Assumptions

- KNN provides better mean predictions than GP.  
- GP gives reasonable uncertainty estimates even if its mean is biased.  
- GA can find near-optimal hyperparameters with modest evaluation budget.

### Expected if Assumptions Hold

- KNN captures local structure with low CV error.  
- EI highlights promising and uncertain candidates.  
- Strategy gradually explores regions likely to improve the current best.

### Expected if Assumptions Break

- KN
erperforms → EI may be misled.  
- GP uncertainty miscalibrated → EI over/underexplores.  
- GA fails → KNN performance degrades.

---

## Critical Analysis

### Strengths

- Non-parametric KNN avoids smoothness assumptions.  
- Mean and uncertainty modelled separately for better accuracy.  
- GA handles mixed integer-continuous hyperparameters gradient-free.  
- EI balances exploration and exploitation.

### Weaknesses

- KNN scales poorly with large datasets.  
- Leave-one-out CV can be noisy for very small datasets.  
- GP uncertainty may be inaccurate for highly irregular functions.

### Potential Failure Modes

- Strong global structure ptured by GP but missed by KNN.  
- Extremely few observations → high variance in both KNN and GP.  
- Hyperparameter search space too large for GA budget.



---

# f4 — Opposition-Based ARD Gaussian Process UCB with CMA-ES Hyperparameter Tuning

## Objective of Submission

Propose a principled, exploration-focused strategy for selecting the next query point in a 4-dimensional expensive black-box function using:

- Opposition-based candidate design  
- ARD Gaussian Process surrogate  
- Upper Confidence Bound (UCB) acquisition function  
- Hyperparameters tuned via CMA-ES  

This strategy aims to balance global coverage and aggressive exploration under limited evaluation budgets.


## Three Key Assumptions

1. The function is 4-dimensional, smooth enough to be modelled by a Matern Gaussian Process, with possible nonlinear interactions.  
2. All four input dimensions are informative; the surface is genuinely 4D rather than sparse.  
3. The GP log marginal likelihood may have multiple local optima, so gradient-based optimisation alone is not reliable.


## Research Backing

- Rahnamayan, S., et al. (2008). *Opposition-Based Differential Evolution.* IEEE Transactions on Evolutionary Computation.  
- Hansen, N., & Ostermeier, A. (2001). *Completely Derandomised Self-Adaptation in Evolution Strategies (CMA-ES).* Evolutionary Computation Journal.  
- Snoek, J., et al. (2012). *Practical Bayesian Optimisation of Machine Learning Algorithms.* NeurIPS.


## Explorative Principle

The strategy combines three main ideas tailored to a 4D, potentially complex surface:

### 1. Opposition-Based Sampling

- For each candidate `x` ∈ [0,1]^4, also consider its opposition point `1 - x`.  
- Doubles coverage and probes regions far from current samples.  
- Useful for functions with declining or changing behaviour across the domain.

### 2. ARD Gaussian Process Surrogate

- Automatic Relevance Determination (ARD) Matern kernel assigns separate length scales per dimension.  
- Captures nonlinear interactions and different smoothness along each axis.  

### 3. UCB Acquisition with CMA-ES Hyperparameter Tuning

- Upper Confidence Bound: `UCB(x) = μ(x) + β * σ(x)` balances exploitation (mean) and exploration (uncertainty).  
- Hyperparameters (4 length scales + 1 noise) are tuned with CMA-ES by minimising the negative log marginal likelihood.  
- CMA-ES avoids local optima common in multimodal likelihood surfaces.


## Black Box Optimization Competition

- **Competition:** GECCO 2022 Real-Parameter Black Box Optimisation Workshop  
- **Winning Team:** Awad et al., using differential evolution with opposition-based population initialisation.


## Why This Strategy Is Ideal for f4

- All four inputs are important; global coverage is necessary.  
- Opposition-based sampling explores both sides of the space relative to existing points.  
- ARD GP adapts to different sensitivities per dimension.  
- CMA-ES robustly tunes hyperparameters in a multimodal likelihood landscape.  
- UCB with high exploration weight aggressively probes uncertain but promising regions.  
- Efficient use of expensive evaluations under limited budget.


## Tech Stack

- `numpy`  
- `scipy`  
- `scikit-learn`  
- `cma` (Python CMA-ES package)


## Hyperparameters and Recommended Settings

### CMA-ES

- `sigma0`: 0.5 — wide initial step to explore 5D hyperparameter space  
- `maxiter`: 300 — sufficient to adapt covariance and converge  
- `restarts`: 3 — multiple runs reduce trapping in poor basins  

### GP Hyperparameters

- Length scale search: log-uniform [1e-3, 15.0] per dimension  
- Noise search: log-uniform [1e-8, 0.1]  

### Candidate Generation

- Sobol candidates: 10,000 (before opposition)  
- Opposition fraction: implicit via mirroring  
- Minimum distance guard: 0.05  

### UCB

- Restarts for L-BFGS-B: 50  
- Beta parameter: 4.0 — high exploration pressure  


## Hyperparameter Tuning Method

- CMA-ES minimises the negative log marginal likelihood of the GP.  
- Covariance-adapting evolutionary strategy, gradient-free.  
- Effective for multimodal, non-convex likelihood landscapes.


## Entire Flow of the Strategy

1. Collect existing data: `X_train_4`, `y_train_4`.  
2. Generate 10,000 Sobol candidates in [0,1]^4.  
3. Create opposition candidates: `x → 1 - x`.  
4. Combine and clip candidates to [0,1].  
5. Apply minimum distance guard.  
6. Fit ARD Matern GP with hyperparameters tuned via CMA-ES.  
7. Define UCB: `μ + β * σ`.  
8. Run L-BFGS-B from multiple starting points, half near opposition of current best.  
9. Evaluate UCB on discrete candidate pool as a safety net.  
10. Compare best candidates from gradient optimisation and discrete pool; respect distance guard.  
11. Select and return the next query point for the black-box function.


## Hypothesis Framework

### Core Assumptions

- Function is smooth enough for Matern GP.  
- All four dimensions are relevant; opposition-based exploration adds value.  
- GP hyperparameter likelihood is multimodal; CMA-ES is preferable to gradient methods.

### If Assumptions Hold

- CMA-ES finds hyperparameters that yield sensible length scales.  
- UCB identifies uncertain, potentially high-value points.  
- Opposition candidates escape local basins and improve global coverage.

### If Assumptions Break

- Extreme noise → GP misestimates uncertainty; UCB may overexplore.  
- Some dimensions irrelevant → ARD kernel wastes capacity.  
- Likelihood nearly convex → CMA-ES may be overkill.  


## Critical Analysis

### Strengths

- Robust hyperparameter tuning in multimodal landscapes via CMA-ES.  
- Opposition-based sampling improves global coverage and escapes local basins.  
- ARD GP captures anisotropic smoothness.  
- UCB with high beta encourages aggressive exploration.

### Weaknesses

- CMA-ES is computationally expensive.  
- Opposition-based sampling may waste candidates if function is asymmetric.  
- Strong non-stationarity can reduce GP surrogate fidelity.

### Potential Failure Modes

- Sharp discontinuities or poor GP fit.  
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

It is designed for expensive 5D optimisation with heavy-tailed outputs and a dominant monotonic input dimension, maximising efficiency under a constrained evaluation budget.

---

# f6 — Differential Evolution Tuned ARD Gaussian Process UCB for Sparse Active Subspaces

## Objective of Submission

The objective is to propose a new query point for a 5-dimensional expensive black-box function (**f6**) using an Automatic Relevance Determination (ARD) Gaussian Process whose hyperparameters are tuned via Differential Evolution.

The strategy is designed to:

- Aggressively explore the domain  
- Focus modelling capacity on truly active dimensions  
- Operate efficiently under a limited evaluation budget  


## Three Key Assumptions

1. The true function depends strongly on only a subset of the 5 input dimensions, so an ARD Gaussian Process can exploit sparsity.

2. The GP log marginal likelihood over hyperparameters is multimodal, so gradient-based optimisers such as L-BFGS-B may converge to poor local optima.

3. Function evaluations are expensive, so it is preferable to invest computation in hyperparameter search and acquisition optimisation rather than random exploration.


## Research Backing

- Eriksson, D., and Jankowiak, M. (2021). *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO).* UAI 2021.  

- Storn, R., and Price, K. (1997). *Differential Evolution — A Simple and Efficient Heuristic for Global Optimisation over Continuous Spaces.* Journal of Global Optimization.  

- Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.  


## Explorative Principle

The strategy combines:

- An ARD Gaussian Process with per-dimension length scales  
- Global hyperparameter search using Differential Evolution  
- Upper Confidence Bound (UCB) acquisition  

This ensures:

- Active dimensions are identified automatically  
- Hyperparameters are optimised globally  
- Exploration and exploitation are balanced in a principled manner  


## ARD Gaussian Process for Sparse Active Subspaces

In a 5D input space, dimensions may not contribute equally.

An ARD kernel assigns a separate length scale to each dimension:

- **Short length scale** → active dimension (rapid variation)  
- **Long length scale** → inactive dimension (flat response)  

This aligns with the assumption that **f6** contains a sparse active subspace.

The model concentrates flexibility on relevant axes and avoids overfitting inactive ones.


## Differential Evolution for Hyperparameter Tuning

The GP log marginal likelihood over:

- 5 length scales  
- 1 noise level  

can be highly multimodal.

Gradient-based methods may converge to a basin where all length scales are similar, missing sparse structure.

Differential Evolution:

- Maintains a population of candidate solutions  
- Uses mutation and crossover  
- Explores multiple basins simultaneously  
- Requires no gradients  

This makes it well-suited for multimodal hyperparameter landscapes.


## UCB Acquisition for Aggressive Exploration

After fitting the GP, the acquisition function is:

---

# f7 — TuRBO-M with Three Trust Regions and CMA-ES Ensemble Weight Tuning

## Objective of Submission

The objective is to propose a new query point for the 7D black-box function **f7** using aggressive structured exploration via multiple local trust regions and an adaptively weighted surrogate ensemble.

The strategy:

- Explores multiple basins in parallel using three trust regions  
- Fits a local surrogate ensemble in each region  
- Tunes ensemble weights and trust region length using CMA-ES  
- Uses Thompson sampling with uncertainty inflation to select candidates  

This is designed for expensive evaluations where each query must maximise expected information gain.


## Three Key Assumptions

1. **f7 is moderately high-dimensional (7D) with limited observations**, so a single global Gaussian Process would oversmooth and miss local structure.

2. **Local models inside trust regions are more accurate than a single global surrogate**, and multiple trust regions can explore distinct basins of attraction in parallel.

3. **Ensemble weights and trust region length are critical hyperparameters**, but their effect on performance is non-smooth and non-differentiable, making gradient-based optimisation inappropriate.


## Research Backing

- Eriksson, D., Pearce, M., Gardner, J., Turner, R., and Poloczek, M. (2019). *Scalable Global Optimisation via Local Bayesian Optimisation (TuRBO-M).* NeurIPS 2019.  
- Kern, S., Hansen, N., Koumoutsakos, P. (2004). *Learning the Weights of Ensemble Models with CMA-ES.* Parallel Problem Solving from Nature.  
- NeurIPS 2020 Black Box Optimisation Challenge (Bayesmark Track) reports, where trust-region-based Bayesian optimisation and ensembles were used by top teams.


## Explorative Principle

The search is divided into multiple local regions explored in parallel. Each region uses a local surrogate ensemble, while CMA-ES tunes:

- Ensemble weights  
- Trust region length  

Exploration focuses on regions where at least one model predicts strong performance and uncertainty is high.


## TuRBO-M Style Trust Regions

Three trust regions are defined as hyper-rectangles in the 7D unit cube.

Trust region centres:

- **Incumbent region**: centred at the current best observed point  
- **Opposition region**: centred at the current worst observed point  
- **Distal region**: centred at the point farthest from the incumbent in input space  

This structure allows exploration of multiple basins simultaneously and reduces stagnation risk.


## Local Surrogate Ensemble (GP + SVR + KNN)

Inside each trust region:

- Gaussian Process (Matern kernel)  
- Support Vector Regressor (RBF kernel)  
- K-Nearest Neighbours (distance-weighted)  

Predictions are combined using weights tuned by CMA-ES to minimise cross-validated mean squared error.

This reduces reliance on any single surrogate under limited data conditions.


## CMA-ES for Hyperparameter Tuning

CMA-ES jointly tunes:

- Ensemble weights (sum = 1, each ≥ 0.05)  
- Trust region length  

The objective is cross-validated MSE of the ensemble.

CMA-ES is used because:

- The objective is non-smooth  
- No gradients are available  
- Weight dominance effects create discontinuities  


## Why This Strategy Is Suitable for f7

Characteristics of f7:

- 7-dimensional input space  
- Low sample-to-dimension ratio  
- Expensive evaluations  

Implications:

- A global GP may oversmooth  
- Local modelling is safer  
- Multiple basins may exist  
- Exploration must be structured and parallel  

This approach:

- Avoids repeated sampling in weak regions  
- Explores different basins simultaneously  
- Adapts model combination to local data quality  
- Prioritises uncertainty-aware candidate selection  


## Tech Stack

- numpy  
- scipy  
- scikit-learn  
- scikit-learn neighbors  
- scikit-learn svm  
- cma (Python package for CMA-ES)  


## Hyperparameters and Recommended Settings

### CMA-ES

- sigma0: 0.2  
- maxiter: 150  

### Trust Regions

- Number of trust regions: 3  
- Initial trust region length: 0.40  
- Length search range: [0.20, 0.80]  

### Ensemble

- Models: GP (Matern), SVR (RBF), KNN (distance-weighted)  
- Weight constraint: sum = 1, each weight ≥ 0.05  
- Variance inflation (GP): 3.0  

### Candidate Generation

- Sobol candidates per trust region: 2000  
- Thompson sampling draws: 100  
- Minimum distance guard: 0.04  


## Full Strategy Workflow

1. Collect existing data: `X_train_7`, `y_train_7`.  
2. Use CMA-ES to tune ensemble weights and trust region length.  
3. Define three trust region centres (best, worst, farthest).  
4. Construct bounded trust regions within `[0,1]^7`.  
5. For each region:  
   - Select local data  
   - Fit GP, SVR, and KNN  
6. Generate Sobol candidates inside region.  
7. Apply minimum distance guard.  
8. Compute ensemble mean and inflated GP uncertainty.  
9. Perform Thompson sampling (100 draws).  
10. Select best candidate per region.  
11. Choose final candidate with highest ensemble mean.  
12. Evaluate f7 and update dataset.  


## Hypothesis Framework

### If Assumptions Hold

- CMA-ES identifies strong ensemble weight combinations  
- Trust region length adapts appropriately  
- Regions explore distinct basins  
- Best observed value improves efficiently  

### If Assumptions Break

- Global surrogate may outperform local models  
- Cross-validation may mislead CMA-ES  
- Sparse data may cause overfitting  
- True optimum may lie outside trust regions  


## Critical Analysis

### Strengths

- Parallel basin exploration  
- Reduced reliance on single surrogate  
- Gradient-free hyperparameter tuning  
- Uncertainty-aware candidate selection  

### Weaknesses

- Higher computational overhead  
- Cross-validation instability at small N  
- Heuristic trust region count  
- Increased implementation complexity  


## Failure Modes

- Nonstationary structure not captured by models  
- Extremely small dataset  
- Poor trust region placement  
- Suboptimal CMA-ES convergence  


## Summary

This strategy applies TuRBO-M style multi-trust-region exploration with a locally fitted surrogate ensemble whose weights and region size are tuned using CMA-ES.

It is designed for expensive 7D optimisation where global models risk oversmoothing and structured, parallel exploration is required.



---

# f8 — Full-Domain ARD Gaussian Process UCB with Exclusion Zone and Multi-Start L-BFGS-B


## Objective of Submission

The objective is to propose a new query point for the 8D black-box function f8 that aggressively explores the full domain while avoiding re-exploration of a known local maximum.

The strategy uses a probabilistic surrogate model to prioritise promising and uncertain regions under a strict budget of expensive function evaluations.


## Three Key Assumptions

1. f8 has a clear active subspace in x1, x3, and x7, which drive most of the output variation.  
2. The Week 3 best point is a local maximum in its neighbourhood and has been sufficiently explored.  
3. The Gaussian Process log marginal likelihood is smooth and well-conditioned, so multi-start L-BFGS-B can reliably recover good hyperparameters.  


## Research Backing

- Srinivas, N., Krause, A., Kakade, S., and Seeger, M. (2010). *Gaussian Process Optimisation in the Bandit Setting.* ICML 2010.  
- Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* Chapter 5, hyperparameter learning.  
- Eriksson, D., Pearce, M., Gardner, J., Turner, R., and Poloczek, M. (2019). *Scalable Global Optimisation via Local Bayesian Optimisation (TuRBO).* NeurIPS 2019.  


## Black-Box Optimisation Context

**Competition:** NeurIPS 2020 Black Box Optimisation Challenge (Bayesmark Track)  

**Winning Team:** NVIDIA Research  

Their approach relied on advanced Bayesian optimisation and ensemble modelling, demonstrating the effectiveness of principled surrogate-based search in expensive black-box settings.


## Explorative Principle

The strategy is designed to:

- Cover the full 8D domain  
- Avoid regions already confirmed as locally optimal  
- Exploit known structure (active dimensions)  
- Maintain robustness under expensive evaluations  

### 1. Full-Domain Sobol Space Filling

Sobol sequences generate candidates that evenly cover the 8D unit cube.

Rationale:

- Prevents clustering around known regions  
- Ensures global exploration  
- Necessary when the true global optimum may lie far from the current best  


### 2. Exclusion Zone Around Known Local Maximum

A hypersphere around the Week 3 best point is excluded.

Purpose:

- Prevents redundant evaluations  
- Forces exploration beyond a region that has failed to improve for multiple weeks  

This is function-specific: f8 exhibits a strong local maximum that has resisted improvement.


### 3. UCB with ARD Gaussian Process + KNN Ensemble

**ARD Gaussian Process**

- Separate length scales per dimension  
- Focuses automatically on x1, x3, and x7  
- Treats inactive dimensions with long length scales  

**Upper Confidence Bound (UCB)**

- Balances mean prediction and uncertainty  
- Encourages exploration in uncertain but promising regions  

**KNN Component**

- Stabilises predictions in sparse regions  
- Adds local bias correction  

This is appropriate because:

- f8 appears smooth and near-deterministic  
- GP can model structure effectively  
- UCB can exploit active subspace while exploring globally  


## Why This Strategy Is Ideal for f8

Observations:

- 8-dimensional input space  
- Clear active subspace (x1, x3, x7)  
- Best value found early and not improved  

Implications:

- Likely local maximum in explored region  
- Expensive evaluations require high information gain  
- Re-exploration near Week 3 best is wasteful  

Design alignment:

- Sobol ensures global coverage  
- Exclusion zone prevents redundancy  
- ARD GP focuses on active dimensions  
- UCB targets promising + uncertain regions  


## Tech Stack

- numpy  
- scipy  
- scikit-learn  
- scikit-learn neighbors  


## Hyperparameters and Recommended Settings

### Exploration

- Sobol candidates: 15000  
  Dense 8D coverage  

- Exclusion zone radius: 0.25  
  Removes known local basin  

- Minimum distance guard: 0.04  
  Prevents redundant evaluations  

- Downweight factor: 0.50  
  Penalises exhausted active subspace corner  


### Acquisition

- UCB beta: 3.0  
  Encourages exploration (consistent with GP-UCB theory)  

- Variance inflation: 3.0  
  Amplifies uncertainty contribution  

---

### Ensemble

- Weights: 0.7 GP, 0.3 KNN  
  GP provides global smooth model  
  KNN stabilises sparse regions  


### Hyperparameter Optimisation

- L-BFGS-B restarts: 30  
  Suitable for smooth likelihood surfaces  

- Active dimension length-scale bounds: [0.01, 3.0]  
  Allows rapid variation  

- Inactive dimension length-scale bounds: [0.1, 50.0]  
  Encourages flatness if appropriate  

- Noise bounds: [1e-6, 1e-2]  
  Tight bounds for near-deterministic outputs  


## Hyperparameter Tuning Method

Multi-start L-BFGS-B maximises the GP log marginal likelihood.

Justification:

- Efficient gradient-based optimiser  
- Effective when likelihood is smooth  
- Recommended in Rasmussen & Williams for GP hyperparameter learning  


## Full Strategy Workflow

1. Collect existing data: X_train_8, y_train_8.  
2. Fit ARD GP using multi-start L-BFGS-B.  
3. Fit KNN regressor.  
4. Generate 15000 Sobol candidates in [0,1]^8.  
5. Remove candidates inside exclusion zone.  
6. Apply minimum distance guard.  
7. Predict GP mean and standard deviation.  
8. Predict KNN mean.  
9. Compute ensemble mean:  
   0.7 × GP mean + 0.3 × KNN mean.  
10. Inflate GP standard deviation by 3.0.  
11. Downweight candidates in exhausted active corner by 0.5.  
12. Compute UCB:  
    Downweighted ensemble mean + beta × inflated std.  
13. Select highest-UCB candidate.  
14. Evaluate black-box function and update dataset.  


## Hypothesis Framework

### Core Assumptions

- Clear active subspace (x1, x3, x7).  
- Week 3 best is a local maximum.  
- Likelihood surface is smooth.  
- Evaluations are expensive.  


### If Assumptions Hold

- ARD learns short length scales for active dimensions.  
- Exclusion zone prevents redundant sampling.  
- UCB focuses on uncertain promising regions.  
- New high-value regions are discovered or global optimality is confirmed.  


### If Assumptions Break

- Misidentified active subspace leads to poor modelling.  
- Exclusion zone hides true global optimum.  
- L-BFGS-B converges to suboptimal hyperparameters.  
- Tight noise bounds underestimate uncertainty if function is noisy.  


## Critical Analysis

### Strengths

- Principled global coverage via Sobol sampling.  
- Exploits structural knowledge through ARD.  
- Avoids redundant evaluations with exclusion zone.  
- Balances smooth GP modelling with local KNN stabilisation.  
- Uses efficient hyperparameter optimisation.  


### Weaknesses

- Sensitive to correct active dimension identification.  
- Exclusion radius is heuristic.  
- Ensemble and variance inflation increase complexity.  
- L-BFGS-B can still converge to local optima.  

---

### Failure Modes

- Active subspace changes or is misestimated.  
- True global optimum lies inside excluded region.  
- Strong nonstationarity not captured by Matern kernel.  
- Too few observations for reliable GP learning.  


## Summary

This strategy prioritises structured global exploration guided by an ARD Gaussian Process while explicitly preventing re-exploration of a known local maximum.

It is designed for expensive, moderately smooth, near-deterministic high-dimensional optimisation where local refinement has stagnated and global uncertainty must be leveraged efficiently.
