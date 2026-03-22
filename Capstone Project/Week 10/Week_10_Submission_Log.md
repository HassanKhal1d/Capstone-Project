## Function-Specific Strategies


## f1 — Two-Zone Log EI Trust Region Exploration with Relaxed GP

## Objective of Submission

Aggressively explore the most promising local region of the expensive black-box function **f1** by:

- Probing the gradient direction in **x1**
- Refining the neighborhood around the current best point
- Using a Gaussian Process with **log Expected Improvement (logEI)**
- Emitting rich diagnostics to validate modeling assumptions

The strategy is explicitly **local and high-resolution**, not global.


## 3 Key Assumptions

1. The true optimum lies near the current best point in the **(x1, x2)** plane.
2. The function varies sharply in **x2** but is relatively flat in **x1**, requiring asymmetric modeling.
3. A Matérn GP with carefully chosen length-scale bounds provides useful calibrated uncertainty for exploration.


## Research Backing

### Key Papers

- Eriksson et al. (2019), *Scalable Global Optimization via Local Bayesian Optimization (TuRBO)*, NeurIPS.  
  Introduces trust-region Bayesian optimization centered around the incumbent.

- Ament et al. (2023), *Unexpected Improvements to Expected Improvement*, NeurIPS.  
  Introduces **logEI**, which stabilizes EI near strong incumbents.

- Bull (2011), *Convergence Rates of Efficient Global Optimization Algorithms*, JRSS B.  
  Shows that asymmetric length-scale bounds aligned with active dimensions improve convergence.


## Explorative Principle

This is **local but aggressive exploration**.

Instead of searching the full domain, we focus on a small trust region around the current best where dramatic improvements were observed.

### Modeling Adjustments

- Relax the **x1** upper length-scale bound to allow near-flat behavior if supported by data.
- Constrain **x2** to short length scales to capture sharp variation.

### Two-Zone Candidate Design

**Zone A — Gradient Continuation**
- Extends the productive x1 direction
- Avoids overshooting

**Zone B — Local Refinement**
- Densely explores around the incumbent
- Captures small-scale improvements

Candidates from both zones are evaluated using **logEI**, selecting the point that balances:

- Predicted improvement
- Model uncertainty

This ensures exploration remains constrained to the productive region while still probing uncertainty.


## Why This Strategy Fits f1

- f1 is expensive → every evaluation must be high-value.
- Week 9 produced a large improvement in a narrow region.
- Global exploration is low expected return.

This approach:

- Exploits the identified trust region
- Uses asymmetric GP bounds aligned with structure
- Employs logEI to prevent vanishing gradients
- Avoids edge artifacts via two-zone design


## Tech Stack

- `numpy` — numerical operations and distance computation  
- `scipy.stats.norm` — normal PDF/CDF for logEI  
- `scipy.stats.qmc.Sobol` — low-discrepancy sampling  
- `sklearn.gaussian_process.GaussianProcessRegressor`  
  - `Matern`
  - `WhiteKernel`
- `sklearn.preprocessing.StandardScaler` — output standardization  


## Hyperparameters

### GP Kernel

- Matérn ν = 2.5 + WhiteKernel

### Length-Scale Bounds

- x1: [0.1, 10.0] (relaxed, allow flatness)
- x2: [0.001, 0.5] (tight, sharp structure)

### Noise

- Bounds: [1e-10, 1.0]

### Optimization

- GP restarts: 30
- Marginal likelihood optimized with multi-start L-BFGS-B

### Candidate Generation

- 4000 candidates per zone (8000 total)

#### Zone A
- x1 ∈ [0.640, 0.680]
- x2 ∈ [0.700, 0.750]

#### Zone B
- x1 ∈ [0.665, 0.705]
- x2 ∈ [0.710, 0.740]

### Distance Guard

- Minimum Euclidean distance: 0.015
- Fallback: 0.008

### Acquisition

- logEI with ξ = 0.0 (pure local exploitation)


## Hyperparameter Rationale

- **Matérn ν = 2.5**: Standard choice for moderately smooth black-box functions.
- **Relaxed x1 upper bound (10.0)**: Allows GP to represent near-flatness if data supports it.
- **Tight x2 bounds**: Reflect empirically observed sharp variation.
- **30 restarts**: Reduce risk of poor local optima in GP hyperparameter fitting.
- **Dense candidate sampling (8000 total)**: High resolution within small 2D trust region.
- **Distance guard**: Prevents redundant evaluations.


## Full Strategy Flow

1. Load existing observations:
   - `X_train_1`
   - `y_train_1`

2. Apply Gaussian copula transform to `y_train_1`  
   - Maps ranks to standard normal
   - Stabilizes modeling for heavy-tailed distributions

3. Standardize transformed outputs to zero mean and unit variance.

4. Fit ARD Matérn GP:
   - Relaxed x1 bounds
   - Tight x2 bounds
   - WhiteKernel for noise

5. Print diagnostics:
   - Fitted length scales
   - Noise level
   - Log marginal likelihood

6. Generate Sobol candidates:
   - Zone A
   - Zone B

7. Combine and clip to [0,1].

8. Apply minimum distance guard.

9. Compute posterior mean and standard deviation.

10. Compute logEI for each candidate.

11. Select candidate with highest logEI.

12. Print diagnostics:
    - Selected coordinates
    - GP mean
    - GP std
    - logEI
    - Zone origin
    - Comparison vs reference point

13. Return selected point.


## Hypothesis Framework

### Core Assumptions

- The optimum lies within the two defined zones.
- x2 is sharply active; x1 relatively flat.
- GP uncertainty is reasonably calibrated locally.

### Expected If Assumptions Hold

- Fitted length scales: x1 long, x2 short.
- Selected point from Zone A or B with high logEI.
- New evaluation improves or closely matches current best.
- Diagnostics consistent and stable.

### Expected If Assumptions Break

- Unexpected length scales (e.g., x2 long).
- Low or uniform logEI values.
- No improvement, suggesting:
  - Optimum lies outside trust region, or
  - Model misspecification.

If logEI collapses or becomes flat, expand or relocate the trust region in the next iteration.

# Week 10 Diagnostic Analysis – Flat Acquisition Surface and Structural Probe Decision

## What the Output Is Actually Saying

### 1. X1 Length Scale Hit 10.0

X1 length scale has hit **10.0**, the new upper bound.

This is not a modelling artefact anymore.

Across three consecutive GP fits:

- Week 9: upper bound = 3.0 → hit  
- Week 10 automatic: upper bound = 3.0 → hit  
- Week 10 override: upper bound = 10.0 → hit  

The GP is unambiguously signalling that the function is **completely flat in X1** across all 19 observations.

This is a consistent structural signal.

### 2. Log EI Identical Across All Candidates

Log EI for every candidate in the viable set is: -2.0184

Examples:

- Automatic recommendation at X1 = 0.6177  
- Override at X1 = 0.6457  
- All other candidates in both zones  

The override is better by exactly: 0.0000 log units


This means:

- Posterior mean is numerically identical
- Posterior standard deviation is numerically identical
- The acquisition surface is completely flat

The GP has no gradient to offer.


### 3. Why the Acquisition Surface Is Flat

The fitted GP has:

- X1 length scale = 10.0  
- X2 length scale = 0.010  

Implications:

- The posterior varies only in X2
- Within the candidate X2 band [0.700, 0.750], variation is negligible
- All candidates are evaluated at effectively equivalent X2 distances relative to training data

The tiny differences in LogEI are below numerical precision, which is why they round identically.


# What This Means Structurally

The GP is:

- Completely flat in X1  
- Extremely sharp in X2  
- Producing a numerically indistinguishable acquisition surface  

The model cannot guide the next move.

Any candidate in the current zones is equivalent under the posterior.


# The Real Question: What Does the Empirical Data Say?

Ignore the GP and examine raw observations near the peak.

| Observation | X1     | X2     | y           |
|------------|--------|--------|------------|
| Week 9 best | 0.6846 | 0.7254 | 1.867e-09 |
| Week 3     | 0.7297 | 0.7293 | 5.315e-15 |
| Week 0     | 0.7310 | 0.7330 | 7.711e-16 |

Observation:

The Week 9 improvement came from shifting:

- X1: 0.730 → 0.685  
- X2 remained near 0.725–0.729  

If X1 is truly flat, why did shifting X1 by -0.045 produce a 400 million times improvement?

There are two possible explanations.


## Explanation A: Narrow X2 Ridge

The function contains a sharp ridge in X2.

The Week 9 point landed closer to the ridge centre in X2.

The X1 movement was coincidental.


## Explanation B: Undetected X1 Gradient

There is a real X1 gradient, but:

- Only two observations exist near X1 = 0.73
- Only one at X1 = 0.685
- N = 19 total observations

Insufficient local density to detect X1 slope.


## Which Explanation Is Supported?

The GP strongly favours Explanation A.

Compute covariance in X2:

- X2 difference: 0.0039  
- Length scale: 0.010  

Covariance:
exp(-(0.0039 / 0.010)^2)
= exp(-0.15)
= 0.86


The GP sees these points as highly correlated.

Yet their function values differ by 400 million times.

This behaviour is consistent with:

- Extremely sharp spike in X2
- Not a smooth X1 gradient


# Week 10 Submission Strategy

The GP cannot discriminate between candidates.

X1 is confirmed flat.

The correct response is to stop relying on the surrogate and instead directly test the structural hypothesis.


## Best Observed X2 Values

- 0.7254 → best result  
- 0.7293 → worse  
- 0.7330 → worse  

Values above 0.7254 have been tested and are inferior.

Values below 0.7254 have not been tested in the peak region.

The unsampled direction is downward in X2.


# Week 10 Submission
X1 = 0.6846
X2 = 0.7180


Rationale:

- Hold X1 fixed at current best
- Probe X2 at 0.7254 - 0.0074
- Step size mirrors prior spacing magnitude
- Tests whether ridge centre lies below 0.7254

No GP fitting required.  
No acquisition optimisation required.  

This is a direct structural probe.


# Learning Point

The override comparison showed:

- Automatic and override LogEI identical
- Acquisition surface flat

Without that diagnostic, one might falsely believe the GP was guiding meaningfully.

When acquisition values are identical across thousands of candidates:

- The surrogate has no posterior gradient
- The acquisition function is no longer informative
- The correct response is structural reasoning

The model has reached informational saturation in this region.

At that point, empirical geometry dominates surrogate inference.

---

## f2 — Geometric Maximin Probe in the Unsampled Low-x2 Strip  
*(x1 anchored near best low-x2)*

---

## Objective of Submission

Place a single, defensible, high-information query in the **unsampled low-x2 strip** inside the confirmed **x1 active band** using **geometry (maximin)** instead of an unstable Gaussian Process fit.

The selected point:

- Maximises distance to existing observations  
- Respects a strict minimum distance guard  
- Avoids near-duplicate submissions  
- Directly probes the most plausible improvement direction  

---

## 3 Key Assumptions

1. The **x1 active band [0.62, 0.76]** is correctly identified and contains the relevant low-x2 cluster.

2. The two existing low-x2 observations are the only reliable local evidence; they are insufficient to fit a stable GP.

3. A **geometric maximin probe** inside the unsampled strip yields the highest information gain per expensive evaluation under extreme data scarcity.

---

## Research Backing

### Rasmussen & Williams (2006)  
*Gaussian Processes for Machine Learning*  
Explains the instability of GP hyperparameter estimation when sample size is extremely small.

### Bull (2011)  
*Convergence Rates of Efficient Global Optimisation Algorithms*  
Supports conservative, structure-aware exploration when model uncertainty is high.

### Srinivas et al. (2010)  
*Gaussian Process Optimization in the Bandit Setting*  
Advocates exploration pressure when uncertainty is large — but only when uncertainty is reliable. When it is not, geometry is safer.

---

## Explorative Principle

The data show:

- Two low-x2 points inside the x1 active band  
- A visible gradient toward **lower x2**  
- No data below those two points  

With only two points, any GP fit is underdetermined:

- Length scale and noise parameters cannot be estimated reliably  
- The model risks collapsing or overfitting  
- Acquisition functions may select near-duplicates  

Instead, we adopt a **pure geometric maximin probe**:

- Define a conservative low-x2 strip below existing data  
- Generate a dense candidate grid  
- Select the point with **maximum minimum distance** to all training data  

This ensures:

- Maximum information gain per query  
- Honest exploration of the unsampled strip  
- No reliance on unstable probabilistic modeling  

---

## Why This Strategy Is Ideal for f2

- Evaluations are expensive and few remain  
- GP attempts (1D and 2D) collapsed due to insufficient data  
- The structural gradient toward lower x2 is clear  
- The unsampled strip is the most plausible improvement direction  

The maximin probe:

- Avoids model-based hallucination  
- Maximises spatial information gain  
- Tests the structural hypothesis directly  

---

## Tech Stack

- `numpy` — vectorised numerical operations  
- `scipy.spatial.distance.cdist` — pairwise Euclidean distance computation  
- `pandas` (optional) — structured diagnostics  

No GP fitting is performed.

---

## Hyperparameters and Settings

### Structural Regions

- **x1 active band:** [0.62, 0.76]  
- **Search range within band:** [0.64, 0.74]  
- **Unsampled low-x2 strip:** [0.01, 0.10]  

### Candidate Generation

- Grid resolution: 60 × 60 (3600 candidates)  

### Distance Guard

- Minimum distance: 0.07  
- Fallback: 0.05  

### Selection Rule

- **Maximin:** choose candidate with largest minimum distance to existing points  

---

## Rationale for Hyperparameters

- **min_dist = 0.07:**  
  Larger than prior 0.06 to ensure meaningful separation from the two nearby low-x2 points.

- **x2 upper bound = 0.10:**  
  Focuses strictly below existing low-x2 observations, aligned with observed gradient.

- **Grid resolution = 60:**  
  Dense enough for reliable spatial separation without unnecessary computation.

These values are chosen from spatial diagnostics, not automated tuning, to avoid overfitting tiny samples.

---

## Full Strategy Flow

1. Identify the x1 active band and the two low-x2 observations.
2. Define rectangular search box:
   - x1 ∈ [0.64, 0.74]  
   - x2 ∈ [0.01, 0.10]  
3. Generate dense uniform grid (3600 candidates).
4. Compute Euclidean distance from each candidate to all training points.
5. Filter candidates violating minimum distance guard (0.07).
6. Select candidate with largest minimum distance (maximin).
7. If none survive, relax guard to 0.05 and repeat.
8. Output:
   - Selected point  
   - Minimum distance value  
   - Top 5 maximin candidates  
   - Reference low-x2 points and their y values  

---

## Hypothesis Framework

### Core Assumptions

- The low-x2 cluster is informative.  
- The gradient continues toward lower x2.  
- Geometry is more reliable than probabilistic modeling at this sample size.  
- Anchoring x1 near the better low-x2 centroid is reasonable.

### Expected If Assumptions Hold

- The probe either improves the low-x2 best  
- Or decisively shows that performance does not increase toward x2 = 0  
- Structural uncertainty about the low-x2 strip is resolved

### Expected If Assumptions Break

- If x1–x2 interaction is strong, fixing x1 near centroid may miss the true 2D optimum  
- If function is noisy, the probe may be inconclusive  
- Either outcome still yields structural information for final exploitation decisions  

---

## Strategic Summary

- Avoid unstable GP fitting with n ≈ 2 in the critical region  
- Use geometry when uncertainty estimates are unreliable  
- Maximise spatial information gain  
- Directly test the most plausible improvement direction  
- Spend the expensive query where it answers the most important structural question

---

# f3 — Deterministic Gap Fill in the A–B Corridor Using Maximin Distance

## Objective of Submission
Place a single new query inside the largest unsampled region between the two best observed points of the expensive black-box function, using only geometric information from `X_train_3` and `y_train_3` and **no surrogate models**.

---

## 3 Key Assumptions

1. The two best observations define a meaningful corridor where improvement is most likely.
2. The function is smooth enough inside this corridor that a maximin point is informative.
3. Surrogate models are too unreliable at \( n \approx 24 \), so a deterministic geometric design is safer.

---

## Research Backing

### Academic Papers Supporting the Strategy
- Johnson et al. (1990) — *Min Distance Designs for Computer Experiments*, Technometrics  
- Santner, Williams, and Notz (2003) — *The Design and Analysis of Computer Experiments*, Springer  
- Loeppky, Sacks, and Welch (2009) — *Choosing the Sample Size of a Computer Experiment*, Technometrics  

These works establish **maximin distance designs** and **space-filling experimental design** as robust strategies when model-based surrogates are unreliable.

---

## Clear Explanation of the Explorative Principle with Function-Specific Rationality

The two best points in `X_train_3` and `y_train_3` lie on a narrow band in **x2 and x3**. This band has **never been sampled between them**. Instead of relying on noisy surrogate predictions, the strategy uses a **maximin design** inside this corridor.

A Sobol sequence generates many candidate points. For each candidate, the **minimum distance to all existing observations** is computed. The candidate with the **largest minimum distance** is selected.

This approach fills the **largest unsampled gap** in the region where improvement is structurally plausible.

---

## Black-Box Optimization Competition

### Name of the Competition
NeurIPS 2020 Black-Box Optimization Challenge

### Winning Teams
Several high-performing teams used **maximin and space-filling designs within trusted subregions** when surrogate models were unreliable. This strategy follows the same principle.

---

## Why This Strategy Is Ideal for My Function

Function **f3** has an extremely **low signal-to-noise ratio**, and all surrogate models collapse in **x3**.

The two best points define a **corridor in x2 and x3** that has never been sampled. The deterministic gap-fill approach uses only this structural evidence and avoids all surrogate failure modes. It is therefore the **safest and most defensible query for f3**.

---

## Justification Based on Expensive Function Evaluations

Each evaluation is costly. Surrogate models cannot detect improvements of size **0.004** when their **leave-one-out error is about 0.08**.

The deterministic gap fill guarantees that the new query lies in the **most informative unsampled region between the two best points**. If this region does not improve, the function is likely **plateaued**.

---

## Tech Stack

### Libraries and Frameworks Used
- **NumPy** — numerical operations and distance calculations  
- **SciPy Sobol sequence generator** — quasi-random candidate generation  

---

## Hyperparameters and Settings

### List of Hyperparameters
- Corridor bounds for **x2**
- Corridor bounds for **x3**
- Minimum distance threshold **δ_min**
- Number of **Sobol candidates**

### Recommended Initial Values and Reasoning
- **x2 ∈ [0.65, 0.88]** — both best points lie in this range  
- **x3 ∈ [0.32, 0.42]** — both best points lie in this range  
- **δ_min = 0.07** — enforces novelty and prevents near duplicates  
- **Sobol candidates = 2048** — dense coverage with low computational cost  

---

## Hyperparameter Tuning Method Used
No tuning is performed. All values are **directly determined from structural evidence in the data**, ensuring that the design remains deterministic and independent of unstable surrogate fitting.

---

## Entire Flow of the Strategy

### Step-by-Step Exploration Process

1. Identify the two best observations in `X_train_3` and `y_train_3`.
2. Define a **corridor in x2 and x3** spanning between these two points.
3. Generate **Sobol candidates** in the unit cube.
4. Rescale the **x2 and x3 coordinates** of these candidates to lie within the corridor.
5. Compute the **minimum distance** from each candidate to all existing observations.
6. Filter out candidates that violate the **minimum distance threshold**.
7. Select the candidate with the **largest minimum distance** (maximin point).
8. Return this point as the **next query**.

---

## Hypothesis Framework

### Core Assumptions
- The corridor contains any remaining improvement.
- The function is smooth inside the corridor.
- Surrogate models are unreliable at \( n \approx 24 \).

### What Is Expected if Assumptions Hold
- The selected point provides **meaningful new information**.
- If it improves the objective, the corridor likely contains a **ridge structure**.
- If it does not improve, the corridor is likely **plateaued**.

### What Is Expected if Assumptions Break
- The true maximum lies **outside the corridor**.
- The function may be **irregular inside the corridor**.
- The deterministic approach may be **too conservative**.
  
---

# f4 — Neighbourhood Exploitation via GP Expected Improvement with Shrinking Radius

## Objective of Submission
Select a single query point for **f4** that exploits the residual uncertainty of a **Gaussian Process (GP)** model in a small neighbourhood around the current best observation, using **Expected Improvement (EI)** as a principled acquisition function.

---

## 3 Key Assumptions

1. The GP surrogate is globally converged and does not believe any distant region has a higher mean than the current best.
2. There is still **non-zero posterior variance** in the immediate neighbourhood of the best point, so local improvement is possible.
3. The function is **smooth enough locally** that nearby points share similar behaviour, making neighbourhood exploitation meaningful.

---

## Research Backing

### Academic Papers Supporting the Strategy
- Jones, Schonlau, and Welch (1998) — *Efficient Global Optimization of Expensive Black-Box Functions*, Journal of Global Optimization  
- Snoek, Larochelle, and Adams (2012) — *Practical Bayesian Optimization of Machine Learning Algorithms*, NeurIPS  
- Eriksson, Pearce, Gardner et al. (2019) — *Scalable Global Optimization via Local Bayesian Optimization (TuRBO)*, NeurIPS  

These works establish **Expected Improvement (EI)** and **trust-region style local Bayesian optimization** as efficient methods for expensive black-box optimisation.

---

## Clear Explanation of the Explorative Principle with Function-Specific Rationality

The exploration principle here is **local exploitation of uncertainty**, not global exploration.

The GP fitted on `X_train_4` and `y_train_4` has its **mean surface peaked at the current best**, but still exhibits **non-zero posterior variance nearby**.

Expected Improvement measures how much a new point could improve over the current best by combining:

- the **predicted mean** (exploitation)
- the **predictive uncertainty** (exploration)

By restricting candidates to a **small neighbourhood around the best point**, the algorithm focuses on the only region where the GP still assigns meaningful probability of improvement. This is an aggressive strategy because the rest of the domain is ignored in favour of **dense local search**.

---

## Black-Box Optimization Competition

### Name of the Competition
NeurIPS 2020 Black-Box Optimization Challenge

### Winning Teams
Top teams, including contributors to **Optuna** and **NVIDIA RAPIDS**, used **EI-based exploitation phases** after exploration budgets were exhausted, often combined with **local trust regions similar to TuRBO**.

---

## Why This Strategy Is Ideal for My Function

For **f4**, the **GP Upper Confidence Bound** has already collapsed to the known best point, meaning the model has **no global optimism left**.

However, the GP **variance at the best point remains non-zero**, indicating **residual local uncertainty**.

Therefore, the only credible path to improvement is **local neighbourhood search**. A shrinking radius around the best observation, combined with dense sampling and EI optimisation, directly targets this uncertainty rather than wasting evaluations on distant regions that the GP already considers suboptimal.

---

## Justification Based on Expensive Function Evaluations

Each evaluation of **f4** is expensive, so queries must be **highly targeted**.

Global exploration at this stage is unlikely to succeed because the GP does not support the hypothesis that distant regions are better.

By focusing on a **tight neighbourhood around the best point** and using EI, the strategy maximises the probability that the next evaluation will either:

- improve the best value, or  
- confirm that the current best is effectively the ceiling.

This makes efficient use of the remaining query budget.

---

## Tech Stack

### Libraries and Frameworks Used
- **NumPy** — numerical operations
- **SciPy** — optimisation and distance computations
- **SciPy QMC (Sobol)** — quasi-random candidate sampling
- **SciPy Stats** — normal distribution functions
- **Scikit-learn** — `GaussianProcessRegressor` and kernel implementations

---

## Hyperparameters and Settings

### List of Hyperparameters
- Matérn kernel smoothness parameter **ν**
- Length-scale bounds per dimension
- Noise level bounds
- Number of GP optimizer restarts
- Neighbourhood radius around the current best point
- Number of Sobol candidates in the local search box
- Minimum distance guard to avoid near-duplicate queries
- EI **ξ (xi)** parameter controlling exploration bias

### Recommended Initial Values and Reasoning

- **Matérn ν = 2.5**  
  Standard choice for moderately smooth functions in Bayesian optimisation.

- **Length-scale bounds = [0.05, 10.0] per dimension**  
  Allows both short-range and long-range variation without forcing parameter saturation.

- **Noise bounds = [1e-8, 0.5]**  
  Covers low to moderate noise levels.

- **GP optimizer restarts = 25**  
  Reduces risk of poor local optima in hyperparameter estimation.

- **Neighbourhood radius = 0.12**  
  Defines a tight region around the best point smaller than earlier trust regions.

- **Number of Sobol candidates = 100,000**  
  Ensures dense coverage of the local region.

- **Minimum distance guard = 0.02**  
  Prevents near duplicates while remaining inside the neighbourhood.

- **EI ξ = 0.0**  
  Pure exploitation with no additional exploration bonus.

---

## Hyperparameter Tuning Method Used

Hyperparameters are **not tuned via cross-validation**. Instead, they are chosen based on **standard Bayesian optimisation practice** and prior experiments.

The GP hyperparameters themselves are optimized by **maximizing the log marginal likelihood** using multiple random restarts.

---

## Entire Flow of the Strategy

### Step-by-Step Exploration Process

1. Combine `X_train_4` and `y_train_4` and identify the **current best observation**.
2. Fit a **Gaussian Process with a Matérn kernel and ARD** on all training data.
3. Compute the GP **mean and variance at the best point** to confirm residual uncertainty exists.
4. Define a **local hypercube neighbourhood** around the best point using the chosen radius and clip to `[0, 1]^d`.
5. Generate a large number of **Sobol candidates** within this box.
6. Retain only candidates inside the **L2 sphere** around the best point.
7. Apply a **minimum distance guard** to remove candidates too close to existing observations.
8. Compute **Expected Improvement** for all remaining candidates.
9. Select the candidate with the **highest EI**.
10. Perform **local gradient-based EI optimisation** within the same neighbourhood.
11. Compare refined and Sobol candidates and select the **best final query point**.

---

## Hypothesis Framework

### Core Assumptions

- The GP surrogate is **reasonably calibrated near the best point**.
- The function behaves **smoothly within the local neighbourhood**.
- The **global optimum lies near the current best region**.

### What Is Expected if Assumptions Hold

- The selected point either **improves the current best value** or  
- Provides strong evidence that the **local region is saturated**.

Expected Improvement will remain **non-negligible**, justifying a focused exploitation step.

### What Is Expected if Assumptions Break

- If the GP is **miscalibrated** or the function contains **sharp discontinuities**, EI may mislead the search.
- The algorithm may fail to find improvement even if **better regions exist elsewhere**, potentially leading to the incorrect conclusion that the current best is the ceiling.

---

# f5 — GP Mean Maximisation with Adaptive Corner Floors for Monotone Ridge Exploitation

## Objective of Submission
Propose a single new query for **f5** by maximising the **Gaussian Process posterior mean** inside a structurally motivated **corner region**, while **adaptively relaxing per-dimension floors** and enforcing a **distance guard** so that the point is both high-value and genuinely novel.

---

## 3 Key Assumptions

1. The function behind **f5** is **monotone increasing in all dimensions** toward the high corner of the domain.
2. The best observed values so far lie near a **corner ridge**, but the true maximum has not yet been sampled.
3. A **Gaussian Process fitted on log-transformed outputs** preserves the **ranking of candidates** inside the corner even if it underestimates absolute values.

---

## Research Backing

### Academic Papers Supporting the Strategy
- Jones, Schonlau, and Welch (1998) — *Efficient Global Optimization of Expensive Black-Box Functions*, Journal of Global Optimization  
- Hutter, Hoos, and Leyton-Brown (2011) — *Sequential Model-Based Algorithm Configuration (SMAC)*, LION 5  
- Eriksson, Pearce, Gardner, Turner, and Poloczek (2019) — *Scalable Global Optimization via Local Bayesian Optimization (TuRBO)*, NeurIPS  

---

## Clear Explanation of the Explorative Principle with Function-Specific Rationality

The exploration principle here is **aggressive local exploitation** inside a **structurally confirmed corner region**.

Earlier analysis for **f5** indicates that **high outputs occur when all coordinates are large**, suggesting a **monotone ridge toward the high corner of the domain**.

Instead of exploring the entire domain, this strategy:

1. Restricts candidates to a **tight hypercube near the corner**.
2. Applies **per-dimension floors** so each coordinate remains close to its **best observed value**.
3. **Relaxes the floors adaptively** if the candidate pool becomes empty.
4. Applies a **distance guard** to prevent near-duplicate evaluations.

Within the filtered candidate pool, the **Gaussian Process posterior mean** is used as the acquisition function, reflecting **pure exploitation** of the most promising region.

---

## Black Box Optimization Competition

### Name of the Competition
NeurIPS 2020 Black-Box Optimization Challenge

### Winning Team
**Squirrel AI** and other top teams used **SMAC-style intensification phases**, where after identifying a promising region the search focuses locally and often uses **GP mean maximisation or low-ξ Expected Improvement** inside a trust region.

---

## Why This Strategy Is Ideal for My Function

For **f5**, the best observations cluster near a **high corner of the domain**, and there is no evidence suggesting alternative peaks elsewhere.

Since each evaluation is **expensive**, spending queries far from this region would likely waste budget.

This strategy focuses on the **empirically best region** while maintaining flexibility:

- **Adaptive floors** keep candidates near the ridge but avoid over-restricting the search.
- **Distance guards** enforce novelty.
- **GP mean ranking** provides a smooth ordering of candidates within the region.

---

## Justification Based on Expensive Function Evaluations

Because evaluations are expensive, each query must have a **high probability of improvement**.

Global exploration would waste budget on regions that the data already indicates are poor.

By focusing on the **corner region** and using **dense Sobol sampling**, this strategy concentrates effort where the **probability of improvement is highest**, while **adaptive floors and distance guards prevent overfitting to a single point**.

---

## Tech Stack

### Libraries and Frameworks Used
- **NumPy** — numerical operations
- **SciPy QMC** — Sobol sequence candidate generation
- **Scikit-learn** — `GaussianProcessRegressor`, Matérn kernel, `WhiteKernel`

---

## Hyperparameters and Settings

### List of Hyperparameters

- Corner lower bound for all dimensions
- Per-dimension floors
- Number of Sobol candidates
- Minimum distance guard
- Log-transform shift
- Matérn ν parameter
- Length-scale bounds
- Noise-level bounds
- Number of GP optimizer restarts

---

### Recommended Initial Values and Reasoning

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Corner lower bound | `0.92` | Restricts search to extreme high region where best points lie |
| Dimension floors | `[0.97, 0.95, 0.95, 0.95]` | Keeps each coordinate near best observed values |
| Floor relaxation | `0.03` | Slight relaxation if strict floors empty the candidate pool |
| Sobol candidates | `10000` | Dense coverage of the small corner hypercube |
| Minimum distance guard | `0.02` (fallback `0.01`) | Ensures novelty while remaining local |
| Log shift | `1.0` | Stabilises GP fitting on positive outputs |
| Matérn ν | `2.5` | Standard choice for moderately smooth functions |
| Length-scale bounds | `[0.01, 5.0]` | Allows short- and long-range variation |
| Noise-level bounds | `[1e-6, 0.05]` | Stabilises GP noise estimation |
| Optimizer restarts | `30` | Reduces risk of poor kernel hyperparameter optima |

---

## Hyperparameter Tuning Method Used

Hyperparameters such as **bounds and kernel smoothness** are chosen based on **standard Bayesian optimisation practice** and prior experiments.

The Gaussian Process kernel parameters are then tuned by **maximising the log marginal likelihood** with multiple restarts, which is the standard training procedure for GP regression.

---

## Entire Flow of the Strategy

### Step-by-Step Exploration Process

1. Load `X_train_5` and `y_train_5` and identify the **current best observation**.
2. Generate **Sobol candidates** inside the tight corner region `[corner_lower, 1.0]^4`.
3. Apply **per-dimension floors** to keep each coordinate above its threshold.
4. If the pool becomes empty, **relax floors slightly** and regenerate candidates.
5. Apply a **distance guard** to remove candidates too close to existing observations.
6. If this also empties the pool, **reduce the guard radius** and regenerate candidates.
7. Apply a **log transformation** to the outputs.
8. Fit an **ARD Matérn Gaussian Process** on the transformed data.
9. Optionally compare acquisitions (**mean, EI, UCB**) to verify consistency in this region.
10. Compute the **GP posterior mean** for all candidates.
11. Select the candidate with the **highest posterior mean**.
12. Invert the log transformation and report the predicted original-scale value and its **distance from the current best**.

---

## Hypothesis Framework

### Core Assumptions

- The function is **monotone increasing toward the high corner**.
- The best region has already been identified within the **corner hypercube**.
- The **Gaussian Process preserves ranking** inside this region.

---

### What Is Expected if Assumptions Hold

- The selected point **improves the current best** or produces a **near miss** very close to it.
- The result confirms that the **true maximum lies in the corner region**.
- The GP mean provides a **reliable ranking of candidates**.

---

### What Is Expected if Assumptions Break

- If the function is **not monotone** or has another peak away from the corner, the strategy may miss it.
- If the GP is **miscalibrated**, the posterior mean ranking may be misleading.
- If floors remain **too strict even after relaxation**, the search region could still be too narrow and miss slightly lower but better-aligned points.
     
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

---

## f7 — Trust Region Ensemble Exploitation with KNN Anchored Local Search

### Objective of Submission
The objective is to submit a single Week 10 query for **f7** that stays strictly inside the confirmed high-value basin. This is achieved by combining:

- a **TuRBO-style trust region**
- an **ensemble surrogate model**
- empirical structural constraints on **x1** (upper cap) and **x6** (lower floor)

These constraints ensure the candidate respects the strongest signals identified during exploratory data analysis (EDA).

---

## Key Assumptions

1. **Localised Global Maximum**  
   The global maximum lies inside the Mapper-identified high-value component, which is small and isolated.

2. **Local Smoothness**  
   Within this component the function behaves smoothly, meaning a trust region around the centroid of the top observations is sufficient to capture the peak.

3. **Dimension Importance Structure**  
   - **x3 and x6** are structurally important variables.
   - **x1** is important but can be constrained with a simple empirical cap instead of full re-optimisation.

---

## Research Backing

The strategy is grounded in **local Bayesian optimisation with trust regions and ensemble surrogates**.

Trust-region methods like **TuRBO** demonstrate that shrinking the search to a local hyper-rectangle around the incumbent solution dramatically improves sample efficiency when the maximum is localised.

Ensemble surrogate models are widely used in practical black-box optimisation because they reduce model bias and capture different aspects of the response surface.

### Supporting Academic Papers

- Eriksson, D., Pearce, M., Gardner, J., Turner, R., & Poloczek, M. (2019).  
  *Scalable Global Optimization via Local Bayesian Optimization (TuRBO).* NeurIPS.

- Jones, D. R., Schonlau, M., & Welch, W. J. (1998).  
  *Efficient Global Optimization of Expensive Black-Box Functions.* Journal of Global Optimization.

- Ozaki, Y. et al. (2020).  
  *Optuna: A Next-generation Hyperparameter Optimization Framework.* NeurIPS Bayesmark competition report.

---

## Explorative Principle and Function-Specific Rationality

The key principle is **aggressive exploration within a small, well-supported region** rather than exploration across the full domain.

Evidence from manifold learning methods supports this:

- **Mapper** reveals four disconnected components.
- The **global maximum lies in a two-node component** clearly separated from others.
- **Isomap embeddings** show that the top observations cluster tightly.

Leaving this cluster is therefore highly likely to produce low values.

The strategy therefore:

1. Centres a **trust region** on the centroid of the top four observations.
2. Uses a **width of 0.20 per dimension**, large enough to include the cluster but small enough to exclude most of the domain.
3. Generates many **Sobol candidates** inside this region.
4. Scores candidates using an **ensemble surrogate** composed of:
   - Gaussian Process (GP)
   - Support Vector Regression (SVR)
   - K-Nearest Neighbours (KNN)
   - Gradient Boosted Trees (XGBoost)

The acquisition function is:
Acquisition = Ensemble Mean + 1.5 × Inflated Variance


Variance is inflated by a factor of **3** to maintain moderate exploration.

Additional structural constraints:

- **x1 ≤ cluster_max(x1) + 0.02**
- **x6 ≥ cluster_min(x6) − 0.02**

These align the search with the strongest **partial correlation and SIR signals**.

---

## Black Box Optimization Competition

This strategy style appeared in the:

**NeurIPS 2020 Black Box Optimization Challenge (Bayesmark Track)**

Top performing teams used **TuRBO-style trust regions combined with ensemble surrogates** during the intensification phase once a promising basin was discovered.

### Winning Teams
Teams such as **Optuna** used TuRBO variants with multi-model surrogates and Thompson sampling once global exploration identified a local basin.

---

## Why This Strategy Fits f7

Evidence from the exploratory analysis indicates a **single strong basin structure**:

- **Mapper:** four disconnected components
- **Global optimum:** located within a two-node component
- **Isomap:** tight clustering of top four observations
- **SIR:** dominant direction defined primarily by **x3 and x6**
- **Partial correlations:** strong negative signal for **x1**

Implications:

- The function likely has **one dominant localised basin**
- Exploration outside this basin is wasteful

Therefore:

- The **trust region centred on the top-four centroid** aligns with the geometry of the data.
- The **ensemble surrogate** captures nonlinear local structure missed by earlier single-model GP attempts.
- The **empirical cap on x1** and **floor on x6** ensure candidates respect observed structural constraints.

---

## Justification Under Expensive Evaluations

Each evaluation of **f7** is expensive, so the strategy must maximise improvement probability per query.

Previous behaviour illustrates the risk of global exploration:

- **Week 9 GP variance exploration** left the high-value component.
- The resulting query produced a **low response value**.

The proposed strategy instead:

- Constrains the search to the **only known high-value basin**
- Uses an ensemble model to reduce prediction bias
- Encourages novelty through a **variance-augmented acquisition function**
- Uses a **distance guard** to prevent redundant sampling

This provides a **cost-effective exploitation strategy**.

---

## Tech Stack

Implementation uses widely adopted scientific computing libraries:

- **NumPy** — numerical array operations
- **SciPy** — Sobol low-discrepancy sequence generation
- **Scikit-learn** — GP, SVR, and KNN models
- **XGBoost** — gradient boosted tree surrogate

These are standard tools in **black-box optimisation research**.

---

## Hyperparameters and Settings

### Trust Region

| Parameter | Value | Rationale |
|---|---|---|
| Trust region centre | Centroid of top 4 observations | Defines basin centre |
| Trust region width | 0.20 per dimension | Covers cluster while excluding most domain |

### Ensemble Models

| Model | Key Settings |
|---|---|
| Gaussian Process | Matérn 2.5 kernel with ARD |
| SVR | RBF kernel, C = 5 |
| KNN | k = 4, distance weighted |
| XGBoost | 200 trees, max depth = 4 |

### Ensemble Weighting

- **5-fold cross validation**
- **Inverse MSE weighting**

### Acquisition Parameters

| Parameter | Value |
|---|---|
| Variance inflation factor | 3 |
| Acquisition scaling | 1.5 × variance |
| Sobol candidate count | 4096 |

### Empirical Structural Constraints

| Parameter | Rule |
|---|---|
| x1 cap | max(top4 x1) + 0.02 |
| x6 floor | min(top4 x6) − 0.02 |

Hyperparameters were **not grid-searched** but selected using:

- empirical behaviour from earlier weeks
- guidance from optimisation literature

---

## Full Strategy Workflow

1. Load `X_train_7` and `y_train_7`.
2. Identify the **top four observations** by response value.
3. Compute their **centroid**.
4. Define a **trust region** with width 0.20 per dimension.
5. Clip the region to the **unit cube [0,1]^7**.
6. Generate **4096 Sobol candidates** inside the region.
7. Apply a **distance guard** to avoid points near existing samples.
8. Apply **empirical constraints**:
   - `x1 ≤ cluster_max + 0.02`
   - `x6 ≥ cluster_min − 0.02`
9. Train the ensemble models:
   - GP
   - SVR
   - KNN
   - XGBoost
10. Perform **5-fold CV** to compute inverse-MSE weights.
11. Refit models on full training data.
12. Predict candidate outputs.
13. Compute:
    - **ensemble mean**
    - **ensemble variance**
14. Calculate acquisition value: Acquisition = mean + 1.5 × (3 × variance)

15. Select the candidate with the **highest acquisition value**.
16. Submit this point as the **Week 10 query**.

---

## Hypothesis Framework

### Core Assumptions

- The high-value basin is **unique and contained within the trust region**
- The **centroid of the top four observations** is a reliable anchor
- The **ensemble surrogate** is accurate within this region

### Expected Outcome if Assumptions Hold

- The new query lies **near the true optimum**
- **x3 and x6** align with the dominant SIR direction
- **x1** remains in a safe empirical range
- Observed **y value improves or approaches the maximum**

### Failure Modes

Possible assumption violations include:

- An **undiscovered basin** outside the trust region
- **Overfitting** of ensemble models due to limited samples
- Miscalibrated acquisition function

If this occurs, the follow-up strategy would:

- shrink the trust region further
- modify ensemble composition
- adjust variance scaling.

---

## Method Summary

**Machine Learning Method**

Ensemble surrogate regression combining:

- Gaussian Process
- Support Vector Regression
- K-Nearest Neighbours
- Gradient Boosted Trees

inside a **TuRBO-style trust region**.

**Submission Objective**

Select a single query that exploits the known **high-value basin** of f7 while maintaining novelty.

**Rationale**

- Single GP models previously misestimated variable importance.
- Ensembles capture **local nonlinear structure** more reliably.
- The trust region prevents wasteful exploration outside the basin.

**Exploration Mechanism**

1. Generate many candidates inside the trust region.
2. Score them using **ensemble mean + variance acquisition**.
3. Select the **highest scoring candidate**.

This ensures exploration remains **aggressive but geographically constrained to the most promising region**.

---

## f8 — Incumbent-Fixed Inactive Dimension LogEI with Tightened Active Subspace Bounds

### Objective of Submission
Propose a single **Week 10 query for f8** by exploiting the confirmed **high-value basin around the incumbent**, while preventing the acquisition optimiser from drifting into boundary corners on inactive dimensions.

The strategy:

- Searches **only the active subspace** `(x1, x3, x7)`
- **Fixes all inactive dimensions** at their incumbent values
- Uses **Log Expected Improvement (LogEI)** for stable acquisition optimisation near the incumbent

---

## Key Assumptions

1. **Local Basin Containment**  
   The global maximum lies inside the **Mapper-identified high-value component** and can be reached through local exploitation.

2. **Active Dimension Structure**  
   Only **x1, x3, and x7** are truly active.  
   All other dimensions should remain **fixed at the incumbent values** to prevent optimisation collapse into irrelevant corners.

3. **Small Improvement Margin**  
   The improvement margin above **9.948** is small, making **LogEI numerically more stable** than standard Expected Improvement.

---

## Research Backing

The strategy is supported by literature on **active subspace optimisation**, **sparse Bayesian optimisation**, and **numerically stable acquisition functions**.

### Supporting Academic Papers

- Eriksson, D., & Jankowiak, M. (2021).  
  *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO).* UAI.

- Hellsten, L. et al. (2025).  
  *Group Testing Bayesian Optimization (GTBO).* NeurIPS.

- Ament, S. et al. (2023).  
  *Log Expected Improvement for Bayesian Optimization.* NeurIPS.

- Bull, A. (2011).  
  *Convergence Rates of Efficient Global Optimization Algorithms.* JRSS B.

These works show that once **inactive dimensions are identified**, fixing them at incumbent values significantly improves convergence.

---

## Explorative Principle and Function-Specific Rationality

The explorative principle is **local exploitation inside the active subspace**.

During **Week 9**, allowing the optimiser to move inactive dimensions freely caused **corner collapse**:

- `x2 = 0`
- `x4 = 0`

These values are structurally inconsistent with the incumbent solution.

EDA reveals:

- The high-value region is a **small isolated topological component**
- The incumbent lies **inside this component**

Therefore the correct strategy is:

1. Search only **active dimensions `(x1, x3, x7)`**
2. **Fix all other dimensions** at incumbent values
3. Focus optimisation effort on the **true improvement directions**

This prevents optimisation drift and concentrates search power where improvement is possible.

---

## Black Box Optimization Competition

### Competition
**NeurIPS 2020 Black-Box Optimization Challenge (Bayesmark Track)**

### Winning Teams
Teams including **Duxiaoman Financial AI** and **NVIDIA Research** used **active-subspace-aware GP models** where inactive dimensions were fixed at incumbent values once identified.

---

## Why This Strategy Is Ideal for f8

The **Week 9 failure** is explained by **inactive dimension drift**.

Evidence:

- **x4 partial correlation:**  
  `r = -0.538`, `p = 0.0003`
- Collapsing `x4 → 0` was a **structural mistake**

Additional evidence:

- **SIR:** indicates the function is near a **plateau but not at the ceiling**
- **Mapper:** high-value cluster is **isolated**
- **Isomap:** best points are **geometrically central**, not boundary outliers

Together, these results strongly support **local exploitation with inactive dimensions fixed**.

---

## Justification Under Expensive Function Evaluations

Only **three queries remain**, making global exploration inefficient.

Fixing inactive dimensions reduces the optimisation problem from:
8 dimensions → 3 dimensions


Benefits:

- Dramatically improves **sample efficiency**
- Prevents optimisation from drifting into **irrelevant corners**
- Focuses search effort near the **known high-value basin**

Thus the expected value of a **local query near the incumbent** is far higher than any global exploration move.

---

## Tech Stack

Libraries and frameworks used:

- **NumPy** — numerical operations
- **SciPy** — Sobol sampling and L-BFGS-B optimisation
- **Scikit-learn**
  - `GaussianProcessRegressor`
  - `Matern`
  - `WhiteKernel`
  - `ConstantKernel`

---

## Hyperparameters and Settings

### Hyperparameter List

- GP kernel type
- Length-scale bounds per dimension
- Inactive-dimension fixing policy
- Active-dimension search bounds
- Sobol candidate count
- Perturbation candidate count
- Minimum distance guard
- LogEI acquisition parameters
- L-BFGS-B warm start count
- GP hyperparameter restart count

---

### Recommended Initial Values

| Parameter | Value | Reason |
|---|---|---|
| GP kernel | Matérn ν = 2.5 | Smooth and differentiable |
| x1, x3 length-scale bounds | `[0.02, 0.20]` | Hit short bounds in Week 9 |
| x4, x5 bounds | moderate | Weak but non-zero signal |
| x6, x8 bounds | long | Confirmed inactive |
| Inactive dims | fixed at incumbent | Prevent corner collapse |
| x1 search range | `[0.05, 0.40]` | EDA-supported region |
| x3 search range | `[0.02, 0.35]` | EDA-supported region |
| x7 search range | `[0.05, 0.55]` | EDA-supported region |
| Sobol candidates | 10,000 | Dense 3D coverage |
| Perturbation candidates | 1,000 | Explore local neighbourhood |
| Minimum distance guard | 0.04 | Ensure novelty |
| Acquisition | LogEI | Stable near incumbent |
| L-BFGS-B warm starts | 25 | Refine top candidates |
| GP restarts | 30 | Robust kernel fitting |

---

## Hyperparameter Tuning Method

Gaussian Process hyperparameters are fitted using:

- **Multi-start L-BFGS-B optimisation**
- **Maximisation of GP log marginal likelihood**
- **Asymmetric length-scale bounds**

This ensures stable and robust GP model fitting.

---

## Full Strategy Workflow

1. Load `X_train_8` and `y_train_8`.
2. Identify the **incumbent**  
   `row 42`, `y = 9.948689`.
3. **Standardise y-values** for stable GP fitting.
4. Build an **ARD Matérn GP** with tightened asymmetric bounds.
5. Fit the GP using **30 restarts**.
6. Verify length-scale structure:
   - **x1, x3:** short scales
   - **x6, x8:** long scales
7. Generate **10,000 Sobol candidates** in the **3D active subspace `(x1, x3, x7)`**.
8. Fix inactive dimensions `(x2, x4, x5, x6, x8)` at incumbent values.
9. Generate **1,000 perturbation candidates** around the incumbent.
10. Combine candidate sets and clip to `[0,1]^8`.
11. Apply a **minimum distance guard of 0.04**.
12. Compute **LogEI** for all candidates.
13. Select the **top 25 candidates** as optimisation warm starts.
14. Run **L-BFGS-B optimisation** from each warm start with inactive dimensions locked.
15. Select the candidate with the **highest LogEI**.
16. Output this candidate as the **Week 10 query**.

---

## Hypothesis Framework

### Core Assumptions

- The high-value region forms a **single isolated basin**
- Inactive dimensions should remain **fixed at incumbent values**
- Improvement margins are small, making **LogEI appropriate**

---

### Expected Outcome if Assumptions Hold

- The selected point lies inside the **Mapper component containing the global maximum**
- The GP correctly ranks candidates in the **active subspace**
- The new evaluation will likely exceed **9.90**, and may exceed **9.948**

---

### Expected Outcome if Assumptions Break

Possible failure modes:

- Another basin exists **outside the incumbent region**
- Hidden **interactions in inactive dimensions**
- **GP miscalibration** causing LogEI misestimation

In these cases, improvement may not occur despite correct optimisation within the assumed structure.