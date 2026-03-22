## Function-Specific Strategies


## f1 — Quadratic Ridge Extrapolation with Copula-Validated Bracketing

### Objective of Submission
Propose a single **high-value query for f1** by exploiting a **one-dimensional ridge in x2**, using a **quadratic model on log-transformed outputs**, while keeping the number of expensive evaluations as low as possible.

---

## Key Assumptions

1. **Dominant Ridge in x2**  
   The function has a dominant **one-dimensional ridge along x2** when **x1 is held near its best value**.

2. **Local Quadratic Structure**  
   In the high-value region, the relationship between **x2 and log10(y)** is smooth and well approximated by a **concave quadratic**.

3. **Correct Basin Already Found**  
   The current best observations already lie inside the correct **basin of attraction**, so **aggressive local exploration along x2** is more valuable than global exploration.

---

## Research Backing

### Supporting Academic Papers

- Mockus, Tiesis, Zilinskas (1978)  
  *The Application of Bayesian Methods for Seeking the Extremum.*  
  Introduces the idea of **directional bracketing using local polynomial models**.

- Jones, Schonlau, Welch (1998)  
  *Efficient Global Optimization of Expensive Black-Box Functions.*  
  Discusses exploiting **dominant one-dimensional directions using low-degree polynomials**.

- Eriksson et al. (2019)  
  *Scalable Global Optimization via Local Bayesian Optimization (TuRBO).*  
  Demonstrates that **shrinking the search to a local trust region** dramatically improves sample efficiency once a basin is identified.

---

## Explorative Principle

The exploration strategy focuses on **aggressive local search along the empirically dominant dimension (x2)** rather than global exploration.

### Core Idea

1. **Fix x1 at its best observed value**  
   Gaussian Process models repeatedly show **x1 is flat near the best region**.

2. **Focus on x2**  
   Observed data show strong improvement as **x2 decreases**, suggesting a ridge.

3. **Stabilise scale with log transform**  
   Transform `y → log10(y)` to improve numerical stability.

4. **Fit a quadratic model**
   log10(y) = a x2² + b x2 + c


5. **Estimate ridge peak**
   x2_peak = -b / (2a)


6. **Query near this peak**, with a distance guard to avoid repeating previous evaluations.

This approach directly targets the **estimated ridge maximum**, rather than balancing exploration and exploitation via a high-dimensional surrogate.

---

## Black Box Optimization Competition

### Competition
**NeurIPS 2020 Black Box Optimization Challenge**

### Winning Team
The **Duxiaoman Financial AI Lab** reported using **explicit one-dimensional gradient-following and polynomial fits** when their surrogate identified a single active dimension.

The GP was used primarily for **confirmation**, while **polynomial fits guided the next query**.

---

## Why This Strategy Fits f1

Evidence suggests:

- A **single basin**
- A **clear gradient along x2**
- A **flat response along x1 near the optimum**

The Gaussian Process acquisition surface has been **nearly flat for several weeks**, meaning it provides **little directional guidance**.

A simple **quadratic fit in one dimension** is therefore more reliable than a high-dimensional surrogate with poorly identified length scales.

---

## Justification Under Expensive Evaluations

Function evaluations are expensive, so each new point must have a **high probability of improvement**.

EDA results show:

- A **single dense high-value node**
- Strong **dependence on x2**

Therefore:

- Global exploration is unlikely to outperform **focused ridge exploitation**
- A quadratic fit on **log10(y) vs x2** uses only a handful of points but directly estimates the **local maximum**

The **distance guard** ensures the query remains informative even if the quadratic estimate is slightly inaccurate.

---

## Tech Stack

Minimal dependencies are used to keep the method simple and robust.

- **NumPy** — polynomial fitting and numerical arrays
- **SciPy (`scipy.stats.norm`)** — Gaussian copula diagnostics

No Gaussian Process model or heavy ML framework is used.

---

## Hyperparameters and Settings

### Hyperparameter List

- `x1_fixed`
- `x1_radius`
- `credible_x2_range`
- `fallback_step`
- `min_dist_guard`

---

### Recommended Values

| Parameter | Value | Reason |
|---|---|---|
| x1_fixed | current best x1 | x1 is flat near optimum |
| x1_radius | 0.08 | selects tight local neighbourhood |
| credible_x2_range | (0.680, 0.730) | brackets ridge peak |
| fallback_step | 0.007 | conservative gradient step |
| min_dist_guard | 0.010 | avoid duplicates |

These values reflect the **bracketing strategy** described by Mockus.

---

## Hyperparameter Tuning Method

Hyperparameters are chosen via **empirical inspection**, not automated tuning.

Reasons:

- Dataset is extremely small
- Evaluations are expensive
- Structure is effectively **one-dimensional**

Early Bayesian optimization research relied heavily on **expert-defined bracketing intervals** rather than automated hyperparameter search.

---

## Strategy Workflow

### Step-by-Step Process

1. Identify the **current best observation** from `y_train_1`.
   
2. Record its `x1` and `x2`.
   
3. Select observations where: |x1 - x1_fixed| ≤ x1_radius
   
4. Compute: log10(y)
   
5. Fit quadratic: log10(y) = a x2² + b x2 + c

6. Check sign of `a`.
    
- If `a < 0` → compute peak
- x2_peak = -b / (2a)
- If `a ≥ 0` → use fallback step.
  
7. Check if `x2_peak` lies inside the **credible range**.

8. If valid → propose `x2_peak`.  
   Otherwise → use: x2_proposed = x2_best − fallback_step

9. Construct candidate point: next_x = [x1_fixed, x2_proposed]

10. Apply **distance guard**.  
If too close to an existing point, shift `x2` by `0.005`.

11. Return the candidate as the **recommended submission**.

---

## Hypothesis Framework

### Core Assumptions

- The high-value region is a **single ridge**
- **x1 is locally flat**
- `log10(y)` vs `x2` is **concave**

---

### Expected Outcome if Assumptions Hold

- Quadratic coefficient `a` will be **negative**
- Residuals will be **small**
- `x2_peak` lies inside the credible interval
- The new evaluation **improves the current best**

Repeated application should converge rapidly to the ridge centre.

---

### Expected Outcome if Assumptions Break

Possible signs of failure:

- Quadratic opens upward (`a ≥ 0`)
- Large residuals
- Peak estimate outside credible range

In this case:

- Use fallback step
- Reconsider ridge assumption
- Potentially reintroduce a surrogate model.

---

# Intuitive Interpretation of the Results

## Physical Picture

Imagine a **very narrow mountain ridge**.

- The ridge runs roughly **left–right** across a valley.
- Moving **along the ridge (x1)** keeps you at similar height.
- Moving **across the ridge (x2)** causes dramatic changes.

One step in the wrong direction drops you from **mountain-top to valley floor**.

This explains why the outputs span **200 orders of magnitude**.

---

## What the Numbers Mean

Each small decrease in `x2` produced **large improvements in log10 space**.

| X2 | y | Interpretation |
|---|---|---|
| 0.729 | 5e-15 | base of spike |
| 0.725 | 2e-09 | six orders better |
| 0.718 | 2e-08 | near ridge top |

Improvements are **decelerating**, which indicates the search is **approaching the peak**.

---

## Role of the Quadratic Model

The quadratic fits the curve through the best observed points and estimates the peak location.

Estimated peak: x2 ≈ 0.715

Predicted value: y ≈ 5e-08

This is about **2.6× better** than the current best.

The submission at **x2 = 0.705** is slightly conservative because the quadratic fit uses only a few points.

---

## Why x1 Does Not Matter

Across eleven weeks:

- GP models consistently assigned **x1 a very large length scale**
- This indicates the function changes **very little along x1**

Conceptually:

- The ridge runs **parallel to the x1 axis**
- Moving in x1 stays on the ridge
- Moving in x2 moves **toward or away from the peak**

---

# EDA Methods to Remove for f1

## Sliced Inverse Regression (SIR)

Remove it.

Bootstrap stability angle: 70.56°

A stable direction would be **near 0°**.

At 70°, bootstrap directions are nearly perpendicular, meaning the method **cannot agree with itself**.

SIR requires roughly: ≈ 10 observations per dimension per slice

With **20 points in 2D**, the result is statistically unreliable.

---

## Integrated Gradients

Remove it.

Integrated gradients measure sensitivity of a **neural network**.

With only **20 observations**, a neural network simply **memorizes the training data**.

The IG result incorrectly claimed **x1 dominates**, contradicting GP and empirical gradients.

Reliable neural attribution requires **hundreds of samples**.

---

## Isomap

Remove for decision-making.

Isomap builds a **neighbourhood graph** and computes **geodesic distances**.

With only **20 points**, the graph becomes fragmented.

Mapper already confirmed: 17 connected components from 19 nodes

The embedding therefore provides **no actionable insight**.

---

## K-means Clustering

Remove as a decision tool.

With 20 points, clusters reflect **sampling geometry**, not function structure.

Example:

- silhouette score suggested **k = 4**
- but clusters simply reflect **sample spacing**

Sorting the data table provides the same information.

---

## Mutual Information on Raw Y

Discard raw MI values:
X2 = 0.017
X1 = 0.000

kNN MI estimators require **50–100 observations** for stability.

At **N = 20**, estimates are dominated by noise.

Keep only the **copula-transformed MI**.

---

# EDA Methods to Add for f1

## Jackknife Correlation Stability

The most valuable addition.

Procedure:

1. Remove one observation
2. Recompute Spearman correlation
3. Repeat for all observations

Large variability indicates the signal depends on **one influential point**.

Rule of thumb:
Std dev > 0.10 → unstable signal

---

## Log10(Y) vs X2 Scatter Plot

A simple visual diagnostic.

Plot: log10(y) vs x2

Restrict to:
0.64 ≤ x1 ≤ 0.75
y > 0

Overlay the **quadratic fit**.

This plot reveals the **ridge immediately**.

---

## Pairwise Distance of Top Observations

Compute Euclidean distances among the **top five points by y**.

Report:

- minimum distance
- mean distance

For f1 this reveals a **tight cluster**, confirming the spike structure.

---

## Quadratic Residual Diagnostic

Whenever fitting the quadratic, report:

- **R²**
- **residuals**

Interpretation:

| R² | Interpretation |
|---|---|
| > 0.95 | quadratic reliable |
| 0.90–0.95 | moderate trust |
| < 0.90 | fallback strategy |

Residual diagnostics determine whether the **quadratic peak estimate can be trusted**.

---

## f2 – Nadaraya–Watson Slice Exploitation Along Confirmed High-Value X1 Band

### Objective of Submission

To choose a single evaluation point for **f2** by maximizing a **Nadaraya–Watson kernel regression prediction** along three empirically confirmed high-value `x1` locations, while enforcing a **minimum distance constraint** from all existing observations in `X_train_2`.

---

## 3 Key Assumptions

1. The **global optimum lies near the three known high-value `x1` values**  
   `(0.679, 0.695, 0.703)` in `X_train_2`.

2. The function is **locally smooth in `x2`** around these `x1` values, meaning kernel regression with a fixed bandwidth produces a meaningful estimate of the peak.

3. **Gaussian Process models are unreliable** for `f2` at the current query budget, so a **simple non-parametric smoother** provides a safer exploitation mechanism.

---

## Research Backing

### Academic Papers Supporting the Strategy

- **Nadaraya (1964)** and **Watson (1964)** introduced kernel regression as a non-parametric estimator for unknown functions.

- **Regis & Shoemaker (2007)** demonstrated that **distance-weighted surrogate models** can be effective for expensive black-box optimization once a promising region is identified.

- **Frazier (2018)** explains that after locating a high-value region, **local exploitation using focused surrogates** becomes the most sample-efficient strategy.

- **NeurIPS Black-Box Optimization competition reports** describe teams using **kernel-based surrogates as fallbacks** when Gaussian Processes became unstable.

---

## Explorative Principle

The explorative principle is **targeted local exploitation along a trusted one-dimensional slice**.

Instead of searching the entire 2D space:

1. Fix `x1` at **three empirically strong values**.
2. Scan `x2` across a **plausible high-value range**.
3. Use **Nadaraya–Watson kernel regression** to estimate the function value at each candidate point.
4. Select the candidate with the **highest predicted value** subject to a **distance constraint** from existing samples.

This is rational for **f2** because prior analysis revealed:

- A **narrow high-value band in `x1`**
- A **structured region in `x2`**

Therefore **refining `x2` within this band** is far more informative than global exploration.

---

## Black Box Optimization Competition

### Competition

**NeurIPS 2020 Black Box Optimization Challenge**

### Observed Strategies

Competition reports describe teams (including **JetBrains participants**) using:

- **Kernel regression**
- **Distance-weighted surrogates**

as fallback models when **Gaussian Process hyperparameters became unstable**, particularly in **low-dimensional structured problems**.

---

## Why This Strategy Is Ideal for f2

### Expensive Function Evaluation Context

Evaluations are expensive and the query budget is extremely small.

Empirical analysis revealed:

- A **single high-value cluster**
- **Three reliable `x1` locations**
- **Repeated Gaussian Process failure** during marginal likelihood optimization

Given this situation, the safest aggressive move is:

1. Fix `x1` at the **three best observed coordinates**.
2. Scan `x2` in a **focused interval**.
3. Predict values using a **stable kernel regression model**.
4. Enforce a **distance guard** to ensure each query adds new information.

---

## Tech Stack

Libraries used:

- **NumPy** — numerical operations and array manipulation  
- **SciPy** (`scipy.spatial.distance.cdist`) — computing distances between candidate points and `X_train_2`

No Gaussian Process libraries are required, keeping the method:

- Simple
- Robust
- Easy to debug

---

## Hyperparameters and Settings

### Hyperparameters

| Parameter | Description |
|---|---|
| `bandwidth h` | Gaussian kernel bandwidth |
| `x1 slices` | `[0.679, 0.695, 0.703]` |
| `x2 range` | `[0.800, 0.940]` |
| `x2 points per slice` | `30` |
| `minimum distance guard` | `0.04` |

---

### Recommended Initial Values

**Bandwidth**

```
h = 0.04
```

Chosen to remain **local relative to cluster spreads**, ensuring predictions are driven by nearby high-value points.

---

**X1 Slice Values**

```
[0.679, 0.695, 0.703]
```

These are **exactly the observed high-value `x1` coordinates**, ensuring exploration remains in the strongest region.

---

**X2 Range**

```
[0.800, 0.940]
```

Covers the entire empirically observed **high-value band** and both **unsampled gaps**.

---

**Grid Resolution**

```
30 points per slice
```

Provides enough resolution to locate predicted peaks without excessive computation.

---

**Distance Guard**

```
0.04
```

Chosen slightly below the observed spacing between cluster-B points so that new samples fall into **true gaps rather than duplicates**.

---

## Hyperparameter Tuning Method

Hyperparameters are chosen from **empirical geometry of the dataset**, not automated optimization.

Reasons:

- Dataset size is **too small for reliable hyperparameter tuning**
- Likelihood-based tuning **already failed for Gaussian Processes**
- Kernel regression heuristics based on **data scale and locality** are more stable

---

## Entire Flow of the Strategy

### Step-by-Step Exploration Process

1. Load all observations `X_train_2` and `y_train_2`.

2. Identify the **current best value and its coordinates**.

3. Define the **three fixed `x1` slices**:

```
[0.679, 0.695, 0.703]
```

4. Construct a **grid of `x2` values**:

```
x2 ∈ [0.800, 0.940]
```

with **30 evenly spaced points**.

5. For each `x1` slice, generate candidate points:

```
(x1_fixed, x2_grid)
```

6. For each candidate:

- Compute its **distance to the nearest point in `X_train_2`**
- Mark it **valid only if distance ≥ 0.04**

7. Apply **Nadaraya–Watson kernel regression** to predict the function value at each candidate.

8. Track the **candidate with the highest predicted value** that also satisfies the distance constraint.

9. Print diagnostics for transparency:

- candidate `x2`
- predicted value
- nearest distance
- guard status

10. Return the **best valid candidate** as the recommended submission.

---

## Hypothesis Framework

### Core Assumptions

- The **true optimum lies near one of the three `x1` slices**.
- The function is **smooth enough in `x2`** for kernel regression with `h = 0.04`.
- The high-value region lies **within the chosen `x2` interval**.

---

### Expected Outcome if Assumptions Hold

- Kernel predictions show **clear peaks along one slice**.
- The selected point lies in a **gap between known high-value observations**.
- The resulting evaluation **improves or matches the current best value**.

---

### Expected Outcome if Assumptions Break

Possible failure cases:

- Optimum lies at **an unseen `x1` location**
- Function is **highly irregular in `x2`**
- High-value region extends **outside `[0.800, 0.940]`**

In these cases the strategy may **miss the global optimum**.

---

# EDA Methods

## Remove

### Mutual Information

MI reported:

```
X2 = 0.000
```

every week.

Other diagnostics consistently showed **X2 matters**, meaning the **k-NN MI estimator is unreliable at N=20**.

Reason:

- Continuous MI estimation requires **N > 50**.

Conclusion:

```
Remove entirely
```

---

### Sliced Inverse Regression (SIR)

Bootstrap stability angle:

```
73.8°
```

This indicates **the leading direction is essentially random**.

Requirements for SIR:

```
N >> p
```

At `N = 20`, reliability is extremely poor.

Conclusion:

```
Remove
```

---

### Isomap 1D Embedding

Problems:

- Nearest-neighbour graph too sparse
- Geodesic distances unstable
- Duplicates information already visible in scatter plots

Conclusion:

```
Remove
```

---

## Keep

### K-Means Clustering

Most actionable EDA method.

Results:

```
k = 2
silhouette = 0.535
```

Clearly isolates the **high-value cluster**.

---

### Mapper Graph

Despite fragmentation it correctly:

- Identified **node 16 as the high-value cluster**
- Confirmed absence of a **global ridge**

---

### Integrated Gradients

Produced **stable importance ranking**

```
X1 > X2
```

across multiple weeks.

---

### Partial Correlation

Only method that **controls for the other variable**.

Confidence intervals are wide but **direction is stable**.

---

### Pairwise Scatter (coloured by Y rank)

Most readable diagnostic.

Should be generated **first every week**.

---

# EDA Methods to Add

## GP Posterior Mean and Sigma Heatmap

Plot both surfaces on a:

```
50 × 50 grid
```

Overlay training points.

Benefits:

- Shows predicted maxima
- Shows uncertainty
- Reveals surrogate bias

---

## Leave-One-Out Residual Plot

Procedure:

1. Remove one observation
2. Refit GP
3. Predict removed point
4. Plot residual

Large positive residuals reveal **systematic underestimation** in high-value regions.

---

## Empirical Variogram

Plot:

```
(y_i − y_j)^2
```

vs pairwise Euclidean distance.

Purpose:

- Estimate spatial correlation length
- Detect kernel mis-specification

---

## Convex Hull Coverage Plot

Plot the **convex hull of sampled points** in the unit square.

Colour by **Y rank**.

This reveals:

- sampled regions
- extrapolation gaps

---

## Nadaraya–Watson Predicted Surface

Now that

```
h = 0.04
```

produces calibration error `< 0.05`, generate a **dense NW surface plot** each week.

Advantages:

- No hyperparameter collapse
- No likelihood optimization
- Reliable local predictions

From **Week 9 onward**, this should **replace the GP surface** for decision making.

---

## f3 – SIR-Constrained Kernel Ridge Regression Exploitation with Isomap-Guided Reweighting

### Objective of Submission

To select a single high-value evaluation point for **f3** by exploiting the structure already discovered in the data using a **Kernel Ridge Regression (KRR) surrogate mean predictor**, constrained to the **one-dimensional SIR ridge** and reweighted using **Isomap geometry** to prioritise candidates that lie on the true manifold of high values.

---

## 3 Key Assumptions

1. The function **f3 contains a single dominant basin**, and all high-value observations lie on a **low-dimensional manifold** captured by **SIR and Isomap**.

2. **Kernel Ridge Regression provides a stable surrogate mean estimate** at the current sample size, enabling reliable ranking of candidate points.

3. **Exploration is no longer beneficial** because the structure of the function has been sufficiently mapped, and exploratory samples have consistently produced worse outcomes.

---

## Research Backing

### Academic Papers Supporting the Strategy

- **Li (1991)** introduced **Sliced Inverse Regression (SIR)** as a method for identifying the effective dimension reduction subspace.

- **Vovk (2013)** showed that **Kernel Ridge Regression with Leave-One-Out Cross-Validation** is stable and reliable for small sample sizes.

- **Tenenbaum et al. (2000)** demonstrated that **Isomap captures nonlinear geometric structure** that linear methods miss.

- **Bull (2011)** proved that once the surrogate stabilises and the basin is identified, **pure exploitation of the surrogate mean becomes asymptotically optimal**.

---

## Explorative Principle

The explorative principle is **constrained exploitation**.

Instead of exploring the full three-dimensional search space, the method:

1. **Restricts attention to the one-dimensional SIR direction**, capturing the dominant variation in the function.
2. Generates candidate points within empirically discovered structural bounds.
3. Projects these candidates onto the SIR axis.
4. Uses **Kernel Ridge Regression** to estimate the surrogate mean.
5. Applies geometric corrections to favour candidates lying on the high-value manifold.

### Two Structural Corrections

**SIR Ridge Penalty**

Candidates far from the peak region identified in the SIR sufficient summary plot receive a penalty.

**Isomap Manifold Reweighting**

Candidates close to the high-value manifold receive a positive score adjustment.

This approach concentrates search effort along the **true structure of the function**, avoiding evaluations in irrelevant regions.

---

## Black Box Optimization Competition

### Competition

**Black-Box Optimization Benchmarking (BBOB) – GECCO 2021**

### Strategy Observed

The **Optuna team (Nomura et al.)** demonstrated that switching from exploratory acquisition functions to **pure surrogate mean exploitation** can outperform exploration strategies on **low-dimensional unimodal functions** once the basin has been identified.

---

## Why This Strategy Is Ideal for f3

The exploratory analysis of **f3** reveals a clear unimodal structure.

Evidence includes:

- **Mapper analysis** showing a single dominant basin.
- **SIR projection** revealing an **inverted-U structure with a peak near SIR = 0**.
- **Isomap embedding** confirming a smooth high-value manifold.
- **Kernel Ridge Regression** achieving stable leave-one-out mean squared error.

Additionally, exploratory submissions have consistently returned **worse values than the initial random samples**, indicating that exploration is no longer productive.

Therefore, **pure exploitation within the discovered basin is the most rational strategy**.

---

## Justification Under Expensive Evaluation

Each evaluation of **f3** is costly.

Because:

- The function structure has already been mapped.
- No evidence suggests additional basins.
- Exploration historically produced poor results.

Exploiting the surrogate mean within the confirmed basin **maximises the probability of improvement** while minimising wasted evaluations.

The **SIR and Isomap constraints ensure the search remains inside the high-value manifold**.

---

## Tech Stack

Libraries used:

- **NumPy** – numerical operations
- **SciPy** – Sobol sampling and SIR computations
- **Scikit-learn** – Kernel Ridge Regression, scaling, and leave-one-out cross-validation

---

## Hyperparameters and Settings

### Hyperparameters

| Parameter | Description |
|---|---|
| `alpha` | KRR regularisation parameter |
| `gamma` | KRR kernel width |
| `sir_slices` | Number of SIR slices |
| `sir_window` | Target SIR score window |
| `sobol_candidates` | Number of Sobol candidate points |
| `x2_bounds` | Structural bounds for X2 |
| `x3_bounds` | Structural bounds for X3 |
| `delta_min` | Minimum distance from existing samples |
| `isomap_neighbors` | Number of neighbours used by Isomap |
| `sir_penalty_width` | Width of Gaussian penalty around SIR peak |

---

## Recommended Initial Values

```
alpha grid = [1e-4, 1e-2, 0.1, 0.5]
gamma grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
sir_slices = 5
sir_window = 0 ± 0.6
sobol_candidates = 8192
x2_bounds = [0.35, 0.95]
x3_bounds = [0.06, 0.60]
delta_min = 0.05
isomap_neighbors = 8
sir_penalty_width = 0.6
```

### Reasoning

- **Alpha and Gamma** selected via **Leave-One-Out Cross-Validation** for small-sample stability.
- **Five SIR slices** is standard for datasets around **N ≈ 25**.
- **Structural bounds** reflect the empirically confirmed high-value region.
- **Sobol sampling** ensures dense candidate coverage.
- **Distance threshold** prevents near-duplicate evaluations.
- **Penalty width** matches the observed SIR peak spread.

---

## Hyperparameter Tuning Method

Hyperparameters are tuned using:

- **Leave-One-Out Cross-Validation** for KRR parameters.
- **Empirical inspection of SIR and Isomap geometry** for structural parameters.

This avoids unstable likelihood optimisation and directly incorporates **data-driven structural insights**.

---

## Entire Flow of the Strategy

### Step-by-Step Exploration Process

1. Fit **Kernel Ridge Regression** to `X_train_3` and `y_train_3`.

2. Use **Leave-One-Out Cross-Validation** to select optimal `alpha` and `gamma`.

3. Compute the **SIR direction** and project training points onto this axis.

4. Identify the **peak region along the SIR axis**.

5. Generate **Sobol candidate points** within structural bounds.

6. Apply the **minimum distance filter** to remove near-duplicate candidates.

7. Project all candidates onto the **SIR axis** and compute their SIR scores.

8. Compute **KRR predicted mean values** for all candidates.

9. Apply a **Gaussian SIR penalty** to candidates far from the peak region.

10. Compute **Isomap manifold distances** and apply a proximity bonus.

11. Combine:

```
final_score =
    KRR_mean
    - SIR_penalty
    + Isomap_bonus
```

12. Select the **candidate with the highest score**.

13. Apply a final structural feasibility check and return the recommended point.

---

# Hypothesis Framework

## Core Assumptions

- The function is effectively **one-dimensional along the SIR direction**.
- The high-value region is **unimodal and fully contained** within the structural bounds.
- **Kernel Ridge Regression produces a stable ranking** of candidate points.

---

## Expected Outcome if Assumptions Hold

- The combined **KRR + SIR + Isomap score shows a clear peak**.
- The selected candidate lies **close to the true optimum**.
- The evaluation **improves or matches the current best value**.

---

## Expected Outcome if Assumptions Break

Possible failure cases:

- Multiple basins exist, causing **SIR to collapse them into a misleading direction**.
- The function is **highly irregular**, causing KRR to oversmooth.
- Structural bounds exclude the **true optimum**.

---

# EDA Methods — What to Remove and Why

## Remove: Kendall Correlation

Kendall Tau measures rank concordance.

For **f3**, it provides no information beyond **Spearman correlation**.

Because:

- There are **no tied ranks**.
- Results have always mirrored Spearman.
- It has never influenced strategy.

Conclusion:

```
Remove to reduce interpretive noise.
```

---

## Remove: Partial Correlation

Partial correlation attempts to isolate linear relationships.

Problems:

- f3 contains **nonlinear relationships**, particularly in `X3`.
- Sample size `N = 25` produces **very wide confidence intervals**.

Observed results:

```
p-values ≈ 0.24 – 0.82
CI ≈ [-0.6, 0.5]
```

Conclusion:

```
Not actionable → remove.
```

---

## Demote: Pearson Correlation

Pearson captures **linear association only**.

The `X3` relationship with `y` is **non-monotonic**, producing misleading negative correlations.

Example consequence:

Low-X3 strategies were attempted based on Pearson signals and **failed**.

Conclusion:

```
Keep but label as secondary diagnostic.
```

---

## Remove: K-Means Clustering (Current Implementation)

Results:

```
Optimal k = 9
N = 25
```

Average cluster size ≈ **2.8 points**, which is not statistically meaningful.

Because:

- K-means assumes **convex spherical clusters**.
- The true structure is a **ridge manifold**.

Conclusion:

```
Cluster labels provide no actionable signal.
```

---

# EDA Methods — What to Add

## Conditional Variance Analysis (X3 Binning)

Purpose:

Reveal the **non-monotonic effect of X3**.

Example code:

```python
bins = np.linspace(0, 1, 6)
bin_labels = np.digitize(X_train_3[:,2], bins)

for b in range(1,6):
    mask = bin_labels == b
    if mask.sum() > 0:
        print(
            bins[b-1], bins[b],
            mask.sum(),
            np.mean(y_train_3[mask]),
            np.max(y_train_3[mask])
        )
```

This quickly reveals the **high-value X3 plateau region**.

---

## SIR Sufficient Summary Plot (Formalised)

The SIR plot shows an **inverted-U curve**.

Extract the peak numerically:

```python
coeffs = np.polyfit(sir_scores_train, y_train_3, deg=2)
sir_peak = -coeffs[1] / (2 * coeffs[0])
print("sir peak:", sir_peak)
```

This provides the **target region for exploitation**.

---

## X2–X3 Interaction Heatmap

Because the ridge depends on **joint interaction**, plot:

```
X2 vs X3
colour = predicted y
```

Example grid prediction:

```python
x2_grid = np.linspace(0.3, 1.0, 50)
x3_grid = np.linspace(0.0, 0.7, 50)
```

Predictions are generated using **KRR** and plotted as a heatmap.

---

## Near-Best Neighbourhood Analysis

Analyse the **top-k observations** to detect structural patterns.

```python
k_best = 5
top_k_idx = np.argsort(y_train_3)[-k_best:]
top_k_X = X_train_3[top_k_idx]
```

Compute:

- centroid
- neighbourhood radius
- per-dimension standard deviation

This identifies the **tightest dimension**, indicating where exploitation should occur.

---

## LOO Prediction Residual Map

Plot **leave-one-out prediction errors** to evaluate surrogate reliability.

```python
loo_residuals = y_train_3 - best_loo_preds
```

If residuals are small near the best points:

```
KRR predictions are trustworthy
```

If large:

```
Surrogate is unreliable in critical regions.
```

---

## F4 — Sequential Shrinking Radius GP Expected Improvement with Pre-Committed Adaptive Radius

### Objective of Submission

The goal is to select a single next evaluation point for **f4** using a **Gaussian Process (GP) surrogate model** and the **Expected Improvement (EI)** acquisition function within a **shrinking neighbourhood around the current best observation**.

The strategy aggressively explores the most promising local region while preventing near-duplicate evaluations and maintaining efficient use of expensive function evaluations.

---

## Key Assumptions

1. **The current best region already contains the primary basin of attraction**  
   The optimal solution is assumed to lie near the current best observation, making local refinement worthwhile.

2. **A Matérn Gaussian Process with Automatic Relevance Determination (ARD)**  
   This model is assumed to capture the local structure of the function sufficiently well to rank candidate points.

3. **Expected Improvement balances exploration and exploitation**  
   EI rewards both high predicted mean values and regions with non-trivial predictive uncertainty.

---

## Research Backing

Several foundational works support this strategy:

- **Jones, Schonlau, Welch (1998)** introduced Expected Improvement as a principled acquisition function for expensive black-box optimisation, demonstrating strong empirical performance.

- **Eriksson et al. (2019)** proposed **TuRBO**, which uses adaptive local trust regions that shrink and expand depending on optimisation success. This directly inspired the shrinking radius idea used here.

- **Mockus (1989)** formalised the Bayesian approach to global optimisation, where the next query is selected by adapting to the posterior distribution and focusing on promising regions.

- Reports from the **NeurIPS 2020 Black Box Optimisation Challenge** show that top teams frequently used **sequential EI with adaptive trust regions** during the exploitation phase.

---

## Explorative Principle

The core principle is **local trust region exploration around the current best point**.

Instead of searching the entire four-dimensional domain, a **ball of radius `r`** is defined around the best observed input.

Inside this neighbourhood:

1. A **Gaussian Process** is fitted to all observations.
2. **Expected Improvement** is used to score candidate points.
3. Candidate points are generated using **Sobol sampling** within the trust region.
4. Points too close to existing observations are filtered out.
5. The highest-EI candidate is selected.

Additionally, a **local gradient-based search (L-BFGS-B)** is performed within the same neighbourhood to refine the best candidate.

This approach is rational because:

- A strong incumbent solution already exists.
- The remaining evaluation budget is small.
- The strategy prioritises **refinement of the most promising basin** rather than restarting global exploration.

---

## Black Box Optimization Competition

**Competition:** NeurIPS 2020 Black Box Optimisation Challenge

**Winning strategy influence:**

The **Optuna developers team** used **sequential EI combined with adaptive trust region contraction** during the exploitation phase. Radius adjustments were pre-committed based on success and failure counts.

The present strategy follows the same spirit: restricting search to a local region while guiding exploration using EI.

---

## Why This Strategy Is Suitable for F4

The available observations indicate that a **promising region already exists**.

Global exploration would:

- consume the evaluation budget
- revisit low-value areas of the search space

Instead, a **local Gaussian Process model** around the best point can capture curvature and guide search to nearby improvements.

Expected Improvement naturally balances:

- exploitation of high predicted values
- exploration of uncertain areas nearby

The neighbourhood radius ensures that the search remains focused where improvements are most likely.

---

## Justification for Expensive Evaluations

Each evaluation of **f4** is costly.

Therefore the strategy:

- restricts search to a **local neighbourhood**
- avoids redundant queries using a **minimum distance guard**
- combines **Sobol candidate sampling** with **local optimisation**

This produces a **dense yet efficient search of the most promising region**.

---

## Tech Stack

Libraries used:

- **NumPy** — numerical operations and array handling  
- **SciPy** — optimisation (`L-BFGS-B`) and Sobol sampling  
- **SciPy Spatial** — distance computations  
- **Scikit-learn** — Gaussian Process regression with Matérn kernel

---

## Hyperparameters and Settings

### Hyperparameters

- Kernel type and smoothness (Matérn kernel with `ν = 2.5`)
- ARD length-scale bounds
- Number of GP optimiser restarts
- Neighbourhood radius around the best point
- Number of Sobol candidate samples
- Minimum distance guard
- Expected Improvement exploration parameter `ξ`
- Number of L-BFGS-B restarts

---

### Recommended Initial Values

| Parameter | Value | Reason |
|----------|------|------|
| Kernel | Matérn (ν = 2.5) | Flexible smoothness assumption |
| Length scale bounds | [0.05, 10.0] | Allows local and global variation |
| GP restarts | 25 | Avoids poor local optima |
| Radius | 0.10 | Focuses tightly on the promising basin |
| Sobol candidates | 100000 | Dense local coverage |
| Min distance | 0.025 | Prevents duplicate sampling |
| EI ξ | 0.0 | Pure exploitation in late budget |
| L-BFGS-B restarts | 50 | Improves chance of locating EI maxima |

These settings are informed by empirical practice and literature on **EI-based Bayesian optimisation** and **TuRBO-style trust regions**.

---

## Hyperparameter Tuning Method

Hyperparameters are chosen based on:

- empirical optimisation experience
- prior literature on EI-based optimisation
- trust-region Bayesian optimisation methods

Adaptive radius updates could be used in practice, but the current approach **pre-commits the radius for simplicity and robustness**.

---

## Strategy Flow

### Step-by-Step Process

1. Compute the **current best observation** from the dataset.

2. Fit a **Matérn ARD Gaussian Process** to all training data.

3. Inspect fitted **length scales** to understand dimensional sensitivity.

4. Define a **trust region radius** around the current best point.

5. Construct a **local search box** clipped to the domain `[0,1]^4`.

6. Generate **Sobol candidate points** inside this region.

7. Filter candidates to keep only those inside the **L2 ball**.

8. Apply the **minimum distance guard** to remove near-duplicates.

9. Compute **Expected Improvement** for all candidates.

10. Identify the **best Sobol candidate**.

11. Run **multiple L-BFGS-B optimisations** to locally maximise EI.

12. Compare the best **gradient candidate** with the best **Sobol candidate**.

13. Select the point with the **highest EI**.

14. Compute diagnostics:

   - predicted mean
   - predictive standard deviation
   - EI value
   - distance from best point
   - distance from nearest training observation

15. Apply simple EI gates to determine whether improvement is meaningful.

16. Output the **final recommended submission point**.

---

## Hypothesis Framework

### Core Assumptions

- The best region still contains improvements.
- The Gaussian Process approximates local behaviour well.
- Expected Improvement ranks candidate points reliably.

---

### Expected Outcome if Assumptions Hold

- The selected point lies near the incumbent optimum.
- EI value indicates meaningful improvement probability.
- Valid candidate points remain within the trust region.

---

### Expected Outcome if Assumptions Fail

Possible failure cases:

- The global optimum lies **outside the trust region**.
- The Gaussian Process **misrepresents the function**.
- The neighbourhood becomes **fully saturated with observations**.

In the last case, the strategy correctly signals that further local exploration is not useful.

---

# EDA Methodology

## Methods to Remove

### 1. Sliced Inverse Regression (SIR)

**Reason for removal**

Bootstrap stability deteriorated from **31° to 43°**, far above the reliability threshold of **15°**.

At this level, bootstrap resamples produce completely different leading directions.

Additionally, the week-9 strategy based on SIR produced the **worst observed value (-4.630)**.

**Verdict:** Remove entirely.

---

### 2. Integrated Gradients (IG)

**Reason for removal**

Integrated Gradients requires a trained neural network model.

With **N ≈ 40 observations**, the neural network itself is unreliable.

The feature rankings produced by IG contradict both:

- partial correlations
- empirical improvement gradients

**Verdict:** Remove.

---

### 3. Mapper (Topological Data Analysis)

**Reason for removal**

Mapper was useful once to identify that the function has **one coherent high-value basin**.

Running it weekly provides **no additional actionable information**.

**Verdict:** Retain as a one-time diagnostic only.

---

### 4. Isomap 1D Embedding

**Reason for removal**

Isomap is unstable at small sample sizes and depends heavily on `n_neighbors`.

More importantly, the embedding **cannot generate new candidate points**, making it useless for optimisation decisions.

**Verdict:** Remove from weekly analysis.

---

# Methods to Retain

## Correlation Analysis

### Pearson, Spearman, Kendall

These remain valuable because they:

- are statistically grounded
- track structural changes over time
- reveal nonlinear relationships when they diverge.

---

## Partial Correlations with p-values

These provide the **most reliable feature importance signal**.

They measure each dimension's contribution **while controlling for all other variables**.

---

## Mutual Information

Retain only as a **secondary signal**.

MI estimates are unreliable below **N = 50**, so they should not be used as primary evidence.

---

## Xi vs Yi Scatter Plots

These provide the most **direct visual evidence of function shape**, revealing inverted-U structures and approximate peak regions.

---

## K-Means Silhouette Analysis

This confirms the number of **distinct clusters** in the input space.

The result **k = 4 clusters** remained stable across weeks and is computationally cheap.

---

# Methods to Add

## 1. Empirical Gradient Table

### Concept

Compare the **top k observations** (k ≈ 5–10) and compute dimension-wise differences.

For each pair of ranked observations:Δx_i = x_best[i] − x_second[i]

Identify consistent directional changes across improvements.

### Motivation

This approach produced the **week-10 improvement** without relying on complex models.

It directly extracts gradient information from observed data.

---

## 2. Local Neighbourhood Coverage Map

### Concept

Measure how much of a candidate trust region is already covered by existing points.

Example:

| Radius | Saturation |
|------|------|
| 0.08 | 78% |
| 0.10 | 65% |
| 0.15 | 40% |

### Motivation

This prevents choosing a radius that is already saturated with existing observations.

---

## 3. GP Posterior Surface Slice Plots

### Concept

Create **2D heatmaps of GP predictions** while fixing two dimensions at the current best values.

Six slices are produced:
x1-x2
x1-x3
x1-x4
x2-x3
x2-x4
x3-x4

### Motivation

These plots reveal:

- where the GP predicts high values
- where predictive uncertainty is high

This directly informs acquisition decisions.

---

## 4. Consecutive Improvement Direction Test

### Concept

Analyse the **directional changes associated with each observed improvement event**.

If multiple improvements share the same sign change in a dimension, it suggests a real gradient direction.

### Motivation

This transforms a small number of improvement events into a **structured directional hypothesis test**.

---

## Summary

The final methodology emphasises:

- **Trust-region Bayesian optimisation**
- **Expected Improvement acquisition**
- **Data-driven EDA rather than unstable dimensionality reduction**

Weak small-sample techniques are removed, while robust statistical diagnostics and empirically grounded tools are retained.

---

## F5 — SIR Directed GP Mean Exploitation with Per-Dimension Targeted Perturbation

### Objective of Submission

The goal is to select a **single high-value evaluation point for f5** by exploiting a **well-calibrated Gaussian Process (GP) surrogate model**, while **biasing candidate selection along the SIR (Sliced Inverse Regression) direction**.

Instead of exploring the entire space, the method concentrates on a **corner region near the current best observation**, applying **dimension-specific perturbations** guided by sufficient dimension reduction insights.

---

## Key Assumptions

### 1. Effective Dimensionality is Lower than the Ambient Dimension

The true response surface of **f5** is assumed to vary primarily along a **lower-dimensional structure**, which the **SIR direction estimated from `X_train_5` and `y_train_5` captures**.

---

### 2. The Gaussian Process is Well Calibrated in the High-Value Region

The GP is fitted on **log-transformed response values**, improving variance stability.  
Under this assumption, **maximising the GP posterior mean** is a valid **terminal exploitation strategy**.

---

### 3. The True Maximum Lies Near a Corner Region

Evidence from the training data suggests the maximum lies near a **corner of the input domain**.

Further improvement is expected to arise from **targeted dimension perturbations**, rather than broad global exploration.

---

## Research Backing

### Li & Wang (2007) — Directional Regression for Dimension Reduction

This work demonstrates that **SIR-type methods identify sufficient reduction directions**, enabling optimisation along informative subspaces rather than the full ambient space.

---

### Jones, Schonlau, Welch (1998)

The **Efficient Global Optimization (EGO)** framework shows that once the surrogate model is calibrated and the optimum region is identified, **maximising the GP posterior mean** becomes an appropriate final exploitation strategy.

---

### Hutter, Hoos, Leyton-Brown (2011) — SMAC

The **SMAC algorithm** introduces **per-dimension perturbations during intensification**, perturbing individual coordinates of the incumbent solution to determine which dimensions still allow improvement.

This concept directly inspires the **targeted perturbation strategy** used here.

---

## Explorative Principle

The principle is **structured local exploitation guided by sufficient dimension reduction**.

Instead of uniform exploration across the full 4D domain:

1. Search is restricted to a **small corner region** near the current best point.
2. Candidate points are evaluated primarily by the **GP posterior mean**.
3. The **SIR projection acts as a directional bias**, favouring points aligned with historical improvement.
4. **Per-dimension constraints** encode prior knowledge about promising coordinates.
5. **Targeted perturbations emphasise dimensions with remaining improvement potential**.

This approach is appropriate because the structure of **f5 suggests a monotonic increase toward a corner of the domain**, and the GP model appears well calibrated in that region.

---

## Black Box Optimization Competition

**Competition:** NeurIPS 2020 Black Box Optimization Challenge (Bayesmark track)

**Winning Strategy Inspiration**

The **Squirrel AI / SMAC3 team (Lindauer et al.)** used **SMAC-style intensification**, where:

- new incumbents are locally perturbed dimension-by-dimension
- surrogate model predictions guide exploitation

The strategy described here follows the same conceptual framework.

---

## Why This Strategy is Suitable for F5

### Structure of Observed Data

Analysis of `X_train_5` and `y_train_5` indicates:

- high values cluster near a **corner of the domain**
- the **SIR direction identifies a strong improvement axis**
- GP prediction error is **small for high-value observations**

Under these conditions, **global exploration is unlikely to produce additional gains**.

Instead, concentrating candidates in the **high-value corner region** and ranking them using **GP mean plus directional bias** increases the probability of selecting a near-optimal point.

---

## Justification for Expensive Evaluations

Because evaluations of **f5 are expensive**, only a small number of submissions remain.

This strategy therefore:

- restricts candidate generation to a **small feasible region**
- biases selection toward **historically successful directions**
- enforces a **minimum distance guard** to avoid duplicate queries

This maximises the **information gained from the final evaluation**.

---

## Tech Stack

### Libraries

- **NumPy** — numerical operations and array handling
- **SciPy** — Sobol sequence generation
- **Scikit-learn** — Gaussian Process regression, preprocessing
- **SIR implementation** — sufficient dimension reduction estimation

---

## Hyperparameters and Settings

### List of Hyperparameters

- Corner region bounds
- Per-dimension floors
- Target improvement band for \(x_3\)
- Number of Sobol candidates
- Minimum distance guard
- GP kernel configuration
- GP optimiser restarts
- SIR bias weight
- Log transform shift
- GP noise lower bound

---

### Recommended Initial Values

| Parameter | Value | Reason |
|-----------|------|------|
| Corner region | `[0.94,1.0]^4` | Focuses on empirically best region |
| `x1` floor | `0.98` | Keeps dimension near maximum |
| `x2` floor | `0.95` | Maintains strong values |
| `x3` floor | `0.95` | SIR indicates upward improvement |
| `x3` target zone | `[0.96,1.0]` | Primary improvement dimension |
| `x4` floor | `0.96` | High-value region constraint |
| Sobol candidates | `12000` | Dense local coverage |
| Min distance | `0.02` | Prevents duplicate queries |
| GP kernel | Matérn ARD + white noise | Flexible smoothness |
| GP restarts | `≈30` | Avoid local optima |
| SIR bias weight | small constant | Keeps GP mean dominant |
| Log shift | `1.0` | Stabilises variance |
| Noise lower bound | `1e-6` | Numerical stability |

---

## Hyperparameter Tuning Method

Hyperparameters are selected based on:

- empirical inspection of `X_train_5` and `y_train_5`
- stability checks of the **SIR direction**
- prior validated GP configurations
- multi-start optimisation of GP marginal likelihood

Candidate count and distance guards balance **coverage and computational efficiency**.

---

## Strategy Flow

### Step 1 — Load and Transform Data

Load `X_train_5` and `y_train_5`.

Apply a **log transform with a small shift** to stabilise variance in response values.

---

### Step 2 — Fit Gaussian Process

Fit a **Matérn ARD GP with white noise** using multiple optimiser restarts to obtain a well-calibrated surrogate model.

---

### Step 3 — Estimate SIR Direction

Compute the **SIR direction from `X_train_5` and `y_train_5`**.

Evaluate stability via bootstrap or angular variability.

Retain the direction only if sufficiently stable.

---

### Step 4 — Generate Candidate Points

Generate **12,000 Sobol samples** within the **corner region**

```
[0.94, 1.0]^4
```

---

### Step 5 — Apply Per-Dimension Floors

Enforce coordinate constraints:

```
x1 ≥ 0.98
x2 ≥ 0.95
x3 ≥ 0.95
x4 ≥ 0.96
```

Prefer candidates with

```
x3 ∈ [0.96, 1.0]
```

---

### Step 6 — Compute Directional Bias

Project each candidate onto the **SIR direction** and normalise the projection to obtain a **directional score**.

---

### Step 7 — Apply Minimum Distance Guard

Compute distance from each candidate to its nearest observation in `X_train_5`.

Discard candidates with distance < `0.02`.

---

### Step 8 — GP Prediction

For each remaining candidate compute:

- GP posterior mean
- GP predictive standard deviation

---

### Step 9 — Candidate Scoring

Combine GP mean with directional bias:

```
score = GP_mean + λ * SIR_projection
```

where \(λ\) is a small weight.

The GP mean remains the dominant term.

---

### Step 10 — Select Best Candidate

Select the candidate with the **highest combined score** as the next evaluation point.

---

### Step 11 — Diagnostics

Report:

- selected candidate
- predicted GP mean
- distance from incumbent best point
- dominant dimension change

The output highlights that **x3 is pushed toward the target zone**, consistent with the SIR signal.

---

## Hypothesis Framework

### Core Assumptions

1. The true maximum lies within the **corner region `[0.94,1.0]^4`**.
2. The **SIR direction correctly identifies the improvement axis**, particularly increasing \(x_3\).
3. The **GP surrogate is well calibrated** in this region.

---

### Expected Outcome if Assumptions Hold

- The selected point will lie near

```
(1.0, 0.97, 0.98, 0.99)
```

- \(x_3\) will increase relative to the incumbent.
- Observed \(y\) will **match or exceed the current best value**.
- GP mean and SIR projection will **align in candidate ranking**.

---

### Expected Outcome if Assumptions Fail

Possible failure cases include:

- The global maximum lies **outside the corner region**
- The **SIR direction is misestimated**
- The **GP surrogate is poorly calibrated**

In these cases the model may assign a high score to a candidate that does not improve the objective.

---

# EDA Methodology

## Methods to Remove

### K-Means Clustering

At **N = 30**, silhouette scores continue rising up to **k = 9**, indicating the algorithm is fitting noise rather than meaningful structure.

Nine clusters from thirty points is statistically implausible.

**Verdict:** Remove.

---

### Mapper (Topological Data Analysis)

Although theoretically useful, Mapper is **extremely sensitive to parameter choices** at small sample sizes.

Small changes in:

- interval count
- overlap
- epsilon

produce drastically different graph topologies.

**Verdict:** Remove from active use.

---

### Integrated Gradients

Integrated Gradients requires fitting a **neural network model**.

With **30 training points**, the network is underdetermined.

The feature rankings produced are **less reliable than direct statistical measures** such as partial correlations.

**Verdict:** Remove.

---

# Methods to Add

## Leave-One-Out GP Residual Analysis

Fit the GP and perform **leave-one-out predictions**.

Plot:

```
predicted vs true values
```

in the original response scale.

### Purpose

This reveals:

- where the GP model is **miscalibrated**
- which regions of the input space are **poorly modelled**

These insights directly inform acquisition strategies.

---

## Pairwise Scatter of Top-k Observations

Select the **top 8 observations by response value**.

Plot pairwise combinations of inputs coloured by response.

### Purpose

At small sample sizes, full scatter plots obscure structure.

Focusing on **top observations reveals joint high-value regions** across dimensions.

---

## GP Posterior Mean Surface Slice

Fix two stable dimensions:

```
X1 = 1.0
X4 = 0.98
```

Plot a **2D heatmap of the GP posterior mean** across:

```
X2 × X3
```

### Purpose

This directly visualises:

- where the GP predicts the optimum
- whether the surface forms a **ridge, peak, or plateau**

Such information guides final candidate selection in the **two most uncertain dimensions**.

---

## Summary

The final approach combines:

- **SIR-based sufficient dimension reduction**
- **Gaussian Process surrogate modelling**
- **corner-region exploitation**
- **per-dimension targeted perturbations**

Weak small-sample techniques are removed, while diagnostics and modelling tools that **directly influence optimisation decisions** are prioritised.

---

## F6 — ARD GP Sequential Expected Improvement with Cluster-1 Subspace Restriction and X1-Aware Candidate Filtering

### Objective of Submission

The objective is to **select the next evaluation point for f6** using **Expected Improvement (EI)** computed from an **Automatic Relevance Determination (ARD) Gaussian Process**, while restricting candidate points to the **empirically confirmed high-value subspace corresponding to cluster 1**.

Candidate generation is further refined through a **soft filter on X1**, allowing aggressive exploration while remaining focused on the region most likely to yield improvement.

---

## Key Assumptions

### 1. Dominant High-Value Basin Exists

The dataset `X_train_6` and `y_train_6` shows that **cluster 1 represents a dominant high-value basin** in the search space.

The strategy assumes that **the true optimum lies inside this basin**, making it safe to restrict exploration to this region.

---

### 2. X4 and X5 Define the Active Ridge

Analysis suggests that **X4 and X5 form the main structural ridge of the function**, while **X1 acts as a weaker secondary contributor**.

Restricting candidates along these axes therefore concentrates exploration along the most promising structure.

---

### 3. ARD Gaussian Process Provides Reliable Local Ranking

An **ARD Matérn Gaussian Process fitted on all observations** is assumed to model the function sufficiently well locally, allowing **Expected Improvement to reliably rank candidate points**.

---

## Research Backing

### Jones, Schonlau, Welch (1998)

The **Efficient Global Optimization (EGO)** framework introduces **Expected Improvement**, showing that sequential EI converges toward optimal regions as the search progresses.

This is especially effective when the search is already near a local optimum.

---

### Srinivas et al. (2010)

The **GP-UCB theoretical framework** demonstrates that restricting the search domain using prior structural knowledge can **reduce regret while preserving optimality guarantees**.

---

### Eriksson & Jankowiak (2021)

Their work on **Bayesian optimization in sparse axis-aligned subspaces** shows that focusing sampling in empirically identified **active subspaces dramatically improves efficiency** in higher dimensions.

This supports concentrating exploration on the **X4–X5 ridge** while incorporating X1 as a softer constraint.

---

## Explorative Principle

The principle is **subspace-focused exploration guided by Expected Improvement**.

Instead of sampling across the full **five-dimensional domain**, the strategy:

1. Restricts candidates to the **cluster-1 region defined by X4 and X5**.
2. Applies a **soft constraint on X1** to avoid historically weak regions.
3. Uses a **Gaussian Process surrogate** to compute EI across candidates.

Expected Improvement is naturally high where:

- the **predicted mean is strong**
- the **predictive uncertainty remains non-negligible**

This provides a balance between **local exploitation and controlled exploration** within the identified ridge.

---

## Black Box Optimization Competition

**Competition:** NeurIPS 2020 Black Box Optimization Challenge (Bayesmark Track)

**Winning Strategy Inspiration**

The **Optuna developers team** used **sequential EI with neighbourhood restriction** after identifying promising basins.

Their strategy of **shrinking candidate regions around high-value clusters** closely mirrors the approach applied here to **f6**.

---

## Why This Strategy is Suitable for F6

Analysis of `X_train_6` and `y_train_6` reveals:

- a **clear ridge structure in X4 and X5**
- repeated improvements obtained from EI with neighbourhood filtering
- emerging evidence that **X1 contributes modestly to improvements**

Given the **high cost of evaluations**, focusing exploration within this productive basin ensures efficient use of the remaining budget.

The **minimum-distance guard** ensures each new evaluation is meaningfully different from existing samples.

---

## Tech Stack

Libraries used:

- **NumPy** — numerical operations and arrays  
- **SciPy** — Sobol sequence generation and utilities  
- **Scikit-learn** — Gaussian Process regression with Matérn + White kernels and response standardisation

---

## Hyperparameters and Settings

### Hyperparameter List

- GP kernel configuration (Matérn + white noise)
- Length-scale bounds for ARD dimensions
- GP optimiser restarts
- Sobol candidate count
- X4 subspace bounds
- X5 subspace bounds
- X1 soft constraint
- Minimum distance guard
- EI exploration parameter \( \xi \)

---

### Recommended Initial Values

| Parameter | Value | Rationale |
|-----------|------|-----------|
| Kernel | Matérn (ν=2.5) + White | Handles smooth ridges with small noise |
| Length scale bounds | [0.05, 10.0] | Allows narrow and broad structure |
| GP restarts | ~30 | Robust hyperparameter optimisation |
| Sobol candidates | 15000 | Dense coverage of restricted region |
| X4 bounds | [0.68, 0.88] | Matches cluster-1 ridge region |
| X5 bounds | [0.02, 0.18] | Reflects best observed values |
| X1 soft bound | ≤ 0.70 | Higher values rarely produce top results |
| Minimum distance | 0.05 | Prevents duplicate sampling |
| EI ξ | 0.003 | Small exploration margin near optimum |

These values are derived from **cluster analysis, mutual information signals, and coordinates of recent best points**.

---

## Hyperparameter Tuning Method

Hyperparameters are determined through:

- empirical inspection of `X_train_6` and `y_train_6`
- cluster centroids and mutual information analysis
- optimisation of GP marginal likelihood with multiple restarts
- practical coverage vs. computation tradeoffs

---

## Strategy Flow

### Step 1 — Identify Current Best Observation

Load `X_train_6` and `y_train_6` and identify the **current best point**, paying particular attention to its **X4 and X5 coordinates**.

---

### Step 2 — Standardise Response

Standardise `y_train_6` to **zero mean and unit variance** to stabilise GP fitting.

---

### Step 3 — Fit ARD Gaussian Process

Fit a **Matérn ARD GP with white noise**, using around **30 optimiser restarts**.

Inspect learned **length scales** to confirm that **X4 and X5 exhibit shorter scales**, indicating higher activity.

---

### Step 4 — Generate Candidate Points

Generate **15,000 Sobol samples** in the full domain:

```
[0,1]^5
```

---

### Step 5 — Apply Subspace Restriction

Filter candidates so that:

```
X4 ∈ [0.68, 0.88]
X5 ∈ [0.02, 0.18]
X1 ≤ 0.70
```

Optional filters may remove historically poor corner combinations.

---

### Step 6 — Apply Distance Guard

Compute distance from each candidate to the nearest observation in `X_train_6`.

Discard candidates with distance:

```
distance < 0.05
```

---

### Step 7 — Compute Expected Improvement

Evaluate **Expected Improvement** for all remaining candidates using:

- the fitted GP
- the current best observed value
- exploration parameter \( \xi = 0.003 \)

---

### Step 8 — Rank Candidates

Sort candidates by EI and select the **highest-ranked point** as the proposed evaluation.

---

### Step 9 — Diagnostics

Print the **top candidates with coordinates and EI values** to verify:

- candidate diversity
- absence of boundary artefacts
- dimensional drivers of predicted improvement

---

## Hypothesis Framework

### Core Assumptions

1. The **cluster-1 subspace fully contains the high-value basin**.
2. **X1 contributes secondarily** to improvement but should be softly restricted.
3. The **ARD Gaussian Process is locally accurate**, allowing EI rankings to be meaningful.

---

### Expected Outcome if Assumptions Hold

- The selected candidate lies along the **X4–X5 ridge**.
- The **X1 soft filter is respected**.
- The new evaluation improves or closely matches the current best value.
- Improvements become **incrementally smaller**, consistent with EI convergence.

---

### Expected Outcome if Assumptions Fail

Potential failure modes:

- The **true optimum lies outside the cluster-1 subspace**
- **X1 is more influential than assumed**
- The **GP surrogate is poorly specified**

In these cases EI rankings may mislead the search or stagnate.

---

# EDA Methodology

## Methods to Remove

### Sliced Inverse Regression (SIR)

Bootstrap stability angle ≈ **89.7°**, indicating essentially random directions.

SIR requires **clean low-dimensional structure**, which is not present at **N = 30 in 5D**.

**Verdict:** Remove.

---

### Mapper / Topological Data Analysis

The graph contained **23 connected components from 27 nodes**, meaning most nodes correspond to individual points.

At this sample size there is insufficient data to produce meaningful topology.

**Verdict:** Remove.

---

### K-Means Clustering

Silhouette score **≈ 0.21**, indicating weak cluster separation.

Cluster centroids are unstable and reflect sample geometry rather than real structure.

**Verdict:** Remove.

---

### Isomap

The 1D embedding reproduces information already visible in **X4–X5 scatter plots**.

It adds complexity without additional insight.

**Verdict:** Remove.

---

### Integrated Gradients

Integrated Gradients is designed for **neural network attribution**, not Gaussian Process models.

Applying it here has **no theoretical justification**.

Feature importance should instead rely on **mutual information or partial correlations**.

**Verdict:** Remove.

---

# Methods to Add

## Gaussian Process Leave-One-Out Residual Plot

Perform **LOO predictions** for each training point and plot:

```
predicted vs true response values
```

Large residuals identify:

- outliers
- regions poorly modelled by the GP

This directly informs acquisition strategy.

---

## Partial Dependence Surface for X4 and X5

Compute a **2D partial dependence surface** for:

```
X4 ∈ [0.65, 0.90]
X5 ∈ [0.02, 0.20]
```

using the GP posterior mean.

This reveals:

- ridge shape
- peak location
- ridge width

---

## Pairwise Scatter of Top-10 Observations

Select the **top 10 highest-value observations**.

Plot pairwise relationships:

```
X1 vs X4
X1 vs X5
X4 vs X5
```

Colour points by rank.

This reveals the **input combinations responsible for the best outputs**.

---

## GP Posterior Mean Surface Slice Plots

Create two diagnostic heatmaps:

1. Fix \(X1, X2, X3\) at their best values and plot **X4 vs X5**.
2. Fix \(X4, X5\) at their best values and plot **X1 vs X3**.

These plots show exactly what the GP surrogate believes the function surface looks like.

---

## Empirical Variogram

Compute the **squared difference in response values** against **Euclidean input distance** for all observation pairs.

This reveals the **empirical correlation length** of the function and validates whether the GP's learned length scales are realistic.

---

## X1 Conditional Analysis

Evaluate the effect of X1 **only within the confirmed good region**:

```
X4 ≥ 0.68
X5 ≤ 0.20
```

Plot:

```
y vs X1
```

within this subset to determine whether **X1 genuinely contributes to improvement**.

---

## Summary

The final framework for **f6** combines:

- **ARD Gaussian Process surrogate modelling**
- **Expected Improvement acquisition**
- **subspace restriction based on empirical ridge structure**
- **candidate filtering informed by domain knowledge**

Unstable small-sample techniques are removed, while diagnostics that directly improve **model calibration and acquisition decisions** are prioritised.

---

# f7 – Tightened Trust Region Ensemble Exploitation with SIR-Updated Directional Scoring

## Objective of Submission
To select a **single, very high-value evaluation point for f7** by exploiting a **locally focused trust region** around the current best in `X_train_7` and `y_train_7`, using an **ensemble of surrogate models** and a **SIR-based directional score** to rank dense candidates inside that region.

---

# 3 Key Assumptions

1. **Single dominant basin**  
   f7 has a single dominant high-value basin, and all top observations in `X_train_7` lie on the same local manifold near the current best.

2. **Local surrogate reliability**  
   Local surrogate models (GP, SVR, KNN, XGBoost) fitted in this basin are accurate enough to rank nearby candidates when combined in an ensemble.

3. **Stable SIR direction**  
   The SIR direction estimated from `X_train_7` and `y_train_7` is stable and correctly indicates how changes in key coordinates move along the high-value ridge.

---

# Research Backing

## Academic Papers Supporting the Strategy

**Eriksson et al. (2019)**  
*Scalable Global Optimization via Local Bayesian Optimization (TuRBO), NeurIPS*  
Introduces **trust-region Bayesian optimization**, restricting search to shrinking regions around the incumbent to improve efficiency near peaks.

**Jones, Schonlau, Welch (1998)**  
*Efficient Global Optimization of Expensive Black-Box Functions, Journal of Global Optimization*  
Provides the theoretical basis for **local exploitation using surrogate models** once a good basin has been found.

**Li (1991)**  
*Sliced Inverse Regression for Dimension Reduction, JASA*  
Demonstrates that **SIR can identify important directions in the input space**, allowing optimization to move along directions most likely to increase the response.

---

# Explorative Principle

The explorative principle is **local trust-region exploitation with directional guidance**.

Instead of searching the full high-dimensional domain, the algorithm:

1. Defines a **hypercube trust region around the best point** in `X_train_7`.
2. Generates many **low-discrepancy candidate points** within this region.
3. Uses a **surrogate ensemble** to predict the mean and uncertainty of each candidate.
4. Applies a **SIR-based directional score** to encourage movement along empirically beneficial directions.

For example:

- Some coordinates may need to **increase**
- Others may need to **decrease**

This approach is rational for **f7** because:

- Data show a **tight cluster of high-value points**
- Only **one evaluation remains**
- The best candidate is almost certainly **near the current peak**

Thus **aggressive local exploitation** is preferable to global exploration.

---

# Black-Box Optimization Competition

## Competition
**NeurIPS 2020 Black Box Optimization Challenge**

## Winning Approach
Top teams (including the **Optuna team**) used **TuRBO-style trust regions** combined with **surrogate ensembles or multiple kernels**, repeatedly:

1. Re-centering the trust region on improvements  
2. Shrinking the region as convergence approached  

The same principle is applied here for **f7**.

---

# Why This Strategy Is Ideal for f7

## Expensive Function Evaluations

Evaluations of **f7 are expensive**, and the existing data already reveal:

- A **tight geometric cluster of high-value points**
- Evidence of a **single local manifold**

Conceptual manifold tools (Isomap, Mapper) indicate this cluster lies on a **single structure**, while **SIR provides a stable direction of improvement**.

With very few evaluations remaining:

- Global exploration is unlikely to outperform the current basin.
- A **tight trust region around the best point** focuses computation where improvement is most plausible.

The **ensemble + directional guidance** allows small but meaningful movements along the ridge of the function.

---

# Tech Stack

Libraries and frameworks used:

- **NumPy** — numerical operations and array handling  
- **SciPy** — Sobol sequence generation  
- **Scikit-learn**
  - Gaussian Process Regression
  - Support Vector Regression
  - K-Nearest Neighbors Regression  
- **XGBoost** — gradient boosted tree regression

---

# Hyperparameters and Settings

## List of Hyperparameters

- Trust region centre (current best point)
- Secondary anchor (cluster centroid)
- Trust region width per dimension
- Number of top observations used for centroid
- Ensemble composition:
  - GP
  - SVR
  - KNN
  - XGBoost
- Cross-validation weights
- Variance inflation factor
- Acquisition coefficient
- Number of Sobol candidates
- SIR directional coefficients

---

# Recommended Initial Values

**Trust region centre**  
Current best row in `X_train_7`.

**Secondary anchor**  
K-means centroid of the high-value cluster.

**Trust region width**  
≈ **0.15 per dimension**  
(contracted from earlier 0.20 following TuRBO success rules).

**Local centroid observations**  
Top **5 observations**.

**Ensemble models**

| Model | Purpose |
|------|------|
| GP | Smooth uncertainty modelling |
| SVR | Robust margin regression |
| KNN (k≈4) | Local averaging |
| XGBoost | Flexible nonlinear patterns |

**Model weights**

Derived from **cross-validated prediction error**.

**Variance inflation factor**

≈ **3.0**

Prevents ensemble overconfidence.

**Acquisition coefficient**

≈ **1.5**

Balances predicted mean and uncertainty.

**Sobol candidates**

≈ **4096**

Provides dense trust-region coverage.

**SIR directional coefficients**

Taken from the **leading SIR eigenvector** estimated on the dataset.

---

# Hyperparameter Tuning

Hyperparameters are tuned via **cross-validation on `X_train_7` and `y_train_7`**:

1. Each surrogate model is trained with a **small hyperparameter grid**.
2. The configuration with **lowest validation error** is selected.
3. **Ensemble weights** are proportional to **inverse validation error**.
4. Trust-region width is adjusted according to **TuRBO-style success rules**.
5. **Bootstrap resampling** validates SIR direction stability.

---

# Entire Flow of the Strategy

## Step-by-Step Exploration Process

### 1. Identify the Best Observation
Locate the best point in `X_train_7` and `y_train_7`.

### 2. Compute High-Value Centroid
Calculate the centroid of the **top observations** to characterize the local basin.

### 3. Define Trust Region
Create a hypercube centred at the best point (optionally blended with the centroid).

### 4. Train Surrogate Ensemble

Fit four models:

- GP
- SVR
- KNN
- XGBoost

Compute **cross-validated prediction errors** and derive ensemble weights.

### 5. Estimate SIR Direction

Compute SIR on `X_train_7` and `y_train_7` and extract:

- Sign of each coordinate
- Relative importance

### 6. Generate Candidate Points

Generate **4096 Sobol candidates** inside the trust region.

### 7. Predict Ensemble Mean and Variance

For each candidate:

- Obtain predictions from all models
- Combine them into **ensemble mean and variance**
- Inflate variance using the chosen factor

### 8. Compute Directional Score

Project each candidate onto the **SIR direction** and scale the result.

Add this value as a **directional bonus**.

### 9. Compute Acquisition Score

The final acquisition score is: 
Score = EnsembleMean
+ AcquisitionCoefficient × sqrt(InflatedVariance)
+ DirectionalBonus

### 10. Select Final Candidate

Choose the candidate with the **highest acquisition score** as the next evaluation point.

---

# Hypothesis Framework

## Core Assumptions

- The true maximum lies **within the defined trust region**.
- The **ensemble provides better predictions** than any single model.
- The **SIR direction correctly indicates the ridge of increasing response**.

---

# Expected Outcome if Assumptions Hold

- The selected candidate lies **close to the existing high-value cluster**.
- It shifts **slightly along the SIR direction**.
- The resulting observation **matches or exceeds the current best**.
- Ensemble predictions and SIR direction **agree on the promising region**.

---

# Expected Outcome if Assumptions Break

- If the global optimum lies **outside the trust region**, it will not be discovered.
- If the **ensemble is poorly calibrated**, acquisition scores may be misleading.
- If the **SIR direction is unstable**, the directional bonus could bias sampling incorrectly.

However, the **ensemble mean prediction still provides protection** against severe misdirection.

---

# EDA Audit for f7

## Methods to Remove

### Mapper (Topological Data Analysis)

Mapper has produced **inconsistent results** across weeks:

- Week 10: 4 components with a meaningful cluster
- Week 11: 33 disconnected components with **zero edges**

This instability occurs because:

- Dataset size is small (**N=40**)
- High dimensionality (**6D**)

The topology is therefore **noise-dominated**, and mapper no longer provides reliable guidance.

---

### Isomap 1D Embedding

Isomap consistently shows:

- High-value points cluster around **embedding ≈ −0.75**

However:

- 1D embedding **loses structural information**
- It cannot map back to actionable input changes

Scatter plots and SIR already provide this information more directly.

---

### Integrated Gradients

Integrated Gradients requires **training a neural network**.

At **N=40 in 6D**:

- MLP models are **severely underdetermined**
- Feature attributions depend heavily on **random initialization**

Observed inconsistencies:

| Method | Top Features |
|------|------|
| Integrated Gradients | X1, X2, X4, X5 |
| SIR | X1, X6, X5 |
| Partial Correlation | X1 only |

Thus IG adds **noise rather than information**.

---

# Methods to Add

## 1. GP Leave-One-Out Cross-Validation Residual Plot

Compute **LOO predictions for all 40 points**.

Plot:
LOO residual vs each input dimension

This reveals where the GP **systematically mispredicts**, identifying regions where the surrogate is least trustworthy.

---

## 2. Pairwise Scatter of Top Observations

Plot **all 15 pairwise dimension combinations** using only the **top-15 y observations**.

Visual encoding:

- Colour = y rank
- Point size = value magnitude

Advantages:

- Preserves the **original coordinate system**
- Reveals **true cluster geometry**
- Avoids dimensionality reduction artifacts

---

## 3. Expected Improvement Surface (Top SIR Dimensions)

Procedure:

1. Fix all variables at incumbent values
2. Sweep the **two largest SIR dimensions (X1, X6)** across their ranges
3. Plot **GP-predicted EI surface**

This reveals:

- Whether the trust region is **centred on the EI peak**
- Or located on its **shoulder**

This is the **most actionable diagnostic** when only one or two evaluations remain.

---

# f8 – Tight Neighbourhood GP Mean Maximisation with Cluster-Informed Trust Region

## Objective of Submission
To select the next evaluation point for f8 by maximising the Gaussian Process posterior mean inside a tight trust region centred on the incumbent cluster in X_train_8 and y_train_8, focusing the remaining evaluation budget on resolving the local peak.

---

# 3 Key Assumptions

1. Near-ceiling behaviour  
f8 is already near its ceiling and the true maximum lies inside the incumbent cluster defined by the best points in X_train_8.

2. Low-dimensional activity near the peak  
Only a subset of dimensions are truly active near the peak, so a low-dimensional trust region in the active subspace captures remaining improvement.

3. Locally calibrated GP surrogate  
A carefully refitted ARD Matern Gaussian Process is locally well calibrated, making the posterior mean a reliable ranking function within this neighbourhood.

---

# Research Backing

## Academic Papers Supporting the Strategy

Eriksson et al. (2019)  
Scalable Global Optimization via Local Bayesian Optimization (TuRBO), NeurIPS.  
Shows that shrinking and recentering a trust region around the incumbent is highly efficient near optima.

Eriksson & Jankowiak (2021)  
High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO), UAI.  
Demonstrates that fixing inactive dimensions and searching only active coordinates dramatically improves efficiency.

Jones, Schonlau, Welch (1998)  
Efficient Global Optimization of Expensive Black-Box Functions, Journal of Global Optimization.  
Shows that when the optimum region is known, the GP posterior mean maximiser is the theoretically correct exploitation strategy.

NeurIPS 2020 BBO Challenge Reports (Campa et al., Duxiaoman Team).  
Top-performing teams switched from EI to pure GP mean maximisation once a dense peak was confirmed.

---

# Explorative Principle

The explorative principle is focused local exploitation in a reduced subspace.

Instead of exploring the entire 8-dimensional domain:

1. Identify the active dimensions (e.g., x1, x3, x7)  
2. Define a tight trust region around the incumbent in this subspace  
3. Fix inactive dimensions at incumbent values  
4. Generate dense Sobol candidates in the active region  
5. Select the candidate with the highest GP posterior mean

This is appropriate because:

- f8 already has multiple near-identical best values  
- the function appears to have a dense local peak  
- remaining uncertainty lies primarily along one or two coordinates (e.g., ridge behaviour in x7)

The goal is therefore refining the peak, not discovering new basins.

---

# Black Box Optimization Competition

## Competition
NeurIPS 2020 Black Box Optimization Challenge

## Winning Team
Duxiaoman Financial AI Team (Top-3)

Their strategy included:

- tight local Gaussian Processes  
- posterior mean maximisation  
- trust-region restriction once the optimum basin was identified

This approach directly inspired the f8 strategy.

---

# Why This Strategy Is Ideal for f8

## Expensive Function Evaluations

Evaluations of f8 are expensive, and the dataset already shows:

- best observations are extremely close in value  
- the function is likely near its maximum

Remaining uncertainty is local and concentrated in a small set of active dimensions.

Therefore:

- global exploration is unlikely to improve results  
- budget should be concentrated on resolving the local optimum

Fixing inactive dimensions and searching only a tight 3D subspace maximises the information gained from each remaining evaluation.

---

# Tech Stack

Libraries used

NumPy — numerical computation and data handling  
SciPy (Sobol sequence) — candidate generation  
Scikit-learn — GaussianProcessRegressor, Matern, WhiteKernel, ConstantKernel

---

# Hyperparameters and Settings

## Hyperparameters

- GP kernel structure  
- length-scale bounds per dimension  
- trust region centre  
- trust region widths  
- active dimension set  
- inactive dimension set  
- Sobol candidate count  
- minimum distance guard

---

# Recommended Initial Values

Kernel

ConstantKernel × Matern(ν = 2.5, ARD) + WhiteKernel

Length Scale Bounds

Example

x1 → [0.02, 0.20]  
x3 → [0.02, 0.20]  
x7 → [0.05, 0.80]  
others → wider bounds

Trust Region

Centre  
best row in X_train_8

Widths

x1 → incumbent ±0.12  
x3 → incumbent ±0.12  
x7 → fixed interval [0.08, 0.25]

Inactive dimensions fixed

x2, x4, x5, x6, x8

Candidate Generation

Sobol candidates ≈ 30,000  
Active dimensions = 3

Minimum Distance Guard

distance ≥ 0.03 from existing points

---

# Hyperparameter Tuning

GP hyperparameters are fitted by maximising marginal likelihood with multiple restarts.

Validation steps

1. Inspect learned ARD length scales  
2. Confirm active dimensions have shorter scales  
3. Adjust trust region widths using TuRBO-style rules

Region updates

- slight shrink after success  
- expand only after repeated failures

---

# Entire Flow of the Strategy

## Step-by-Step Exploration Process

1. Identify Incumbent  
Find the best observation in X_train_8 and y_train_8.

2. Identify Active Dimensions  
Based on ARD length scales and earlier analysis.

Example

Active → x1, x3, x7  
Inactive → x2, x4, x5, x6, x8

3. Define Trust Region

x1 → incumbent ±0.12  
x3 → incumbent ±0.12  
x7 → [0.08, 0.25]

Clip to [0,1].

4. Fix Inactive Dimensions  
Inactive coordinates remain equal to incumbent values.

5. Fit ARD Gaussian Process  
Train GP on X_train_8 and y_train_8.

Verify incumbent prediction error is small.

6. Generate Sobol Candidates  
Generate ~30,000 candidates inside the active subspace.

Each candidate is augmented with the fixed inactive dimensions.

7. Apply Distance Guard  
Remove candidates closer than 0.03 to existing points.

8. Compute Posterior Mean  
Use GP to compute μ(x) for all candidates.

9. Select Candidate  
Choose the candidate with maximum μ(x).

10. Diagnostics

Report

- selected point  
- predicted value  
- distance to incumbent  
- movement along active dimensions

---

# Hypothesis Framework

## Core Assumptions

- the true maximum lies within the trust region  
- inactive dimensions can be fixed safely  
- GP surrogate is locally accurate

## Expected Outcome if Assumptions Hold

- candidate differs mainly along active dimensions  
- especially along x7  
- observed value slightly exceeds or matches incumbent  
- improvements are extremely small

This behaviour matches near-ceiling optimisation.

## Expected Outcome if Assumptions Break

- optimum lies outside trust region  
- some inactive dimensions actually matter  
- GP posterior mean misranks candidates

---

# EDA Audit for f8

## Methods to Remove

1. Isomap 1D Embedding  
Remove completely.

Reasons

- dominated by inactive dimensions  
- misleading geometry  
- contributed to boundary exploration failures

2. K-Means Clustering  
Remove entirely.

Silhouette scores 0.11–0.16 indicate no real cluster structure.

3. Full 8×8 Correlation Heatmaps  
Remove matrices.

Replace with ranked xi–y correlation chart.

4. Current TDA Mapper Configuration  
Remove current configuration because the filter function relies on Isomap.

---

# Methods to Add

## 1. Pairwise Active Dimension Scatter Plots

Three plots

x1 vs x3  
x1 vs x7  
x3 vs x7

Points coloured by y value.

Purpose

- visualise active subspace geometry  
- detect boundary vs interior peaks

---

## 2. 1D GP Posterior Probes

For each active dimension

- sweep 200 evenly spaced points  
- fix other dimensions at incumbent

Plot GP mean ±2σ.

Mark

- incumbent coordinate  
- GP mean maximum

---

## 3. GP Posterior Mean Surface Heatmap

Grid

50 × 50 over (x1, x3)  
x7 fixed at probe peak

Panels

- posterior mean  
- posterior sigma

Purpose

visualise acquisition landscape.

---

## 4. GP Posterior Gradient Norm Map

Compute analytic gradient

∇μ(x) = dK(x,X)/dx · K(X,X)^{-1} · y

Plot

- gradient magnitude heatmap  
- direction arrows

Indicates whether the peak is

- interior critical point  
- flat plateau

---

## 5. XGBoost Feature Importance Chart

Compare three signals

ARD inverse length scales  
Mutual information  
XGBoost gain importance

Agreement increases confidence in active dimensions.

---

## 6. ARD Length Scale Tracking Heatmap

Matrix

8 dimensions × optimisation weeks

Cell value

normalised inverse ARD length scale

Shows stability of active dimensions over time.

---

## 7. Regret Curve with Plateau Detection

Plot

best observed y vs week

Add

estimated regret = GP predicted max − best observed

Add threshold

0.1% of observed range

Mark plateau zone once the threshold is crossed.

---

# Keep but Fix

## TDA Mapper

Keep method but change filter function.

Use

- GP posterior mean  
or  
- SIR score

This aligns topology with function value.

---

## Ranked Correlation Bar Chart

Replace full heatmaps with

a horizontal bar chart showing

Pearson, Spearman, and Kendall correlations between xi and y

sorted by absolute Spearman rank.

This preserves the useful information while improving interpretability.