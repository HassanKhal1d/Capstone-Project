# Week 6 — Return to Baselines + Structured Global Exploration

This document outlines the Week 6 submission plan. The strategy shifts away from the warping and residual correction philosophy of Week 5 and adopts a return-to-baseline principle for stagnating functions, alongside structured global exploration for functions confirmed as locally stuck.

Emphasis:

- Return to the simplest strategy that produced the best result for each function  
- Remove unvalidated components that caused regression in Week 5  
- Apply global exploration only where local maxima are strongly indicated  
- Retain only calibration improvements empirically validated in Week 5  

---

## Meta-Strategy

### Return to Validated Baselines + Targeted Global Exploration

**Primary surrogate**

- Single GP per function  
- Kernel configuration matches the setup that produced that function’s best historical result  

**Warping policy**

- No warping unless it was present in the best-performing week  
- Box-Cox retained for f3  
- All other warping removed  

**Residual correction**

- No MLP residual correction  
- Removed globally due to:
  - Catastrophic calibration failure on f4 (Z = 8.75)  
  - Regression on f7  
- Treated as structurally unreliable at small sample sizes  

**Acquisition**

- UCB with function-specific β for functions returning to baseline  
- Full-domain Sobol + UCB + exclusion zone for functions confirmed locally stuck  

**Ensemble policy**

- Ensemble composition modified only where Week 5 demonstrated improved performance on a novel evaluation point  
- No changes based on theoretical diversity arguments alone  

---

## Learning Objectives

1. Does returning to the simplest validated baseline reliably recover performance after a complex strategy fails, or does the additional data from failed weeks materially change GP behaviour?

2. Which functions have genuinely exhausted their local neighbourhood and require global exploration, versus those still recoverable via local refinement?

3. Is MLP failure at small N structural — driven by wide output ranges and limited training data — or coincidental to specific functions tested?

4. Does soft directional weighting over Sobol candidates improve acquisition relative to hard filtering, and does effectiveness vary by dimensionality?

---

## Structural Hypothesis for Week 6

- Simplicity outperforms complexity under small-N conditions.
- Calibration fixes should be retained.
- Architectural experimentation should be paused.
- Exploration should be geometric, not statistical, when local maxima are suspected.

Week 6 prioritises stability, interpretability, and validated configurations over experimentation.


## Function-Specific Strategies

## f1 — Spearman-Guided X2 Exploitation Strategy

### Objective
Directly test whether the Spearman X2 signal discovered in Week 5 represents a real shallow monotonic gradient, rather than a rank artefact caused by near-zero outputs.  
The goal is not to maximise y in absolute terms, but to determine whether high‑X2 points consistently rank higher than previous observations.

### ML Method & Rationale

1. **Gaussian Process (RBF + WhiteKernel)**  
   - Same surrogate as Week 5  
   - GP already shows excellent calibration (NLL ≈ −6, Z ≈ 0.1)  
   - No neural network or ensemble  
   - Sufficient for extremely low-signal function with small dataset (N = 15)

2. **Spearman-Guided Candidate Generation**  
   - Generate 1024 Sobol points in [0,1]²  
   - Apply X2 > 0.65 directional filter  
   - Apply δ_min = 0.10 to avoid duplicates  
   - If too few candidates survive, relax to X2 > 0.55

3. **Adaptive UCB Acquisition**  
   - Use the same β-schedule as Weeks 3–5:  
     \[
     \beta_t = \beta_{end} + (\beta_{start} - \beta_{end}) \exp(-t/T)
     \]  
     - β_start = 2.0  
     - β_end = 0.1  
     - T = 50  
     - At Week 6 → β₆ ≈ 1.79 (still exploratory)  
   - Exploration is appropriate because the X2 gradient is not yet confirmed

4. **Final Selection**  
   - Score all candidates with UCB  
   - Select the argmax(UCB)  
   - Do not apply the flat-surface gate this week (to allow structural testing)

5. **Rank-Based Confirmation Diagnostic**  
   - After receiving y_actual:  
     - Compute the rank percentile of the new point among all 16 values  
     - If rank ≥ 0.80 → X2 gradient confirmed  
   - Absolute magnitude is irrelevant because f1 outputs are near zero

6. **GP Update**  
   - Append the new point  
   - Refit the GP  
   - Recompute Spearman and Kendall correlations  
   - Check whether the X2 signal strengthens or collapses

### Key Assumptions
- Spearman ρ_X2 = 0.646 and Kendall τ_X2 = 0.467 indicate a real monotonic trend  
- Pearson r_X2 = −0.179 contradicts Spearman because the relationship is non-linear  
- MI = 0.0 is unreliable at N = 15  
- X1 remains structurally irrelevant  
- A high‑X2 query is the only reliable way to confirm or falsify the Week 5 signal

### Hypothesis
**If the hypothesis holds:**  
- New point ranks in the top 20% of all observations  
- Updated Spearman ρ_X2 ≥ 0.50  
- Updated Kendall τ_X2 ≥ 0.35  
- f1 is reclassified as weakly structured along X2  
- Week 7 shifts to UCB exploitation with β = 1.0 and X2 > 0.70

**If the hypothesis breaks:**  
- New point ranks in the lower half  
- Spearman weakens or reverses  
- Week 5 signal was a rank artefact  
- f1 is reconfirmed as flat  
- Week 7 restores the flat-surface gate and minimal allocation

### Summary Logic
This is a targeted structural test rather than a verification-only control:  
- One high‑X2 query  
- Rank-based confirmation  
- Reclassify f1 based on whether the monotonic X2 signal survives  

This preserves modelling budget while still respecting the Week‑5 pre‑commitment to test the Spearman signal directly.


---

### f2 — RBF-WhiteKernel GP with Asymmetric Length Scales and High-β UCB
**Objective**: Restore Week‑3 GP structure (RBF + WhiteKernel) with shorter X1 length-scale bounds, relaxed directional filtering, and β = 2.0 UCB to correct persistent upward quantile bias and discover a new best value \(y > 0.682\).

**ML Method & Rationale**

1. RBF + WhiteKernel GP  
   - Restores honest uncertainty lost in Weeks 4–5 when WhiteKernel was removed  
   - Week 3 calibration was optimal  

2. Asymmetric length scales  
   - X1 is dominant → shorter initial length scale and tighter bounds to capture steep gradient  
   - X2 weaker → wider bounds  

3. Directional filter X1 > 0.65  
   - Expands exploration compared with Week 5 (X1 > 0.75)  
   - Correct region for potential improvement  

4. High-β UCB (β = 2.0)  
   - Maintains exploratory pressure until calibration recovers  
   - Prevents anchoring to underestimated mean  

**Key Assumptions**

- WhiteKernel is essential for calibration  
- X1 dominates the response  
- Prior X1 length scales were too long  
- β = 2.0 required until quantile returns toward 0.5  
- X1 > 0.65 is the correct exploitation region  

**Hypothesis**

- **If it holds**: Quantile moves toward 0.5, gap < 0.30, fitted X1 length scale shortens, UCB finds \(y > 0.682\), NLL improves toward Week‑3 values  
- **If it breaks**: Quantile remains near 1.0, X1 > 0.65 region does not improve, WhiteKernel noise hits upper bound, GP may be fundamentally misspecified → consider 1D GP on X1 only  

**Summary Logic**: Restore calibrated GP with correct noise model and X1-focused exploration; maintain high-β UCB to recover Week‑3 performance.


---

### f3 - KNN-Guided Candidate Filtering with GP Uncertainty Scoring (Box-Cox Warped GP + KNN Ensemble)

**Objective of Submission**  
To test whether KNN-guided filtering — selecting the top 30 percent of Sobol candidates using KNN predictions before scoring them with GP-based Expected Improvement (EI) — can locate a point with  
y > −0.034835, breaking a five-week streak of no improvement.

**ML Method & Rationale**

1. **Box-Cox Warped GP (Matérn*ConstantKernel + WhiteKernel)**  
   Box-Cox warping improved calibration for three consecutive weeks.  
   The GP alone cannot capture the local non-linear structure in f3, but it provides uncertainty, which KNN lacks. The GP is kept for EI scoring.

2. **KNN as the Structural Guide**  
   KNN has outperformed the GP every week it was compared:
   - Week 4: KNN error 4.2e‑03 vs GP 6.8e‑02  
   - Week 5: KNN error 1.59e‑02 vs GP 4.00e‑01  

   This is strong evidence that f3 contains local non-linear patterns that KNN captures, which the GP smooths away. KNN defines the promising region, and the GP scores uncertainty within that region.

3. **K = 5**  
   K = 3 was too sensitive to outliers.  
   K = 5 averages 25 percent of the dataset (20 points), providing a stable local estimate.

4. **KNN Volatility Check**  
   Before filtering, compute the interquartile range (IQR) of KNN predictions across all candidates.  
   If IQR < 0.01 → KNN surface is too flat → fallback to x3 < 0.5 directional bias.  
   This prevents silently filtering on noise.

5. **Ensemble Mean**  
   Combine KNN and GP predictions:  
   - 0.70 * KNN  
   - 0.30 * GP  

   This reflects five weeks of evidence that KNN is the superior point estimator.

6. **Expected Improvement (EI)**  
   EI targets improvement over the current best −0.034835, which has held since Week 0.  
   EI is appropriate because the explicit goal is to beat a stubborn incumbent.

**Key Assumptions**

- KNN’s consistent superiority indicates real local structure.  
- The top 30 percent of KNN predictions represent a meaningful high-value region.  
- Box-Cox warping stabilizes GP calibration.  
- K = 5 reduces volatility without oversmoothing.  
- Exploratory filtering across the full Sobol pool is appropriate at Week 6.

**Hypothesis**

- **If the hypothesis holds**  
  - KNN IQR > 0.01 → filter is meaningful  
  - EI selects a point with y > −0.034835  
  - KNN remains the best predictor after update  
  - Box-Cox gap stays below 0.15  

- **If the hypothesis breaks**  
  - KNN IQR < 0.01 → fallback triggers  
  - y does not improve despite filtering  
  - KNN surface misidentifies promising regions  
  - Calibration worsens without the MLP (unlikely but monitored)


---

## f4 — Full-Domain GP-UCB with High-β Exploration and Multi-Restart L-BFGS-B

### Objective of Submission

To test whether returning to the Week 3 GP-UCB baseline, with:

- A slightly higher β (3.5 instead of 3.0)
- More acquisition restarts (40 instead of 25)
- Full-domain search in `[0,1]^4`

can finally beat the long-standing best value **0.0705**, after three consecutive complex strategies failed.

---

### ML Method and Rationale

#### 1. Single GP Surrogate (Matérn 2.5)

Week 3 demonstrated that a simple Gaussian Process with a Matérn kernel was the only model that improved f4.  
Every additional layer introduced since then — TuRBO, SVM masking, MLP correction, variance inflation — degraded calibration or restricted exploration.  
The GP is therefore restored to its cleanest and most stable configuration.

---

#### 2. High-β UCB

β = 3.5 maintains strong exploration pressure.  
At N = 35, the posterior variance naturally contracts. Increasing β counteracts premature exploitation and preserves exploratory behaviour in uncertain regions.

---

#### 3. 40 L-BFGS-B Restarts

As N grows, the UCB acquisition surface becomes increasingly multimodal.  
Using 40 random restarts ensures the optimiser can escape local maxima and more thoroughly explore the acquisition landscape.

---

#### 4. Full `[0,1]^4` Search

TuRBO is removed because it restricts exploration prematurely.  
SVM masking is removed because the decision boundary is unreliable at N = 35.  
The GP must be allowed to explore the entire domain without artificial constraints.

---

#### 5. No MLP, No Inflation

The MLP previously caused severe prediction instability and is removed entirely.  
Variance inflation is also removed until raw z-scores demonstrate a systematic calibration failure.

---

### Key Assumptions

- The Week 3 improvement was structural, not random.
- β = 3.5 is appropriate for N = 35.
- 40 restarts are sufficient to explore the UCB landscape.
- Full-domain search is necessary because no improvement has occurred in three weeks.
- EDA shows all features negatively correlated with the target → UCB should naturally explore low-x1, low-x2, low-x4 regions.

---

### Hypothesis

#### If the Hypothesis Holds

- UCB selects a point in the low-x1, low-x2, low-x4 region.
- Observed \( y > 0.0705 \) (first improvement since Week 3).
- Raw \( |z| < 2.0 \) → GP posterior is well-calibrated.
- Fitted length scales reflect EDA importance (x1, x2 shortest; x3 longest).

---

#### If the Hypothesis Breaks

- \( y < 0.0705 \) → The GP may already be near the true optimum.
- Raw \( |z| > 2.0 \) → GP is overconfident → WhiteKernel or inflation may be required.
- UCB repeatedly pushes to domain corners → Gradient structure may be misleading.

---

## f5 - Full-Domain Log-GP with EI and 50-Restart L-BFGS-B (X3 Boundary Exploitation)

This submission captures the essence of:

- Log-transformed outputs
- Single Matérn GP
- EI acquisition
- 50 continuous restarts
- Full [0,1]⁴ domain
- Natural exploitation of the strong X3 → 1.0 gradient


### Objective of Submission

To test whether returning to the Week-3 log-GP baseline — with:

- No warp
- No ensemble
- No MLP
- No inflation
- 50 L-BFGS-B restarts
- Full-domain continuous optimization

Allows EI to rediscover the high-value region near [0.899, 0.964, 1.000, 0.792] and find y > 4171.1, the best observed since Week 3.


### ML Method & Rationale

1. **Log Transform**  
   - f5 spans several orders of magnitude  
   - Log-transform stabilizes variance and makes the GP prior appropriate  
   - This is exactly what worked in Week 3  

2. **Single GP (Matérn 2.5)**  
   - Week 3 showed the GP alone was:  
     - Calibrated  
     - Stable  
     - Able to follow the strong positive gradient  
     - Able to find the best point  
   - Every added layer since then (SVR, MLP, warp, inflation) degraded performance  

3. **No Warp**  
   - Yeo-Johnson on X3 compressed the gradient that EI needs to follow to reach X3 = 1.0  
   - The Week-3 code had no warp and found the best point  

4. **EI Acquisition**  
   - EI is ideal when:  
     - The goal is to beat a known best  
     - The function is smooth  
     - The gradient is strong and monotonic  
   - EI naturally pushes toward high-X3, high-X2, high-X4 regions  

5. **50 L-BFGS-B Restarts**  
   - The EI surface in 4D is highly multimodal  
   - 50 restarts ensure the optimizer can rediscover the Week-3 region  

6. **Full-Domain Search**  
   - No TuRBO, no SVM, no masking  
   - The GP must be allowed to explore the entire domain  


### Key Assumptions

- The Week-3 improvement was structural, not luck  
- X3 is the dominant driver (partial r = 0.638)  
- The Yeo-Johnson warp distorted the true gradient  
- The GP alone is well-calibrated in log space  
- EI with 50 restarts will naturally push toward X3 → 1.0  
- The high-value region near the Week-3 point still exists  


### Hypothesis

**If the hypothesis holds:**

- EI pushes X3 → 1.0 again  
- Next_x is near the Week-3 region  
- y > 4171.1  
- Raw Z < 1.5 (GP well-calibrated)  
- Fitted X3 length scale is the shortest (kernel confirms EDA)

**If the hypothesis breaks:**

- EI does not push toward the Week-3 region → landscape shifted  
- y < 4171.1 but near the same region → tighten search locally  
- y << 4171.1 → GP not capturing the high-value region  
- X3 not near boundary → gradient structure may be weaker than EDA suggests  


---

## f6 — Full-Domain 5D ARD-GP with UCB and Sobol Exploration

This submission implements:

- Full 5D search  
- ARD Matérn kernel  
- WhiteKernel for calibration  
- UCB acquisition  
- Sobol candidate generation with δ-min filtering  


### Objective of Submission

Test whether returning to a full 5D ARD-GP with UCB, without fixing any dimensions, allows the model to:

- Rediscover combinations that beat the long-standing best value −0.5649  
- Confirm the active subspace through learned length scales  
- Exploit strong signals in X4 and X5 while allowing X2 to vary freely  

Previous four constrained strategies failed to improve results. This submission evaluates whether removing constraints and letting the GP learn the structure directly is the correct approach.


### ML Method & Rationale

1. **Full 5D ARD-GP**  
   - Learns a separate length scale for each dimension  
   - Confirms active subspace (short scales for X4, X5)  
   - Detects emerging X2 signal  
   - De-emphasizes weak dimensions (long scales for X1, X3)  
   - More principled than manually fixing dimensions  

2. **WhiteKernel**  
   - Absorbs observation noise and prevents overconfidence  
   - Fixes calibration failures observed in f2  

3. **Sobol Candidates + δ-Min Filtering**  
   - Sobol ensures good coverage of the 5D space  
   - δ-Min prevents candidates from being too close to existing points  

4. **UCB Acquisition**  
   - Function has not improved in four weeks, exploration needed  
   - β = 3.0 balances exploration and exploitation at N = 25  

5. **No Warping**  
   - Week 2 best was found without warping  
   - Yeo-Johnson on X4 and X5 may distort gradient structure  


### Key Assumptions

- Fixing X1–X3 prevented exploring combinations that could beat −0.5649  
- ARD will confirm the active subspace (short length scales for X4 and X5)  
- Emerging partial correlation in X2 justifies allowing it to vary  
- UCB with β = 3.0 is appropriate at N = 25  
- WhiteKernel maintains calibration in 5D  


### Hypothesis

**If the hypothesis holds**:  

- ARD length scales: X4 and X5 shortest, X1 longest  
- y > −0.5649 (first improvement since Week 2)  
- Z-score < 1.5 and CI pass  
- X2 length scale shorter than X1 and X3  

**If the hypothesis breaks**:  

- y < −0.5649 but ARD confirms X4–X5 → return to fixed X1–X3 with finer grid  
- y < −0.5649 and ARD uniform → GP not detecting structure; increase `n_restarts_optimizer`  
- WhiteKernel noise hits upper bound → widen noise bounds  


---

## f7 — Four-Model Conservative Ensemble with SVR Restored and Sobol-Weighted Acquisition


### Objective

To test whether restoring the Week‑4 four-model ensemble (GP, SVR, KNN, XGB) with:

- CV-based weighting  
- 3× variance inflation  
- raw input space  
- Sobol candidates  
- soft directional weighting  

recovers the performance that produced y = 1.5985 and finds y > 1.5985.


### ML Method & Rationale

1. **Restore SVR**  
   - SVR was the best predictor in Week 4  
   - Its removal in Week 5 caused sharp regression  
   - Restoring it recovers the ensemble’s strongest signal  

2. **Restore KNN and XGB**  
   - Non-parametric, no stationarity assumptions  
   - F7 is diffuse in 7D, so these models capture structure GP and MLP cannot  

3. **Remove Standardisation**  
   - Week 4 fitted all models on raw inputs and succeeded  
   - Standardisation in Week 5 altered geometry and suppressed uncertainty  

4. **Sobol Candidates**  
   - Sobol sequences fill 7D space more uniformly than random  
   - Reduces the chance of missing high-value regions  

5. **Soft Directional Weighting**  
   - Week 5 directional bias was correct but too aggressive  
   - Applied as a soft multiplicative weight on the acquisition, not candidate generation  

6. **3× Variance Inflation**  
   - Week 4 calibration was good with 3× inflation (Z = 1.01)  
   - Retained for stability  


### Key Assumptions

- SVR removal caused Week 5 regression  
- KNN and XGB provide essential non-parametric diversity  
- Raw input space is better for F7’s diffuse structure  
- CV weighting is more stable than LOO at n = 35  
- 3× inflation remains appropriate  
- Directional bias should influence acquisition, not candidate generation  


### Hypothesis

- **If the hypothesis holds:**  
  - SVR receives high CV weight again  
  - KNN receives highest weight again  
  - Ensemble mean is closer to truth than any single model  
  - y > 1.5985  
  - Z < 1.5 and CI pass  

- **If the hypothesis breaks:**  
  - CV weights shift → new data changed the landscape  
  - y < 1.5985 but Gap small → calibration stable, increase candidates  
  - y < 1.5985 and Gap large → increase inflation to 4×  


---

## f8 - Full-Domain Sobol Exploration with ARD-Informed UCB Weighting


### Objective

To query a point **outside** the previously explored neighbourhood of the Week 3 best (9.9487), using:

- Full-domain Sobol exploration  
- ARD-GP mean and uncertainty  
- UCB with β = 3.0  
- Soft weighting favouring low X1, low X3, low X7  
- Exclusion zone to guarantee novelty  

The goal is to test whether higher values exist elsewhere in the 8D domain, or whether 9.9487 is the global maximum.


## ML Method & Rationale

### 1. Abandon Local Search

- Three consecutive weeks failed to beat 9.9487.  
- Week 5 Thompson Sampling moved deeper into the low-X1, low-X3, low-X7 corner and returned a worse value.  
- This falsifies the assumption that the optimum lies further in that direction.  

Shift from local refinement to global exploration.


### 2. Full-Domain Sobol Exploration

- Sobol sequences provide uniform coverage of the full 8D space.  
- Reduces clustering bias.  
- Enables discovery of new high-value regions.  


### 3. ARD-GP with WhiteKernel

- ARD identifies globally relevant dimensions.  
- WhiteKernel prevents variance collapse in sparsely sampled regions.  
- Preserves uncertainty in unexplored areas.  


### 4. UCB with β = 3.0

- High β promotes exploration.  
- Appropriate when querying far from the training cluster.  
- Balances mean exploitation with uncertainty-driven discovery.  


### 5. Soft Directional Weighting

- Active subspace (X1, X3, X7) remains structurally validated.  
- Reward candidates with low X1, low X3, low X7.  
- Do not restrict the search to that region.  

Directional bias influences ranking, not feasibility.


### 6. Exclusion Zone

- Remove candidates within distance 0.20 of any training point.  
- Guarantees novelty.  
- Prevents re-sampling the known local basin.


## Key Assumptions

- Week 3 best (9.9487) is a local maximum, not the global maximum.  
- Active subspace (X1, X3, X7) generalises across the full domain.  
- Unexplored regions contain y > 9.9487.  
- GP uncertainty identifies exploration-worthy regions.


## Hypothesis

### If the Hypothesis Holds

- Selected point lies in a new region of the 8D space.  
- X1, X3, X7 differ significantly from the Week 3 cluster.  
- Predicted y is high due to strong mean or high uncertainty.  
- Actual y > 9.9487.


### If the Hypothesis Breaks

- y < 9.9487  
  → Week 3 region likely contains the global maximum.  

- y < 9.5  
  → GP uncertainty misled exploration; reduce β to 2.0 next week.  

- Selected point has X1, X3, X7 near 0.5  
  → Directional weighting too weak; introduce explicit constraints next week.
