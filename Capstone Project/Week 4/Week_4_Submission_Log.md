# Week 4 – Learning-Focused Submission Strategy (SVM/KNN Integration)

This document outlines the Week 4 submission plan, integrating **SVM, KNN, and GP-based surrogates** for both learning and competition objectives. Emphasis is on **ensemble learning, cross-validation, and method diversity**.

---

## Meta-Strategy: Ensemble Learning + Cross-Validation + Method Diversity

**Core Philosophy**

- **Primary surrogate:** GP-based (maintains Bayesian Optimization theoretical guarantees)
- **Complementary methods:** SVM, KNN (to learn strengths/weaknesses)
- **Validation:** 5-fold CV on all surrogates (robust candidate selection)
- **Ensemble:** Weighted combination based on CV performance

**Learning Objectives**

1. When does SVM outperform GP in BO?
2. When does KNN provide useful local information?
3. How does CV-based candidate selection compare to pure acquisition optimization?
4. Which surrogate types work best for which function characteristics?

---

# Function-Specific Strategies


---

## Strategy Name
**Minimal Surrogate Comparison Study**


## Objective of This Submission

Evaluate whether any surrogate model (**GP, SVM, KNN**) can detect structure on a function that **Week-3 analysis already confirmed is essentially flat**.  
This establishes a **baseline** for comparing model behaviour when the **true signal is near zero**.


## ML Method & Rationale

### Models Used

#### 1. Gaussian Process (RBF Kernel)
- Smooth prior, suitable for detecting **subtle structure**
- Expected behaviour: **predict ~0 everywhere** if the surface is truly flat


#### 2. ε-SVR (RBF Kernel)
- Tests whether a **margin-based model hallucinates structure**
- `ε = 0.1` chosen to **avoid overfitting tiny noise fluctuations**


#### 3. KNN Regressor (k = 5)
- **Non-parametric baseline**
- Expected behaviour: **averages neighbours → ~0 prediction**


### Rationale

- Using **three fundamentally different model families** checks whether any method **falsely detects structure**.
- **5-fold Cross-Validation** evaluates prediction **stability when the true signal is absent**.
- **Equal-weight ensemble** ensures **no single model dominates** conclusions.
- A **random interior sampling point** is selected because **no acquisition function is meaningful on a flat surface**.


## Key Assumptions

- **f1 is flat**, supported by Week-3 diagnostics:  
  - **Negative Log Likelihood (NLL): −6.73**  
  - **Z-score: 0.08**
- All surrogate models should predict **approximately zero everywhere**.
- Any disagreement between models arises from **modelling artefacts**, not genuine structure.


## Hypothesis

### If the Hypothesis Holds (Expected Behaviour)
- All models produce predictions **≈ 0**
- **Cross-Validation MSE** is **extremely low** for all three models
- **Ensemble prediction** remains **stable and near zero**
- **Random interior sampling** is sufficient


### If the Hypothesis Breaks (Unexpected Behaviour)
- Models **disagree significantly**
- **CV error spikes** for one or more models
- **SVM or GP detects a false trend**, indicating possible:
  - **Overfitting**
  - **Kernel mis-specification**
  - **Hidden structure missed in earlier analysis**

This outcome would signal the need to **revisit assumptions about f1** and reassess earlier diagnostics.


## Summary Logic

- **GP →** Sensitive detector of smooth global structure  
- **SVM →** Tests for margin-based false positives  
- **KNN →** Local averaging sanity check  
- **CV →** Stability validation under near-zero signal  
- **Random Sampling →** Appropriate when optimisation signal is absent

This study functions as a **control experiment**, ensuring later surrogate-based optimisation results are interpreted relative to a **known flat baseline**.


---

# Strategy Name  
**f2 — SVM–GP Ensemble with Thompson Sampling**


## Objective of This Submission

Test whether an **SVM with an RBF kernel** can capture the **X1 monotonic trend** as well as (or better than) a **Gaussian Process (GP)**, and whether an **ensemble of GP + SVM + KNN**, weighted by **cross-validated performance**, can guide a **Thompson Sampling–style selection** that improves on previous best values.


## ML Method & Rationale

### Ensemble Components

#### 1. Heteroskedastic-Style Gaussian Process (Matérn 2.5 Kernel)
- Models **smooth but slightly rough** response functions  
- Provides **predictive uncertainty (σ)** required for Thompson Sampling  
- Serves as the primary global surrogate


#### 2. ε-SVR with RBF Kernel
- Effective at learning **smooth monotonic trends**, especially along **X1**
- **Grid search over `C` and `ε`** to balance underfitting vs. overfitting
- Acts as a structural competitor to GP rather than just a supplement


#### 3. KNN Regressor (Distance-Weighted)
- **Simple local model**
- Captures **neighbourhood behaviour** near current best points
- Helps reduce global model bias in dense data regions


### Rationale for Ensemble

- **SVM:** Explicitly tests whether the **X1 monotonic structure**  
  - Observed in EDA metrics:  
    - Pearson ≈ 0.408  
    - Kendall ≈ 0.256  
    - Mutual Information ≈ 0.215  
  is captured as effectively as by the GP.

- **KNN:** Acts as a **local sanity check**, improving behaviour near dense or high-value regions.

- **CV-Based Weighting:**  
  - Models with **lower CV-MSE receive higher weights**  
  - Prevents weak learners from degrading ensemble quality

- **Thompson Sampling:**  
  - Uses **GP uncertainty + ensemble mean**  
  - Maintains a **principled exploration–exploitation balance**


## Key Assumptions

- The **X1 monotonic structure is genuine**, not an artefact  
  (supported by correlation and mutual information metrics).
- An **RBF SVM** can effectively model this **smooth monotonic trend**.
- A **Matérn 2.5 GP** is a reasonable prior for the response surface.
- **CV-MSE** is a reliable proxy for model quality in a **small-data regime**.


## Hypothesis

### If the Hypothesis Holds
- **SVM CV-MSE** is **comparable to GP** (not significantly worse)
- **Ensemble predictions are stable**
- **Thompson Sampling** identifies a point with **improved y** compared to previous weeks
- **KNN** contributes positively near the **current best region** without overfitting


### If the Hypothesis Fails
- **SVM CV-MSE** is **substantially worse than GP**, indicating poor structural capture
- SVM predictions **inject noise** into the ensemble and are **down-weighted by CV**
- **Thompson Sampling fails to improve y**, suggesting:
  - Surrogate family may be unsuitable, or  
  - Candidate search space requires revision


## Summary Logic

- **GP →** Global smooth structure + calibrated uncertainty  
- **SVM →** Explicit monotonic trend learner along X1  
- **KNN →** Local corrective mechanism  
- **CV Weighting →** Empirical performance control  
- **Thompson Sampling →** Probabilistic decision strategy

This strategy evaluates whether **structural bias (SVM)** combined with **probabilistic modelling (GP)** and **local correction (KNN)** yields more reliable optimisation than any single surrogate alone.


---

# Strategy Name  
**f3 — Robust GP–KNN Portfolio with CV-Weighted Lower Confidence Bound**


## Objective of This Submission

To build a **noise-robust surrogate ensemble** using:

- **ARD-GP (anisotropic kernel)**
- **KNN local regression**
- **Cross-validated uncertainty estimates**

and select the next query point using a **CV-weighted Lower Confidence Bound (LCB)** that avoids being misled by noise.

This replaces the previously unstable **SVM-classification approach**.


## ML Method & Rationale

### 1. ARD-GP (Matérn 2.5 Kernel)
- Learns **different length-scales per dimension**
- Captures **anisotropic structure**  
  - Important for **f3**, where `X2` / `X3` show higher relevance
- Provides **predictive mean + predictive uncertainty**

---

### 2. KNN (k = 5, Distance-Weighted)
- **Non-parametric**
- Very **stable under noise**
- Effective for **local refinement**
- Complements GP by reducing over-smooth bias

---

### 3. Cross-Validated Uncertainty Estimation
For each model:

- Perform **5-fold cross-validation**
- Compute **CV MSE**
- Convert MSE → **standard deviation estimate**
- Use this as a **robust uncertainty proxy**

**Why?**  
GP posterior variance alone can be miscalibrated under noise; CV error reflects *empirical* predictive risk.

---

### 4. Ensemble Lower Confidence Bound (LCB)

For each candidate point \( x \):

\[
LCB(x) =
\frac{1}{2}
\Big[
(\mu_{GP} - \sigma_{GP,cv}) +
(\mu_{KNN} - \sigma_{KNN,cv})
\Big]
\]

We **maximize this LCB** because:

- **High mean → promising**
- **Low uncertainty → reliable**
- **CV-based σ → noise-robust**

This is particularly suitable for **noisy f3** surfaces where UCB/EI may over-exploit variance spikes.

---

## Key Assumptions

- **f3** has **moderate structure** but **high noise** (Calibration Gap ≈ 0.30)
- **ARD-GP** captures anisotropy (`X2`, `X3` more informative)
- **KNN** provides **stable local predictions**
- **CV-based uncertainty** is more reliable than raw GP variance
- **LCB** is safer than **UCB / EI** in noisy optimisation regimes

---

## Hypothesis

### If the Hypothesis Holds
- GP and KNN predictions **agree in high-value regions**
- **CV-LCB** identifies a **stable, promising point**
- The next query **improves previous best values**
- **Interval widths shrink** in informative regions

---

### If the Hypothesis Breaks
- GP and KNN **disagree strongly**
- CV intervals are **wide everywhere**
- LCB becomes **uniformly low**
- No improvement occurs → **f3 may be too noisy or effectively flat**

---

## Summary Logic

- **GP** → global structure + smooth uncertainty  
- **KNN** → local stability + noise resistance  
- **CV weighting** → empirical risk control  
- **LCB acquisition** → conservative, reliability-focused optimisation

This portfolio balances **exploration safety** with **local exploitation**, prioritising *decision reliability over aggressive gain*.


---

## Strategy Name
**f4 - TURBO-SVM Trust-Region Optimisation with Variance-Inflated ARD-GP**


## Objective of This Submission

Stabilise optimisation on **f4** by combining:

- **TURBO** (local trust-region Bayesian optimisation)
- **SVM support-vector geometry** (to identify *safe / promising* regions)
- **Variance-inflated ARD-GP** (to avoid overconfidence)

This strategy directly addresses **Week-3 global GP miscalibration** (`Z = 2.17`) and aims to achieve:

- **|Z| < 1.5**
- **Calibration Gap < 0.35**


## ML Method & Rationale

### Stage 1 — SVM Boundary Detection

**Procedure**

- Label data points:
  - **Class-1:** `y > 60th percentile`
  - **Class-0:** otherwise
- Train **RBF SVM classifier** (`C = 5`)
- Extract **support vectors** → approximate the boundary between *good* and *bad* regions
- Validate with **5-fold Cross-Validation accuracy**

**Why?**

Support vectors approximate the **geometry of promising regions**.  
These geometric constraints help TURBO **avoid exploring known-poor areas**.


### Stage 2 — TURBO with SVM-Constrained Trust Region

**Procedure**

1. Initialise **TURBO trust region**  
   - Radius = `0.2`
   - Centered at the **current best point**

2. **Constrain** trust region to remain **inside SVM Class-1 region**

3. Fit an **ARD-Gaussian Process** using only points **inside the trust region**

4. Apply **3× Variance Inflation**: This counters Week-3’s **catastrophic variance underestimation**.

5. Optimise **Expected Improvement (EI)** within the constrained region

6. Validate GP stability using **5-fold CV-MSE**


### Why This Works

- **TURBO** → prevents uncontrolled global exploration  
- **SVM** → keeps optimisation within *safe / high-value* geometry  
- **Variance Inflation** → reduces overconfidence risk  
- **Cross-Validation** → ensures local GP stability  


## Key Assumptions

- Week-3 failure resulted from **global GP miscalibration**, not absence of structure
- **f4 exhibits locally smooth behaviour**, making trust-region modelling viable
- SVM can **meaningfully separate** promising vs poor regions
- **Variance inflation is necessary** to mitigate overconfidence
- **Local GP modelling** is more stable than global GP modelling


## Hypothesis

### If the Hypothesis Holds

- **SVM boundary is stable** (`CV accuracy > 0.65`)
- TURBO remains within **promising regions**
- **GP CV-MSE** inside the trust region is **low**
- **Expected Improvement** finds meaningful gains
- Target metrics achieved:
- **|Z| < 1.5**
- **Gap < 0.35**


### If the Hypothesis Breaks

- **SVM boundary unstable** → support vectors meaningless
- TURBO trust region **collapses or drifts**
- **GP CV-MSE remains high**, even locally
- **Expected Improvement fails**, implying:
- f4 is too complex
- Smooth surrogate modelling is insufficient


## Conceptual Summary

| Component | Role |
|--------|------|
| **SVM** | Geometric boundary detector |
| **TURBO** | Local exploration controller |
| **ARD-GP** | Anisotropic local modelling |
| **Variance Inflation** | Overconfidence safeguard |
| **Cross-Validation** | Stability verification |

This hybrid strategy aims to **replace unstable global optimisation** with a **controlled, geometry-aware, uncertainty-robust local search**.


---

## Strategy Name
**f5 - Log-GP + Log-SVR Ensemble with Batch EI and CV-Robust Candidate Selection**

## Objective of This Submission

Evaluate whether an **SVM trained on log-transformed targets** can match the performance of a **log-GP**, and whether **cross-validated selection** from a batch of **EI-generated candidates** yields more reliable improvement than pure EI optimisation.

This directly tests whether SVM adds value on a function where the GP was already **well-calibrated in Week 3**.

## ML Method & Rationale

### 1. Log-Transform the Targets
\[
y' = \log(y + 1)
\]

**Why?**

- f5 has **large-magnitude outputs**
- **Log-transform stabilises variance**
- Improves **GP and SVM behaviour**
- Avoids **negative values inside the log**

### 2. Log-GP (Matérn 2.5 Kernel)
- Smooth and flexible
- **Well-calibrated in Week 3** (`Z = 0.46`, Gap = 0.18)
- Provides **uncertainty estimates** required for Expected Improvement (EI)

### 3. Log-SVR (ε-SVR, RBF Kernel)
- Models **smooth, monotonic, large-magnitude structure**
- Works well with **log-transformed targets**
- Provides a **complementary inductive bias** to GP

### 4. 5-Fold Cross-Validation
- Compute **CV-MSE** for both models
- Compute **ensemble weights** using inverse CV-MSE:

\[
w_i = \frac{1 / \text{MSE}_i}{\sum_j 1 / \text{MSE}_j}
\]

**Purpose:**

- Better models get **higher weight**
- Poor models do **not degrade the ensemble**

### 5. Batch EI (5 Candidates)
- Use **Sobol multi-start**
- Generate **5 EI-optimised candidates**
- Benefits:
  - **Increases diversity**
  - Reduces **acquisition-function artefacts**

### 6. CV-Based Candidate Selection
For each candidate:

1. Compute **5-fold CV predictions** for both models
2. Compute **CV-based prediction intervals**
3. Compute **CV-Lower-Bound (CV-LCB)**:

\[
LCB(x) = \mu_\text{ens}(x) - \sigma_\text{CV}(x)
\]

- Select the candidate with **highest LCB**

**Why?**

- Avoids **overfitting to EI**
- Robust to **noise**
- Ensures the chosen point is **reliably high-performing**

## Key Assumptions

- **Log-GP** is well-calibrated (supported by Week 3 metrics)
- **Log-SVR** can capture similar structure
- **CV-MSE** is a reliable **model-quality indicator**
- **Batch EI** increases candidate diversity
- **CV-LCB** prevents acquisition-function over-optimisation

## Hypothesis

### If the Hypothesis Holds
- **Log-SVR CV-MSE ≈ Log-GP CV-MSE**
- Ensemble predictions are **stable**
- **CV-selected candidate** outperforms pure EI
- CV intervals are **narrow** for the chosen point

### If the Hypothesis Breaks
- **Log-SVR CV-MSE >> Log-GP CV-MSE**
- Ensemble **collapses to GP**
- CV-selected candidate ≈ **pure EI candidate**
- SVM adds **computation but no value**

## Summary Logic

| Component | Role |
|-----------|------|
| Log-GP    | Global smooth surrogate + uncertainty |
| Log-SVR   | Captures complementary monotonic structure |
| CV weighting | Ensures ensemble reliability |
| Batch EI  | Candidate diversity and noise mitigation |
| CV-LCB    | Safely selects reliable next query point |

This ensemble strategy prioritises **robust, noise-aware optimisation** for a **high-magnitude function** with known well-calibrated GP behaviour.


---

## Strategy Name
**f6 - REMBO + KNN Local Refinement in a 2D Embedded Space**


## Objective of This Submission

Leverage a **random low-dimensional embedding (REMBO)** with a **2D GP+KNN ensemble** to exploit the fact that **f6 has effective dimensionality 2** (only X4/X5 matter).  

This submission tests whether:

- Modelling in a **2D embedded space**  
- Using a **GP + KNN ensemble**  

improves **calibration and local prediction quality** compared to working directly in the original 5D space.


## ML Method & Rationale

### 1. REMBO: Random Embedding from 2D → 5D

- Assumes f6 depends only on **2 effective dimensions** (MI analysis: X4/X5 active)  
- Generate a **random linear map**:

\[
A : \mathbb{R}^2 \to \mathbb{R}^5, \quad x = A z
\]

- To embed existing 5D points into 2D:

\[
z = A^+ x
\]

- **Why?**
  - If the function truly lives in a 2D subspace, modelling there is:
    - Easier
    - Less noisy
    - Better for KNN (avoids curse of dimensionality)  
  - REMBO is designed for **low effective dimension in high ambient dimension** problems


### 2. GP + KNN Ensemble in 2D

- **GP-2D:** Matérn kernel GP on \( z \in \mathbb{R}^2 \)  
- **KNN-2D:** k=5, distance-weighted KNN on the same 2D points  
- **KNN-5D:** KNN on original 5D inputs to test 2D vs 5D performance

- Compute **5-fold CV-MSE** for:
  - GP-2D
  - KNN-2D
  - KNN-5D

- Construct **2D ensemble**:

\[
\text{weights} = \text{inverse CV-MSE} \quad (\text{better model → higher weight})
\]

\[
\mu_\text{ens} = \text{weighted average of GP-2D and KNN-2D means}
\]

**Why?**

- GP provides **smooth, uncertainty-aware predictions** in 2D  
- KNN is **strong locally** in low dimensions  
- CV indicates **which model performs better**  
- Ensemble combines them **without arbitrary weighting**


### 3. UCB Optimisation in 2D and Mapping Back to 5D

- In 2D:
  - GP-2D provides **mean** \( \mu(z) \) and **std** \( \sigma(z) \)  
  - Ensemble mean \( \mu_\text{ens}(z) \) from GP + KNN  
  - Define **Upper Confidence Bound (UCB)**:

\[
\text{UCB}(z) = \mu_\text{ens}(z) + \beta \sigma(z), \quad \beta = 1.0
\]

- Search over many random 2D candidates → pick **highest UCB**  
- Map back to 5D:

\[
x_\text{next} = A z_\text{best}
\]

- Clip \( x_\text{next} \) into \([0,1]^5\)

**Why UCB?**

- Balances **exploration and exploitation** in the embedded space  
- Rewards **high mean** and **high uncertainty**, ideal for BO in low-dimensional subspace


## Key Assumptions

- Effective dimensionality = 2 for f6 (X4/X5 dominate)  
- Random 2D subspace (REMBO) captures relevant structure  
- KNN performs better in 2D than in 5D (curse of dimensionality)  
- GP-2D + KNN-2D ensemble is better calibrated than GP-5D alone  
- Projection matrix \( A \) is **not pathological** (doesn’t destroy important directions)


## Hypothesis

### If the Hypothesis Holds
- **KNN-2D CV-MSE ≪ KNN-5D CV-MSE** → projection retains structure  
- GP-2D + KNN-2D ensemble shows **better calibration** (Gap < 0.35)  
- UCB in 2D finds a strong candidate improving previous best  
- GP and KNN agree in 2D (small disagreement)


### If the Hypothesis Breaks
- KNN-2D CV-MSE ≈ or > KNN-5D → projection loses critical information  
- GP-2D + KNN-2D ensemble **does not outperform GP-5D**  
- UCB in 2D fails to improve → f6 not captured by simple 2D embedding  
- Conclusion: **full 5D ARD-GP is preferable**


## Summary Logic

| Component      | Role |
|----------------|------|
| REMBO          | Random low-dimensional embedding |
| GP-2D          | Smooth surrogate with uncertainty in 2D |
| KNN-2D         | Strong local predictions in 2D |
| KNN-5D         | Baseline for 5D comparison |
| CV-MSE weighting | Data-driven ensemble weighting |
| UCB in 2D      | Exploration + exploitation for next candidate |

This strategy evaluates whether **low-dimensional embedding + local refinement** improves BO performance and calibration for a **function with low effective dimensionality**.


---

## Strategy Name
**f7 - Ultra-Conservative Stacked Ensemble with Aggressive Cross-Validation**

## Objective of This Submission

For **f7**, the goal is **not to trust any single surrogate** (Week 3 failed catastrophically: `Z = -4.11`).  

Instead, we aim to:

- Build a **diverse ensemble** of very different models  
- Use **aggressive cross-validation** to measure both error and stability  
- Construct a **pessimistic, variance-inflated acquisition** that is unlikely to collapse  

Key question:  
*"When structure is weak and noisy, which model types behave best, and can a conservative ensemble avoid another disaster?"*

## ML Method & Rationale

### 1. Diverse Base Models (Maximising Inductive Diversity)

**Models used:**

- **GP-Matérn**: Smooth, probabilistic, uncertainty-aware. Standard BO surrogate  
- **ε-SVR (RBF)**: Margin-based, kernelised, good for smooth but non-Gaussian structure  
- **KNN**: Local, non-parametric, no global assumptions  
- **XGBoost**: Tree-based, good for non-smooth, piecewise structure  

**Rationale:**  
Week 3 showed no single model is reliable on f7 → maximize model diversity to cover as many functional types as possible.

### 2. Aggressive Cross-Validation (Error + Stability)

- Run **5-fold CV** per model, compute:
  - **CV-MSE** → average error across folds  
  - **CV-variance** → how much error changes across folds (stability)  

- Define ensemble weights:

\[
w_i = \frac{1}{\text{MSE}_i + \lambda \cdot \text{Var}_i}, \quad \lambda = 0.5
\]

- Normalize weights to sum to 1

**Rationale:**  
- Want models that are **both accurate and stable**  
- Penalising **error + instability** ensures the ensemble is **conservative**  
- Avoids trusting models that occasionally fail catastrophically

### 3. Meta-Ensemble with Variance Budget

- **Ensemble mean** = weighted average of 4 models’ predictions  
- **Ensemble variance**:

\[
\text{Var}_\text{budget} = 3.0 \times \max(\sigma_\text{GP}, \sigma_\text{KNN,local})
\]

- Acquisition function:

\[
\text{Acq}(x) = \mu_\text{ens}(x) + 1.5 \cdot \text{Var}_\text{budget}(x)
\]

**Rationale:**  
- Avoid Week 3’s overconfidence  
- Inflate uncertainty deliberately → never trust a single model too much  
- Acquisition remains **optimistic but with a large safety margin**

### 4. Stability Diagnostics

- Compute per-model **CV-MSE** and **CV-variance** → identify most/least stable models  
- Compute **pairwise disagreement** between models → disagreement indicates uncertain or complex regions  

**Rationale:**  
- f7 is weakly structured  
- Disagreement signals **“interesting but risky” areas** for further exploration

## Key Assumptions

- No single surrogate is reliable on f7 (Week 3 `Z = -4.11`)  
- **Model diversity + pessimistic variance** is safer than relying on one model  
- **CV-MSE and CV-variance** are reliable indicators of model quality and stability  
- **GP and KNN** provide complementary uncertainty signals  
- The surface is not pure noise—some structure exists to exploit

## Hypothesis

### If the Hypothesis Holds
- **Ensemble CV-error < best single model CV-error**  
- |Z| < 2.5 (avoiding catastrophic miscalibration)  
- Ensemble weights concentrate on **most stable models**  
- Disagreement is **moderate in good regions**, high in unexplored regions

### If the Hypothesis Breaks
- All models have **high CV-error and high CV-variance**  
- Ensemble performs **no better than any single model**  
- |Z| remains large → surface may be **near-pure noise or extremely complex**  
- Conclusion: f7 is **beyond what these surrogates can handle**

## Summary Logic

| Component            | Role |
|----------------------|------|
| Diverse base models  | Cover multiple function types and inductive biases |
| Aggressive CV        | Measures both accuracy and stability |
| Weighted ensemble    | Conservative predictions weighted by error + variance |
| Variance budget      | Inflates uncertainty to avoid overconfidence |
| Acquisition function | Optimistic yet safe exploration of uncertain regions |
| Stability diagnostics| Identify reliable models and complex/uncertain regions |

This strategy prioritises **robustness and safety** over aggressive optimisation in a **weakly-structured, noisy function**.


---

# f8 — ARD-GP + SVM-RFE Feature Consensus with KNN-Refined Coordinate Descent

## Strategy Name
ARD-GP + SVM-RFE Feature Consensus with KNN-Refined Coordinate Descent

## Objective of This Submission

Exploit strong evidence that X1 and X3 dominate f8 by:

- Validating feature importance using three independent methods:
  - ARD-GP length-scales  
  - SVM-RFE rankings  
  - Mutual Information (MI) from Week-3 EDA  
- Performing coordinate descent only along X1 and X3
- Using KNN for local refinement to avoid GP oversmoothing
- Applying CV-based acceptance to prevent drift or overfitting

This is a sparse-structure exploitation strategy designed for functions where only a subset of variables matter.

## ML Method & Rationale

### Stage 1 — Feature Selection Consensus (ARD-GP + SVM-RFE + MI)

**ARD-GP**

- Fit a Matérn GP with ARD (one length-scale per feature)
- Small length-scale indicates higher importance
- Expectation: X1 and X3 have the smallest length-scales

**SVM-RFE**

- Train ε-SVR with RBF kernel
- Apply Recursive Feature Elimination (RFE)
- Expectation: X1 and X3 ranked top-2

**MI Reference**

- Week-3 EDA Mutual Information already indicated X1/X3 dominance
- Acts as a third independent validation

**Consensus Check**

- If ARD-GP, SVM-RFE, and MI all agree on X1/X3 → proceed
- If not → hypothesis fails

### Stage 2 — Coordinate Descent with KNN Refinement

Optimise only X1 and X3, holding all other coordinates fixed at the current best.

For each coordinate (X1 then X3):

1. Fit ARD-GP on full data for global smooth structure
2. Fit KNN (k=5) in a local neighbourhood around the current best (±0.2 window)
3. Build a local ensemble:

   μ_ens = 0.7 μ_GP + 0.3 μ_KNN

   - GP provides smooth global structure  
   - KNN provides sharp local detail  

4. Optimise Expected Improvement (EI) along the 1D coordinate
5. Update the coordinate
6. Perform 5-fold CV on ensemble predictions to ensure stability

This forms a hybrid local–global coordinate descent.

### Stage 3 — CV-Based Acceptance

After updating X1 and X3:

- Compute CV mean and CV std for the final candidate
- Compute CV Lower Confidence Bound:

  LCB_CV = μ_CV − σ_CV

**Acceptance Rule**

- Accept candidate only if LCB_CV > y_best

This prevents coordinate descent from drifting into misleading or noisy regions.

## Key Assumptions

- X1 and X3 are truly dominant (MI, correlations, partial r)
- SVM-RFE agrees with ARD-GP
- KNN is reliable locally within a small neighbourhood
- f8 is separable enough for coordinate descent
- CV intervals reflect true uncertainty

## Hypothesis

### If the Hypothesis Holds

- SVM-RFE ranks X1/X3 as top-2
- ARD-GP length-scales for X1/X3 are smallest
- Coordinate descent improves the best value
- CV-LCB > current best → stable improvement
- Calibration improves (Gap < 0.35)

### If the Hypothesis Fails

- SVM-RFE disagrees with ARD/MI → feature dominance unclear
- KNN refinement fails → structure may be non-separable
- CV-LCB ≤ current best → candidate rejected
- f8 may require full-dimensional Bayesian Optimisation

## Summary Logic

| Component | Role |
|--------|------|
| ARD-GP | Global smooth structure and feature importance |
| SVM-RFE | Independent feature ranking |
| MI (EDA) | Historical evidence of dominance |
| KNN | Local refinement near current best |
| Coordinate Descent | Sparse optimisation on key variables |
| CV-LCB | Stability and overfitting safeguard |

This strategy targets high-confidence sparse exploitation where only a few variables meaningfully drive the response surface.
