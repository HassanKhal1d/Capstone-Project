# Week 5 – Warping + GP Spine + Selective Residual Learning

This document outlines the Week 5 submission plan, shifting from surrogate comparison to systematic calibration correction and acquisition reliability.  

Emphasis is on:

- Distribution warping (HEBO-style philosophy)
- Heteroscedastic Gaussian Processes as the uncertainty spine
- Selective residual neural correction (only where justified)
- Removal of overly conservative CV gating where it blocked exploration
- Controlled trust-region exploration (TuRBO where appropriate)

### Meta-Strategy
**Warping-First + GP Uncertainty Spine + Controlled Exploration**

- Primary surrogate: (Heteroscedastic) ARD-GP
- Warping before modelling: Yeo–Johnson / Box–Cox where skewness detected
- Neural residuals: Only when systematic mean bias is demonstrated
- Acquisition: Thompson Sampling or EI (function-dependent)
- No unnecessary ensembles that dilute calibrated uncertainty

### Learning Objectives

- Does input warping reduce calibration error across skewed functions?
- When does heteroscedastic GP outperform variance inflation?
- Where do small residual MLPs improve mean accuracy without harming uncertainty?
- Which failures were acquisition-driven vs surrogate-driven?

## Function-Specific Strategies

### f1 — Verification-Only Control Strategy
**Objective**: Confirm that f1 remains structurally flat and avoid wasting modelling capacity. Acts as a calibration control benchmark.

**ML Method & Rationale**

- GP (RBF * ConstantKernel)  
  - Predict ~0 everywhere  
  - exploitation coefficient (β = 0.3)
  - Sobol Sequence for candidate inputs  
  - Single verification query  
- No neural network, no ensemble  

**Key Assumptions**

- Week-3 metrics showed near-perfect calibration  
- No structural signal exists  
- Additional modelling complexity adds no value  

**Hypothesis**

- If it holds: Prediction ≈ 0, calibration stable, no structural emergence  
- If it breaks: Unexpected non-zero structure, requiring reclassification of f1  

**Summary Logic**: Minimal allocation. Preserve budget for exploitable functions.

---

### f2 — Yeo–Johnson Warped ARD-GP with Aggressive UCB
**Objective**: Correct upper-tail underexploration along X1 using a single calibrated ARD-GP with increased exploration.

**ML Method & Rationale**

1. Yeo–Johnson transform on X1 to correct skew and improve stationarity  
2. ARD Matérn 2.5 GP to learn anisotropic structure and preserve calibrated uncertainty  
3. Aggressive UCB (β = 2.0) to push into upper-confidence regions  

**Key Assumptions**

- X1 monotonicity is genuine  
- Previous failure was acquisition miscalibration  
- Warping improves length-scale estimation  

**Hypothesis**

- If it holds: Upper-tail candidate discovered, calibration maintained, improved best observed  
- If it breaks: Length-scale collapse, no meaningful upper-region improvement  

**Summary Logic**: Remove ensemble dilution and push uncertainty where structure is known.

---

### f3 — Box–Cox + KNN-Dominant Portfolio with Residual GP Correction
**Objective**: Exploit local structure while controlling noise-driven instability.

**ML Method & Rationale**

1. Box–Cox target transform to stabilise moderate skew  
2. ARD-GP to capture anisotropic structure  
3. KNN (k=5, distance-weighted) for strong local behaviour  
4. Tiny residual MLP (4 units) trained on GP residuals, clipped for stability  

**Ensemble Weighting**: 50% KNN, 35% GP, 15% GP+MLP  

**Key Assumptions**

- Moderate structure + noise  
- KNN superiority reflects real locality  
- MLP corrects mean only, not uncertainty  

**Hypothesis**

- If it holds: Improved local predictions, reduced bias, stable calibration  
- If it breaks: MLP overfits, GP/KNN disagreement widens  

**Summary Logic**: Local-first optimisation with bounded residual learning.

---

## f4 — SVM-Guided Dual-GP Neural Ensemble with TuRBO and Thompson Sampling

### Objective

Design a directionally guided, uncertainty-aware Bayesian optimisation framework capable of escaping deep negative basins and rediscovering the rare positive spike of **f4** under extreme heteroskedastic noise.

---

### ML Method & Rationale

- **Heteroskedastic ARD Gaussian Process (Primary Surrogate)**  
  Models input-dependent noise and provides calibrated posterior uncertainty for acquisition.

- **UCB ARD Gaussian Process (Directional Expert)**  
  Preserves global structural signal and reinforces prior knowledge of the Week-3 optimum region.

- **Residual MLP Mean Corrector**  
  Captures sharp local nonlinear curvature and stabilises ensemble mean predictions.

- **SVM Boundary Classifier**  
  Filters statistically hopeless regions and prevents wasted exploration in deep negative zones.

- **TuRBO Trust Region (Adaptive Radius Control)**  
  Enforces stable local optimisation with controlled expansion.

- **Directional Bias Mechanism**  
  Prioritises candidates aligned with known structural drivers of f4.

- **Variance Inflation Factor (√3)**  
  Applied to GP posterior standard deviation to counter severe non-stationary noise.

- **Acquisition Blend**  
  - 80% Expected Improvement  
  - 20% Thompson Sampling  
  Balances stability with aggressive basin-jumping behaviour.

---

### Ensemble Structure

- **Primary Mean**: Weighted dual-GP ensemble with residual MLP correction  
- **Uncertainty Backbone**: Heteroskedastic GP  
- **Directional Pull**: UCB GP  
- **Local Bias Correction**: Residual MLP  
- **Candidate Filtering**: SVM classifier before acquisition evaluation  

---

### Acquisition Strategy

- **Expected Improvement (EI)**  
  Ensures stable incremental improvement.

- **Thompson Sampling (TS)**  
  Injects stochastic exploration to escape negative basins.

- **TuRBO Constraint**  
  Limits proposals to adaptive trust regions for controlled exploration.

---

### Key Assumptions

- The positive region of f4 is narrow but structurally coherent.  
- Noise is strongly heteroskedastic and non-stationary.  
- The SVM boundary generalises meaningfully beyond observed samples.  
- Local smooth structure exists around the optimum basin.  
- The ensemble mean is more reliable than any individual surrogate.

---

### Hypothesis

#### If the Hypothesis Holds

- The SVM excludes deep negative regions.  
- The UCB GP pulls search toward structurally promising directions.  
- The heteroskedastic GP maintains calibrated uncertainty.  
- Variance inflation prevents overconfidence.  
- Thompson Sampling enables basin jumps.  
- TuRBO expands toward the positive spike.  

Expected observed behaviour:

- Stabilised Z-scores  
- Reduced calibration gap  
- Progressive improvement toward values near **0.07 or higher**

#### If the Hypothesis Breaks

- The SVM may exclude the optimum region.  
- The UCB GP may misdirect exploration.  
- TuRBO may contract prematurely.  
- EI may collapse under skew.  
- Thompson Sampling may overspread exploration.  

Observed behaviour would include:

- Oscillation across negative basins  
- Failure to surpass prior optimum  
- Persistent calibration instability  

---

### Summary Logic

Global directional structure plus local heteroskedastic modelling plus classifier-guided filtering replaces naive variance inflation.

This method is engineered to be calibrated, structurally informed, and exploration-capable within a pathological, skewed landscape.

---

### f5 — Log-Warped Heteroscedastic GP with Controlled EI
**Objective**: Recover from over-conservative CV gating and exploit high-magnitude regions.

**ML Method & Rationale**

1. Log-transform targets (y′ = log(y+1)) to stabilise magnitude  
2. Heteroscedastic GP to model noise directly  
3. EI with soft inflation (4–5× if needed) for controlled uncertainty  
4. Small diagnostic MLP (4 units) for mean check only  
5. No CV-LCB gating  

**Key Assumptions**

- Log-GP remains valid  
- Failure was acquisition over-constraint  
- Upper region still exploitable  

**Hypothesis**

- If it holds: Significant recovery, stable calibration, improvement beyond prior peak  
- If it breaks: Extreme-region modelling unstable, noise overwhelms signal  

**Summary Logic**: Rollback to validated log-GP core. Remove overconservative candidate filtering.

---

### f6 — Direct 2D Warped ARD-GP with Thompson Sampling
**Objective**: Model only X4/X5 directly, abandoning REMBO embedding.

**ML Method & Rationale**

1. Yeo–Johnson transform on X4/X5 to improve stationarity  
2. ARD-GP in 2D to avoid curse of dimensionality  
3. Thompson Sampling for moderate exploration  
4. No ensemble, no NN  

**Key Assumptions**

- Effective dimensionality = 2  
- Direct modelling superior to random embedding  

**Hypothesis**

- If it holds: Improved CV, calibration stabilises, better candidate found  
- If it breaks: 2D insufficient, hidden interaction exists  

**Summary Logic**: Simplify. Model true subspace directly.

---

### f7 — Heteroscedastic GP with Pareto (EI + UCB) Exploration
**Objective**: Prevent catastrophic overconfidence while maintaining exploratory pressure.

**ML Method & Rationale**

1. Heteroscedastic ARD-GP to replace manual inflation  
2. Transitional 3× inflation safeguard  
3. Residual MLP (6 units) for mean correction only  
4. Pareto selection across EI(x) and UCB(x)  

**Key Assumptions**

- Weak but real structure exists  
- Noise is dominant failure source  
- Pareto prevents single-metric collapse  

**Hypothesis**

- If it holds: |Z| reduced, no catastrophic regression, incremental improvement  
- If it breaks: Noise overwhelms modelling, surrogate family insufficient  

**Summary Logic**: Balance optimism and caution in high-uncertainty regime.

---

### f8 — TuRBO + Thompson Sampling with Diversity Enforcement
**Objective**: Fix acquisition stagnation while preserving strong calibration.

**ML Method & Rationale**

1. ARD-GP (well-calibrated)  
2. TuRBO (moderate radius) to avoid global plateau exploration  
3. Thompson Sampling replaces CV-LCB gating  
4. Diversity constraint: enforce minimum distance δ_min = 0.15  
5. Diagnostic MLP (8 units) for mean check only  

**Key Assumptions**

- Surrogate correct  
- Failure was acquisition conservatism  

**Hypothesis**

- If it holds: Break stagnation, moderate improvement, calibration preserved  
- If it breaks: Plateau intrinsic to surface, requires dimensional re-evaluation  

**Summary Logic**: Model remains. Acquisition corrected.

---

### Final Strategic Summary

| Theme | Week 4 | Week 5 Shift |
|-------|--------|--------------|
| Ensemble Complexity | High | Reduced |
| CV Gating | Aggressive | Selective |
| Variance Inflation | Manual | Heteroscedastic |
| Warping | Limited | Systematic |
| Neural Nets | Broad | Residual-only |
| Acquisition | Conservative | Calibrated + exploratory |
