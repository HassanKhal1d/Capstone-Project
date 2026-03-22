# Week 3 – Feature-Guided, Uncertainty-Aware Exploration Submission

## Objective of This Submission

Test whether **feature-guided, uncertainty-aware sampling** identifies regions of the search space that improve best-observed values for each function (f1–f8), thereby validating or falsifying assumptions about:

- Feature importance  
- Surrogate smoothness  
- Uncertainty estimates  

**Single objective:** belief revision about the function landscape for each function using controlled experimentation.

---

## Key Assumptions (Max 3)

1. **Surrogate Smoothness**  
   The surrogate can generalise locally around observed points; predicted trends approximate true function behaviour.

2. **Uncertainty Calibration**  
   Predicted variance from the surrogate accurately reflects regions of high risk or high potential improvement.

3. **Feature Relevance**  
   Inputs identified as strong predictors via partial correlation, Spearman, or mutual information reliably indicate directions for effective exploitation.

---

## Hypothesis (Directional)

If assumptions hold, then a **heteroskedastic Gaussian Process (or GP + RF ensemble for weak/complex functions) with UCB acquisition**, combined with **feature-informed sampling**, will outperform the Week 2 baseline in maximising best value per function (f1–f8), while providing structural insight into feature effects.

- **Baseline (B):** Week 2 best-observed values  
- **Metric (X):** Best value per function  
- **Functions (Y):** f1–f8  

---

## Methods & Design Choices

| Component | Strategy |
|----------|---------|
| **Surrogate Model** | Heteroskedastic GP; GP + RF ensemble for weak-signal functions (f3, f7) |
| **Acquisition Function** | UCB with function-specific β |
| **Initialisation Strategy** | Seed with Week 2 best points + feature-guided exploratory points |
| **Budget Allocation** | 1 submission per function (8 total) + 2 extra exploratory points (f2/f4/f7) |
| **Constraints** | Respect bounds; avoid overfitting; allow duplication only for local exploitation |

---

# Function Strategies (f1–f8)

---

## f1 — Flat-Surface Verification Strategy

### Observations
- Near-zero variance across all observations  
- GP predicts μ ≈ 0 and σ ≈ 0 globally  
- No detectable structure  

### Objective
Confirm flatness and rule out hidden interior peaks.

### Assumptions
- Function is effectively constant  
- No dimension is influential  

### Hypothesis
- **If true:** new samples remain ≈ 0  
- **If false:** rare deviations reveal hidden structure  

### Method
Low-β GP-UCB with random interior sampling.

---

## f2 — Local Basin Exploitation Strategy

### Observations
- Mild upward trend in high-X1 regions  
- GP suggests shallow basin  

### Objective
Exploit and validate basin structure.

### Assumptions
- X1 is relevant  
- Surface is smooth and locally unimodal  

### Hypothesis
- **If true:** consistent improvement near high-X1  
- **If false:** basin collapses  

### Method
Moderate-β GP-UCB with X1 emphasis.

---

## f3 — Uncertainty-Driven Exploration Strategy

### Observations
- High uncertainty  
- Unstable GP hyperparameters  

### Objective
Reduce uncertainty and stabilise surrogate.

### Assumptions
- Structure exists but is under-sampled  

### Hypothesis
- **If true:** σ decreases and structure emerges  
- **If false:** σ remains high  

### Method
High-β GP-UCB with space-filling sampling.

---

## f4 — Local Refinement Strategy

### Observations
- Moderately strong local region  
- Stable GP predictions  

### Objective
Refine local optimum.

### Assumptions
- Smooth function with dominant region  

### Hypothesis
- **If true:** improvement near region  
- **If false:** GP overfit exposed  

### Method
Low–moderate β GP-UCB with soft edge penalty.

---

## f5 — Log-Stabilised GP with Expected Improvement (4D)

### Strategy Name
Log-Stabilised GP + EI with Multi-Start Optimisation

### Observations
- Strong global structure  
- Large output magnitudes  
- Smooth behaviour  

### Objective
Test EI stability under 4D extension.

### Assumptions
- Smooth, log-stabilised outputs  
- EI can exploit gradients  

### Hypothesis
- **If true:** stable convergence  
- **If false:** EI failure or instability  

### Method
- Matérn 2.5 GP  
- Log transform  
- EI acquisition  
- Multi-start optimisation  

---

## f6 — Moderate-Exploration UCB with Robust GP (5D)

### Observations
- Conflicting monotonic relationships  
- Large GP residuals  
- Ambiguous regions  

### Objective
Test uncertainty calibration under structural conflict.

### Assumptions
- Mixed monotonic structure  
- GP remains valid  

### Hypothesis
- **If true:** UCB explores meaningful ambiguity  
- **If false:** exploration is misdirected  

### Method
Moderate-β UCB with heteroskedastic-robust GP.

---

## f7 — GP + RF Ensemble-UCB (7D)

### Strategy Name
Ensemble-UCB with GP–RF Disagreement

### Observations
- Weak structure  
- Potential non-smoothness  
- GP alone insufficient  

### Objective
Test whether model disagreement identifies missed maxima.

### Assumptions
- RF captures local effects  
- GP captures global structure  

### Hypothesis
- **If true:** disagreement → improvement  
- **If false:** disagreement = noise  

### Method
- GP + RF ensemble  
- Acquisition = mean + uncertainty + disagreement  

---

## f8 — Feature-Guided GP-UCB with Soft Edge Penalty (8D)

### Strategy Name
Feature-Guided GP-UCB with Interior Bias

### Observations
- High dimensional (8D)  
- Boundary bias observed  
- Some features more influential  

### Objective
Improve stability and reduce boundary attraction.

### Assumptions
- Smooth structure  
- Interior contains meaningful optima  

### Hypothesis
- **If true:** stable interior convergence  
- **If false:** boundary collapse persists  

### Method
- GP (Matérn 2.5)  
- Moderate β UCB  
- Feature-weighted uncertainty  
- Soft edge penalty  
- Multi-start optimisation  

---
