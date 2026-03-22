# Week 3 – Feature-Guided, Uncertainty-Aware Exploration Submission

## Objective of This Submission
Test whether **feature-guided, uncertainty-aware sampling** identifies regions of the search space that improve best-observed values for each function (f1–f8), thereby validating or falsifying assumptions about **feature importance, surrogate smoothness, and uncertainty estimates**.  
**Single objective only:** belief revision about the function landscape for each function using controlled experimentation.

---

## Key Assumptions (Max 3)
1. **Surrogate Smoothness:** The surrogate can generalize locally around observed points; predicted trends approximate true function behaviour.  
2. **Uncertainty Calibration:** Predicted variance from the surrogate accurately reflects regions of high risk or high potential improvement.  
3. **Feature Relevance:** Inputs identified as strong predictors via partial correlation, Spearman, or mutual information reliably indicate directions for effective exploitation.  

---

## Hypothesis (Directional)
If assumptions hold, then a **heteroskedastic Gaussian Process (or GP+RF ensemble for weak/complex functions) with UCB acquisition**, combined with **feature-informed sampling**, will outperform the Week 2 baseline in maximising best value per function for f1–f8, while providing structural insight into feature effects.

- **Baseline (B):** Week 2 best-observed values  
- **Metric (X):** Best value per function  
- **Functions (Y):** f1–f8  

---

## Methods & Design Choices

| Component | Strategy |
|----------|---------|
| Surrogate Model | Heteroskedastic GP; GP+RF ensemble for weak/complex functions (f3, f7) |
| Acquisition Function | UCB with function-specific β (high for uncertain functions, moderate for structured ones, low for near-flat/near-optimal) |
| Initialisation Strategy | Seed with Week 2 best points; add exploratory points in high-variance regions; feature-guided sampling |
| Budget Allocation | 1 submission per function + extra exploratory points for high-residual functions |
| Constraints | Respect bounds; avoid overfitting; limited duplicate sampling |

---

# Function Strategies (f1–f8)

---

## f1 — Flat-Surface Verification Strategy

### Observations
- Function is nearly constant around 0  
- GP mean ≈ 0, variance extremely low  
- No structure detected  

### Objective
Confirm flatness and ensure no hidden structure exists.

### Assumptions
- Function is constant or extremely smooth  
- No meaningful feature influence  

### Hypothesis
- If true → values remain near 0  
- If false → deviation reveals hidden structure  

### Method
Low-β GP-UCB with random interior sampling  
→ Avoids wasting queries on meaningless exploitation  

---

## f2 — Local Basin Exploitation Strategy

### Observations
- Slight upward trend in higher X1 regions  
- GP suggests shallow basin  

### Objective
Exploit and validate the basin.

### Assumptions
- X1 is relevant  
- Surface is smooth and unimodal  

### Hypothesis
- If true → consistent improvement near high X1  
- If false → basin collapses  

### Method
Moderate-β GP-UCB with X1 emphasis  

---

## f3 — Uncertainty-Driven Exploration Strategy

### Observations
- High uncertainty  
- Unstable GP hyperparameters  

### Objective
Reduce uncertainty and stabilise model.

### Assumptions
- Structure exists but is under-sampled  

### Hypothesis
- If true → uncertainty decreases  
- If false → noise dominates  

### Method
High-β GP-UCB with space-filling sampling  

---

## f4 — Local Refinement Strategy

### Observations
- Moderately stable region identified  

### Objective
Refine and confirm local optimum.

### Assumptions
- Smooth surface  
- Single dominant region  

### Hypothesis
- If true → improvement or confirmation  
- If false → performance drops  

### Method
Low-to-moderate β GP-UCB with edge penalty  

---

## f5 — Log-Stabilised GP with Expected Improvement (4D)

### Observations
- Smooth structure  
- Large output magnitude  
- Strong gradients  

### Objective
Test EI stability under 4D extension.

### Assumptions
- Smooth and well-behaved function  
- Log transform stabilises variance  

### Hypothesis
- If true → stable convergence  
- If false → instability or poor EI performance  

### Method
- GP (Matérn 2.5)  
- Log transform  
- Expected Improvement  
- Multi-start optimisation  

---

## f6 — Moderate-Exploration UCB with Robust GP

### Observations
- Conflicting monotonic relationships  
- Large residuals  

### Objective
Test uncertainty calibration under structural conflict.

### Assumptions
- Mixed monotonic structure  
- Smooth but complex interactions  

### Hypothesis
- If true → UCB explores meaningful ambiguous regions  
- If false → exploration is misdirected  

### Method
Moderate-β GP-UCB with heteroskedastic robustness  

---

## f7 — GP + RF Ensemble-UCB (Weak Structure)

### Observations
- Weak/non-smooth structure  
- GP alone insufficient  

### Objective
Test if model disagreement identifies improvements.

### Assumptions
- GP captures global trend  
- RF captures local irregularities  

### Hypothesis
- If true → disagreement correlates with improvement  
- If false → behaves like noise  

### Method
Ensemble-UCB combining:
- GP mean + uncertainty  
- RF predictions  
- GP–RF disagreement  

---

## f8 — Feature-Guided GP-UCB with Edge Penalty (8D)

### Observations
- High dimensionality  
- Boundary bias observed  

### Objective
Improve stability and reduce boundary artefacts.

### Assumptions
- Smooth function  
- Interior optima more likely  

### Hypothesis
- If true → interior sampling improves stability  
- If false → edge optima suppressed  

### Method
- GP (Matérn 2.5)  
- Moderate β UCB  
- Feature-weighted uncertainty  
- Soft edge penalty  
- Multi-start optimisation  

---

## Final Insight
Week 3 is not about maximisation alone — it is a **controlled experimental phase** to validate:

- Whether uncertainty is meaningful  
- Whether feature signals are actionable  
- Whether surrogate assumptions hold  

Every function strategy is designed to **test a specific structural hypothesis**, not just optimise blindly.
