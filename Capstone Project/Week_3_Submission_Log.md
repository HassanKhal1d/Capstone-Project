# Function Strategies (f1–f8)

---

# f1 — “Flat-Surface Verification Strategy”

## Observations

- EDA and previous queries show **f1 is nearly flat around 0** with extremely low variance.
- GP predictions show tiny σ and μ ≈ 0 everywhere.
- No meaningful structure detected.

## Objective

- Confirm flatness and avoid wasting queries on exploitation.
- Ensure no hidden peaks exist in interior regions.

## Assumptions

- f1 is essentially constant or extremely smooth.
- No single dimension meaningfully influences the output.

## Hypothesis

- **If true:** new samples will also return values near 0, GP variance will shrink further.
- **If false:** a rare interior point will show deviation, revealing hidden structure.

## ML Method + Rationale

**Low-β GP-UCB with randomised interior sampling.**

- Rationale: GP is stable on flat functions; low β avoids chasing meaningless uncertainty.

---

# f2 — “Local Basin Exploitation Strategy”

## Observations

- Prior rounds show a mild upward trend in regions with higher X1.
- GP mean shows a shallow basin; uncertainty is moderate but structured.

## Objective

- Exploit the region where the GP predicts slightly higher values.
- Validate whether the basin is real or an artefact of sparse sampling.

## Assumptions

- X1 is a relevant feature.
- The surface is smooth and unimodal in that region.

## Hypothesis

- **If true:** sampling near high-X1 interior points yields consistently higher values.
- **If false:** new samples flatten the basin, revealing GP overfitting.

## ML Method + Rationale

**GP-UCB with moderate β, slight feature emphasis on X1.**

- Rationale: balances exploitation with enough uncertainty sampling to confirm the basin.

---

# f3 — “Uncertainty-Driven Exploration Strategy”

## Observations

- High posterior uncertainty across multiple dimensions.
- GP hyperparameters unstable → suggests under-sampling or multimodality.

## Objective

- Reduce uncertainty by sampling diverse interior points.
- Stabilise GP length-scales.

## Assumptions

- The function has structure but is poorly mapped.
- Exploration is more valuable than exploitation.

## Hypothesis

- **If true:** new samples reduce σ significantly and reveal clearer trends.
- **If false:** σ remains high, indicating noise or extreme flatness.

## ML Method + Rationale

**High-β GP-UCB with space-filling candidate generation.**

- Rationale: exploration is essential when the model is unstable.

---

# f4 — “Local Refinement Strategy”

## Observations

- Prior queries show a region with moderately good performance.
- GP mean is more stable than f3; uncertainty is lower.

## Objective

- Refine the local region to determine whether it contains a true optimum.
- Avoid drifting to edges.

## Assumptions

- The function is moderately smooth with a single dominant region of interest.

## Hypothesis

- **If true:** new samples near the region improve or confirm the local optimum.
- **If false:** performance drops, indicating the GP overestimated the region.

## ML Method + Rationale

**Low-to-moderate β GP-UCB with soft edge penalty.**

- Rationale: controlled exploitation while avoiding boundary artefacts.

---

# f5 — Log-Stabilised GP with Expected Improvement (4D Extension)

## Strategy Name  
Log-Stabilised Homoskedastic Gaussian Process with Expected Improvement and Multi-Start Optimisation (4D)

---

## Observations  

- f5 converged quickly in lower dimensions → indicates a strong global structure.  
- Outputs are large in magnitude → risk of numerical instability in GP fitting.  
- The function exhibits smooth behaviour.  
- Improvement gradients are present and exploitable.  
- Moving from lower dimensions to 4D increases acquisition multi-modality.  

---

## Objective of This Submission  

To test whether Expected Improvement (EI) remains stable and effective when f5 is extended to 4D, and whether the log-transformed GP surrogate continues to model the function reliably under increased dimensionality.

**One objective only.**

---

## Assumptions About the Function  

- The function is smooth and globally structured.  
- Output scale is large but stabilisable via log transformation.  
- Noise is approximately homoskedastic after transformation.  
- Improvement gradients exist and EI can exploit them.  
- The function remains well-behaved when extended to 4D.  

---

## Hypothesis  

A log-transformed GP surrogate combined with Expected Improvement will remain numerically stable and will continue to identify meaningful improvement directions in 4D, converging efficiently despite increased dimensionality.

- **If the hypothesis holds:**  
  - Stable GP hyperparameters.  
  - Consistent EI-driven improvement.  
  - Smooth convergence toward a strong maximum.  
  - No numerical instability from large outputs.  

- **If the hypothesis fails:**  
  - GP instability or poorly conditioned covariance matrices.  
  - EI becoming overly exploitative or ineffective.  
  - Slower convergence due to dimensionality-induced multi-modality.  
  - Evidence that exploration-heavy acquisition (e.g., UCB) would have been preferable.  

---

## ML Method and Rationale  

### Gaussian Process (Matérn 2.5 Kernel)  
Captures smooth-but-not-perfectly-smooth behaviour while allowing moderate curvature flexibility.

### Log Transform  
Compresses large output magnitudes, reduces effective heteroskedasticity, and improves numerical conditioning of the GP.

### Expected Improvement (EI)  
Well-suited for functions with a strong global trend; efficiently balances exploration and exploitation when improvement gradients exist.

### Multi-Start Optimisation  
EI becomes multi-modal in 4D. Multiple restarts reduce the risk of acquisition optimisation getting trapped in local optima.


---

# f6 — Moderate-Exploration UCB with Heteroskedastic-Robust GP (5D)

## Strategy Name  
Moderate-Exploration UCB with a Heteroskedastic-Robust Gaussian Process

---

## Observations  

- f6 exhibits opposing monotonic directions (e.g., \( X_4 \uparrow \) increases f6, while \( X_5 \uparrow \) decreases f6).  
- The GP previously misestimated the surface (large residual ≈ −298%).  
- The surface appears smooth but structurally conflicting.  
- Ambiguous regions exist where monotonic trends break down.  
- Expected Improvement (EI) would likely collapse prematurely toward misleading local structure.  

---

## Objective of This Submission  

Test whether GP-estimated uncertainty is correctly calibrated in 5D when monotonic trends conflict, by using moderate-β UCB to explore under-sampled and structurally ambiguous regions.

**One objective only.**

This is a falsifiable experiment:

- If uncertainty is meaningful → UCB explores the correct ambiguous regions.  
- If uncertainty is miscalibrated → UCB explores irrelevant regions.

---

## Assumptions About the Function  

- f6 has mixed monotonic structure (some inputs positively associated, others negatively associated).  
- The function remains smooth enough for a Matérn GP.  
- Noise behaves approximately heteroskedastically; added jitter improves robustness.  
- Sample size is small, so uncertainty estimation materially affects decisions.  
- EI would collapse early due to misleading local monotonic structure.  

---

## Hypothesis  

A moderately exploratory UCB-GP will correctly identify regions where monotonic assumptions break down, improving uncertainty calibration and guiding sampling toward the true maxima.

### If the Hypothesis Holds  

- UCB explores the correct ambiguous regions.  
- Predictive residuals shrink.  
- The GP reconciles opposing monotonic trends.  
- Posterior uncertainty becomes better aligned with true model error.  

### If the Hypothesis Fails  

- UCB explores structurally irrelevant areas.  
- Residuals remain large.  
- The GP fails to reconcile monotonic conflicts.  
- Uncertainty estimates remain poorly calibrated.  

---

## ML Method and Rationale  

### Matérn Gaussian Process (Heteroskedastic-Robust)  
Models smooth but interaction-heavy behaviour.  
Additional jitter improves numerical stability under heteroskedastic noise patterns.

### Moderate-β Upper Confidence Bound (UCB)  
Encourages structured exploration without excessive wandering.  
Specifically targets regions of high predictive uncertainty where monotonic assumptions may break down.

---

# f7 — GP + RF Ensemble-UCB (Weakly Structured 7D)

## Strategy Name  
Ensemble-UCB with Gaussian Process + Random Forest Disagreement

---

## Observations  

- The problem appears weakly structured and potentially non-smooth.  
- A single smooth surrogate (GP) risks missing local irregularities.  
- Residual structure suggests possible model misspecification under a smooth prior.  
- Small-sample regime → uncertainty estimation materially affects decisions.  
- 7D increases structural complexity and interaction effects.  

---

## Objective of This Submission  

Test whether GP–RF disagreement reliably identifies regions containing missed global maxima in weakly structured 7D functions.

**One objective only.**

---

## Assumptions About the Function  

- The function is weakly structured with latent local interactions.  
- A smooth GP captures global trends but may miss local non-smooth behaviour.  
- A Random Forest can capture local, non-smooth interactions.  
- Sample size is limited, so both uncertainty and model disagreement influence acquisition.  
- RF ensemble variance is a useful proxy for local model instability or misspecification.  
- Simple averaging of GP and RF predictions is a reasonable first-order ensemble strategy.  

---

## Hypothesis  

Regions where the GP and RF disagree (high disagreement) are more likely to contain missed improvements. Targeting these regions using ensemble-UCB will outperform GP-only Bayesian Optimisation in Week-3.

### If the Hypothesis Holds  

- High GP–RF disagreement correlates with realised improvements.  
- Ensemble-UCB samples structurally informative regions.  
- New maxima exceed GP-only performance.  
- Disagreement reduces as the ensemble refines the surface.  

### If the Hypothesis Fails  

- Disagreement does not correlate with realised improvement.  
- Ensemble-UCB explores regions that do not yield superior maxima.  
- GP-only BO performs equally well or better.  
- Disagreement behaves like noise rather than a structural signal.  

---

## ML Method and Rationale  

### Gaussian Process (Matérn Kernel, Normalised Targets)  
Captures smooth global structure and provides calibrated predictive uncertainty.

### Random Forest Ensemble (Raw Targets)  
Captures local, non-smooth interactions and provides an independent mean and variance signal.

### Acquisition: Ensemble-UCB  
Combines:
- Ensemble mean  
- GP predictive uncertainty  
- GP–RF disagreement  

### Rationale  

- GP uncertainty drives principled exploration.  
- RF disagreement highlights potential smooth-surrogate misspecification.  
- Combining both focuses sampling on regions that are uncertain *and* structurally ambiguous.  

This directly tests whether model disagreement is a reliable signal for missed maxima in a small-sample, weakly structured 7D regime.


---

# f8 — Feature-Guided GP-UCB with Soft Edge Penalty (8D)

## Strategy Name  
Feature-Guided Gaussian Process UCB with Soft Edge Penalty

---

## Observations  

- The problem operates in high dimension (8D), increasing acquisition multi-modality and optimisation instability.  
- Previous runs frequently returned boundary solutions (0 or 1), suggesting edge-seeking behaviour from the acquisition.  
- The objective appears smooth enough for a GP but structurally complex across dimensions.  
- Week-3 priority is controlled exploitation rather than aggressive global exploration.  
- Some features are believed to be more influential (e.g., X1, X3), motivating feature-guided uncertainty weighting.  

---

## Objective of This Submission  

Test whether a feature-guided UCB acquisition with a soft interior bias improves sampling stability in 8D by:

- Encouraging exploitation of model-informed regions,  
- Reducing unjustified boundary solutions, and  
- Maintaining calibrated uncertainty-driven search.

**One objective only.**

---

## Assumptions About the Function  

- The function is sufficiently smooth for a Matérn 2.5 Gaussian Process.  
- The signal-to-noise ratio allows meaningful uncertainty estimation.  
- Certain input dimensions contribute more strongly to variation.  
- Boundary optima are not systematically dominant (i.e., edges are not inherently optimal).  
- Moderate exploration (β ≈ 0.6) is sufficient in Week-3.  

---

## Hypothesis  

A feature-guided GP-UCB with a soft edge penalty will:

- Focus sampling in structurally informative interior regions,  
- Reduce artificial boundary attraction, and  
- Improve stability of convergence in 8D.

### If the Hypothesis Holds  

- Proposed points concentrate in meaningful interior regions.  
- UCB remains stable without oscillatory boundary jumps.  
- Residual error decreases steadily.  
- Feature-weighting enhances sensitivity along influential dimensions.  

### If the Hypothesis Fails  

- Acquisition still collapses to boundaries despite penalty.  
- Interior bias suppresses valid edge optima.  
- Feature-weighting does not meaningfully influence sampling.  
- Convergence stagnates or becomes erratic.  

---

## ML Method and Rationale  

### Gaussian Process (Matérn 2.5 Kernel)  
Models smooth but moderately flexible structure.  
Provides predictive mean and uncertainty necessary for UCB.

### Upper Confidence Bound (Moderate β)  
UCB = μ + βσ  
Balances exploitation and exploration, aligned with Week-3 objectives.

### Feature-Guided Uncertainty Scaling  
Per-dimension feature weights are aggregated into a scalar multiplier applied to the uncertainty term.  
Rationale: Emphasise uncertainty in dimensions believed to matter most without introducing directional instability.

### Soft Edge Penalty  
A smooth exponential decay penalty applied near domain boundaries.  
Rationale: Discourages boundary solutions unless strongly justified by GP predictions, preventing artificial edge bias.

### Multi-Start Optimisation  
Multiple L-BFGS-B restarts reduce risk of local acquisition traps in 8D.

---

This design explicitly tests whether structured exploitation with controlled interior bias produces more stable and reliable optimisation behaviour in high-dimensional settings.

