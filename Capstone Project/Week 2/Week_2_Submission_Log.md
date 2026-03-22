# Week 2 – Exploratory Bayesian Optimization Submission

## Objective of This Submission
To establish a **baseline exploratory strategy using a uniform Upper Confidence Bound (UCB) Bayesian Optimisation** across all black-box functions, focusing on learning the function landscapes efficiently.

---

## Key Assumptions (Max 3)
1. Gaussian Process (GP) surrogates provide a reasonable approximation of each function with limited observations.  
2. High posterior uncertainty regions contain previously unexplored optima.  
3. A uniform UCB acquisition function with basic hyperparameter tuning is sufficient for initial exploratory coverage.

---

## Hypothesis (Directional)
If assumptions 1–3 hold, then **UCB BO** will improve or maintain the **Best Observed** values over random sampling for all functions.

---

## Methods & Design Choices
- **Surrogate Model:** Gaussian Process (RBF kernel, default hyperparameters)  
- **Acquisition Function:** Uniform Upper Confidence Bound (UCB)  
- **Initialisation:** Week 0 Initial Output, Week 1 Random Sampling  
- **Budget:** 1 UCB BO iteration per function (baseline exploratory insight)  
- **Constraints:** Minimal grid search for β, no exploitation-heavy adjustments  

---

## Performance Metrics

| Function | Week 0 (Initial) | Week 1 (Random) | Week 2 (UCB BO) | Best Observed |
|----------|-----------------|----------------|----------------|---------------|
| f1       | 7.711e-16       | 1.403e-21      | 9.069e-223     | 7.711e-16     |
| f2       | 0.6112          | -0.07641       | -0.07976       | 0.6112        |
| f3       | -0.03483        | -0.04117       | -0.20497       | -0.03483      |
| f4       | -4.02554        | -1.80143       | -2.04175       | -1.80143      |
| f5       | 1088.8596       | 684.5607       | 2434.3286      | 2434.3286     |
| f6       | -0.71427        | -0.68724       | -0.56486       | -0.56486      |
| f7       | 1.36497         | 0.02654        | 0.76536        | 1.36497       |
| f8       | 9.59848         | 8.44521        | 9.83513        | 9.83513       |

## Performance Metrics
| Function | Predicted (UCB) | Actual (Week 2) | Residual Error | Percent Error |
|----------|----------------|----------------|----------------|---------------|
| f1       | 0.0212         | 9.07e-223      | 9.07e-223      |  Large        |
| f2       | 1.8786         | -0.0798        | -1.9584        | -2453.24 %    |
| f3       | 0.4030         | -0.205         | -0.6080        | -296.32 %     |
| f4       | 24.8417        | -2.042         | -26.8837       | -1316.89 %    |
| f5       | 1945.3804      | 2434.329       | 488.9486       | 20.09 %       |
| f6       | 1.1214         | -0.5649        | -1.6863        | -298.47 %     |
| f7       | 2.3270         | 0.7654         | -1.5616        | -204.17 %     |
| f8       | 13.3933        | 9.8351         | -3.5582        | -36.18 %      |

**Observations:**
- **f5, f8:** UCB BO reached the true best observed value, demonstrating effective exploration.  
- **f1, f2, f3, f4:** UCB BO did not improve Best Observed; likely underexploration in regions with extreme minima or steep gradients.  
- **f6, f7:** Moderate improvement, indicating partial coverage of high-potential regions.  
- UCB BO performance varies by function shape and scale, highlighting the need for adaptive hyperparameters or multiple iterations.

---

## Critical Reflection Questions
1. **Which assumption failed first?**  
   Assumption 3 – uniform UCB without per-function calibration underexplored extreme or steep regions (f1–f4).

2. **Was failure due to bias, variance, uncertainty miscalibration, or optimisation error?**  
   Primarily **uncertainty miscalibration**, with GP variance underestimating steep minima/maxima.

3. **Did uncertainty guide or mislead decisions?**  
   Misled decisions on functions with extreme or skewed outputs, resulting in minimal gains.

4. **Would this approach be trusted in a production setting?**  
   Only for **baseline exploratory assessment**. Not sufficient for reliable optimisation without adaptive tuning or multiple iterations.

5. **What is the single most important insight gained?**  
   **Uniform UCB is effective for coarse exploration but requires per-function adaptation**; functions with extreme or highly non-linear outputs need targeted strategies.

---

### Summary
- Week 2 establishes a **baseline UCB BO framework**, showing where exploratory coverage succeeds or fails. 
- Next stage (Week 3) will incorporate bespoke strategies per function.


