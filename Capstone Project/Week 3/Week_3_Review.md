# Critical Reflection on Week 3 Results (With Full Rationale)

This reflection analyses each function using:

- EDA structural findings  
- Residual error  
- % change  
- NLL  
- Z-score  
- 95% CI check  
- Quantile position  
- Calibration Gap  

For each function, I answer:

1. Which assumption failed first  
2. Type of failure  
3. Whether uncertainty guided or misled  
4. Production readiness  
5. Key insight  

Each answer now includes explicit data justification.

---

## f1 — Near-Zero / Flat Landscape

**Key Metrics**

- Residual: `+0.000022`  
- NLL: `-6.7265`  
- Z: `0.0768`  
- CI: PASS  
- Quantile: `0.5306`  
- Calibration Gap: `0.0306`  

### 1. First Failed Assumption
- **Assumption:** Meaningful exploitable structure exists  
- **Rationale:**  
  - EDA showed negligible correlation and mutual information.  
  - Residual ≈ 0 confirms no exploitable gradient.  
  - Z ≈ 0 and quantile ≈ 0.53 show predictions were centered but did not improve anything.  
  - Model worked correctly, optimisation problem was trivial.

### 2. Failure Type
- **None** (no bias, variance, or calibration failure)  
- **Rationale:**  
  - Calibration Gap = 0.0306 (very low)  
  - CI passed  
  - Extremely negative NLL confirms correct uncertainty modelling

### 3. Did Uncertainty Guide or Mislead?
- **Guided correctly**  
- **Rationale:**  
  - Z near zero + quantile central → appropriate uncertainty expressed

### 4. Production Trust
- **Yes**  
- **Rationale:** No hallucinated certainty, no tail risk misestimation

### 5. Key Insight
- In flat regimes, calibration quality matters more than optimisation intensity

---

## f2 — Clean Low-Dimensional Structure

**Key Metrics**

- Residual: `-0.1816`  
- Z: `-0.67`  
- CI: PASS  
- Quantile: `0.2514`  
- Calibration Gap: `0.2486`  
- NLL: `-1.5446`  

### 1. First Failed Assumption
- **Assumption:** Perfectly modelled curvature  
- **Rationale:**  
  - Residual negative → slight overestimation  
  - Quantile 0.25 → systematic downward bias  
  - Z < 1 and CI pass → curvature mostly captured

### 2. Failure Type
- **Small bias + mild variance underestimation**  
- **Rationale:** Calibration Gap ≈ 0.25 → modest compression of variance. NLL still negative

### 3. Did Uncertainty Guide?
- **Yes**  
- **Rationale:** No CI failure, Z within ±1, optimisation stable

### 4. Production Trust
- **Yes**  
- **Rationale:** No extreme tails, no overconfidence

### 5. Key Insight
- Low effective dimensionality → stable GP behaviour

---

## f3 — Weak Nonlinear Structure

**Key Metrics**

- Residual: `-0.1834`  
- Z: `-0.8494`  
- CI: PASS  
- Quantile: `0.1978`  
- Calibration Gap: `0.3022`  
- NLL: `-1.4774`  

### 1. First Failed Assumption
- **Assumption:** Stable monotonic structure  
- **Rationale:**  
  - Quantile ~0.20 → truth drifting toward lower tail  
  - Calibration Gap increased from 0.25 (f2) to 0.30  
  - Structure less predictable than f2

### 2. Failure Type
- **Mild bias + moderate uncertainty compression**  
- **Rationale:** Residual similar to f2, larger calibration gap shows variance shrinking relative to actual dispersion

### 3. Did Uncertainty Guide?
- **Mostly yes**  
- **Rationale:** Z < 1, CI passed, but quantile shift suggests risk accumulation

### 4. Production Trust
- **With monitoring**  
- **Rationale:** Trend toward tail compression must be tracked

### 5. Key Insight
- Variance deterioration begins before catastrophic failure

---

## f4 — Structured Surface with Exploitation Failure

**Key Metrics**

- Residual: `-0.4224`  
- Z: `2.1712`  
- CI: FAIL  
- Quantile: `0.9850`  
- Calibration Gap: `0.4850`  
- NLL: `2.6019`  

### 1. First Failed Assumption
- **Assumption:** Local smoothness during exploitation  
- **Rationale:**  
  - Quantile 0.985 → truth nearly outside predictive mass  
  - CI failure → underestimated variance  
  - Z > 2 → statistical miscalibration

### 2. Failure Type
- **Severe uncertainty miscalibration + optimisation error**  
- **Rationale:** Calibration Gap ≈ 0.49, NLL positive and large → poor likelihood fit

### 3. Did Uncertainty Guide?
- **No — misled exploitation**  
- **Rationale:** High confidence in incorrect region

### 4. Production Trust
- **No**  
- **Rationale:** Tail misestimation → unstable decision-making

### 5. Key Insight
- Overconfidence amplifies optimisation error

---

## f5 — Large Magnitude Surface (Log-Stabilised)

**Key Metrics**

- Residual: `+3623`  
- Z: `0.4585`  
- CI: PASS  
- Quantile: `0.6767`  
- Calibration Gap: `0.1767`  
- NLL (log): `2.5116`  

### 1. First Failed Assumption
- **None — scaling handled correctly**  
- **Rationale:** Large raw residual, small Z → scale effect; log transform stabilised

### 2. Failure Type
- **No structural failure**

### 3. Did Uncertainty Guide?
- **Yes**  
- **Rationale:** CI passed, calibration gap low

### 4. Production Trust
- **Yes**

### 5. Key Insight
- Transformation choice determines stability

---

## f6 — Sparse 5D Structure

**Key Metrics**

- Residual: `-3.945`  
- Z: `-1.6242`  
- CI: PASS (borderline)  
- Quantile: `0.0522`  
- Calibration Gap: `0.4478`  
- NLL: `1.6325`  

### 1. First Failed Assumption
- **Assumption:** Equal feature importance  
- **Rationale:** Quantile ~0.05 → truth near lower extreme, high calibration gap → variance compression

### 2. Failure Type
- **Moderate uncertainty miscalibration**

### 3. Did Uncertainty Guide?
- **Partially — underestimated downside risk**

### 4. Production Trust
- **Only with ARD enforcement**

### 5. Key Insight
- Effective dimensionality drives uncertainty quality

---

## f7 — Severe Overconfidence (Critical Failure)

**Key Metrics**

- Residual: `-1.247`  
- Z: `-4.1110`  
- CI: FAIL  
- Quantile: `0.0000`  
- Calibration Gap: `0.5000`  
- NLL: `7.0501`  

### 1. First Failed Assumption
- **Assumption:** Ensemble averaging guarantees calibration  
- **Rationale:** Quantile 0 → truth outside distribution, Calibration Gap 0.5, Z = -4 → extreme deviation

### 2. Failure Type
- **Severe uncertainty collapse + optimisation failure**

### 3. Did Uncertainty Guide?
- **No — strongly misled**

### 4. Production Trust
- **No**

### 5. Key Insight
- Uncertainty failure more dangerous than mean error

---

## f8 — Accurate Mean, Narrow Variance

**Key Metrics**

- Residual: `-0.1358`  
- Z: `-1.8453`  
- CI: PASS  
- Quantile: `0.0325`  
- Calibration Gap: `0.4675`  
- NLL: `-0.3288`  

### 1. First Failed Assumption
- **Assumption:** Posterior variance captured tail risk  
- **Rationale:** Quantile 0.03 → near-boundary, high calibration gap → variance compression

### 2. Failure Type
- **Variance underestimation (mild)**

### 3. Did Uncertainty Guide?
- **Yes — optimisation succeeded despite compression**

### 4. Production Trust
- **Yes, with variance inflation safeguard**

### 5. Key Insight
- Mean accuracy and calibration quality are independent

---

## Final Cross-Function Diagnosis

Across all functions:

- Catastrophic failures (f4, f7) share:  
  - Z > ±2  
  - CI failure  
  - Calibration Gap > 0.45  
  - High positive NLL  

- Residual magnitude alone **did not predict collapse**  
- Calibration metrics were **earlier warning signals**

### Core Learning

- In Bayesian Optimisation:  
  - **Uncertainty calibration** → primary determinant of decision quality  
  - **Mean error** → affects efficiency  
  - **Miscalibrated uncertainty** → causes strategic failure
