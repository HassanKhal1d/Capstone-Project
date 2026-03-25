# Black-Box Optimization Capstone Project
### Imperial Business School | Professional Certification in Machine Learning and Artificial Intelligence
#### Sequential Bayesian Optimization of Unknown Functions

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)
![Weeks](https://img.shields.io/badge/Submissions-13%20Weeks-purple?style=flat-square)
![Functions](https://img.shields.io/badge/Functions-f1%20to%20f8-red?style=flat-square)

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Relevance to Quantitative Finance](#2-relevance-to-quantitative-finance)
3. [Inputs and Outputs](#3-inputs-and-outputs)
4. [Challenge Objectives](#4-challenge-objectives)
5. [Technical Approach](#5-technical-approach)
6. [Results and Performance](#6-results-and-performance)
7. [Documentation](#7-documentation)
8. [Function Summary Cards](#8-function-summary-cards)
9. [Contact](#9-contact)

---

## 1. Project Overview

**Non Technical Explanation**

Imagine you are trying to find the highest point in a vast, foggy landscape where
you can only take one step at a time and each step costs you something. You cannot
see the terrain ahead, you have no map, and every step is permanent. That is
exactly the challenge this project tackles, applied to eight different mathematical
landscapes simultaneously over 13 weeks.

Each week, a decision was made about where to step next based on everything learned
so far. Smarter stepping strategies were developed over time, moving from random
guesses toward targeted, evidence-based decisions. By the final week, all eight
landscapes had been meaningfully improved, with one function improving by a factor
of over eight billion from its starting point. The project finished 9th out of 51
participants.

**Formal Description**

This repository documents a 13-week sequential **Black-Box Optimisation (BBO)**
challenge conducted as part of the MSc Financial Technology capstone at Imperial
Business School. The challenge involves optimising eight unknown oracle functions
(f1 through f8) under strict evaluation budgets, with no access to gradients,
analytical forms, or prior knowledge of the function landscape.

Each week, a single candidate input vector is submitted per function. The oracle
returns a scalar output. The objective is to accumulate the highest possible
**best-observed output** across all functions by the end of Week 13.

The project is a practical implementation of **model-based sequential optimisation**:
the exact framework used in hyperparameter tuning, algorithmic trading strategy
calibration, drug discovery, and engineering design. The constraint that each
evaluation is expensive and irreversible mirrors conditions encountered routinely
in production quantitative research.

> "Bayesian optimisation is what you do when every experiment costs money,
> time, or risk capital and you cannot afford to be wrong."

---

## 2. Relevance to Quantitative Finance

### Why This Matters on the Buy Side

In quantitative asset management, the problems that matter most are rarely
differentiable. Portfolio construction under transaction costs, optimal execution
scheduling, signal combination weights, and risk model calibration all involve
objective functions that are:

- **Expensive to evaluate** (require backtests, simulation, or live trading)
- **Noisy** (market regimes shift; the same parameter produces different outcomes
  at different times)
- **Without closed-form gradients** (trading costs, slippage, and regulatory
  constraints introduce non-smooth structure)

Bayesian optimisation is the principled response to this class of problems. The
same surrogate-acquisition loop developed here for abstract oracle functions maps
directly to:

| BBO Component | Quant Finance Analogue |
|---------------|----------------------|
| Oracle function | Backtest engine or live P&L function |
| Input vector x | Strategy parameters (lookback, threshold, position limits) |
| Output y | Sharpe ratio, information ratio, or drawdown-adjusted return |
| Surrogate model | Gaussian Process or ensemble fit to backtest results |
| Acquisition function | Decision rule for next parameter set to evaluate |
| Budget constraint | Compute budget or live capital at risk during search |
| Exploration vs exploitation | Trying new regimes vs refining known good parameters |

The mathematical framework behind this project, specifically Gaussian Process
regression, Expected Improvement, and active subspace detection, is directly
taught in graduate programmes in Mathematical Finance at Oxford, Cambridge, and
ETH Zurich as part of stochastic optimisation and quantitative modelling curricula.

For a buy-side quant researcher, the ability to design, run, and critically
evaluate a sequential optimisation pipeline is a core technical competency. This
project demonstrates that ability across eight structurally diverse problems under
realistic budget and uncertainty constraints.

---

## 3. Inputs and Outputs

### Input Format

Each function f{i} accepts a d-dimensional input vector x drawn from the unit
hypercube [0, 1]^d. Dimensions vary by function:

| Function | Dimensions (d) | Input Variables |
|----------|---------------|-----------------|
| f1 | 2 | X1, X2 |
| f2 | 2 | X1, X2 |
| f3 | 3 | X1, X2, X3 |
| f4 | 4 | X1, X2, X3, X4 |
| f5 | 4 | X1, X2, X3, X4 |
| f6 | 5 | X1, X2, X3, X4, X5 |
| f7 | 6 | X1, X2, X3, X4, X5, X6 |
| f8 | 8 | X1, X2, X3, X4, X5, X6, X7, X8 |

### Example Input (Week 13 Submissions)
```python
inputs_week_13 = [
    array([0.684583, 0.65543]),                          # f1
    array([0.703, 0.843]),                               # f2
    array([0.74689, 0.904, 0.47931]),                    # f3
    array([0.383446, 0.403478, 0.364093, 0.397868]),     # f4
    array([0.99982, 0.999861, 0.999107, 0.984214]),      # f5
    array([0.5574, 0.911, 0.6036, 0.7227, 0.1229]),      # f6
    array([0.078107, 0.222377, 0.608715, 0.229931,
           0.369375, 0.716646]),                          # f7
    array([0.163, 0.209525, 0.178, 0.198246, 0.813824,
           0.387519, 0.188, 0.66367])                    # f8
]
```

### Output Format

The oracle returns a single scalar value for each submitted input:
```python
outputs_week_13 = [
    -0.0012205886235109422,   # f1
     0.5956321650793815,      # f2
    -0.05251903077949714,     # f3
     0.3345514168224786,      # f4
  8343.237138327517,          # f5
    -0.9491766567356656,      # f6
     2.6948969486268437,      # f7
     9.96583818412            # f8
]
```

### Full Dataset

The complete 13-week input-output record is stored in:
- `inputs_13.txt` -- all submitted candidate vectors, indexed by week and function
- `outputs_13.txt` -- all corresponding oracle returns

---

## 4. Challenge Objectives

### Primary Objective

Maximise the **best observed output** for each function fi over 13 weeks subject
to the constraint of one submission per function per week.

### Constraints

- **Budget:** One oracle call per function per week; no replications
- **No gradient access:** The oracle returns only the scalar output
- **No analytical form:** The function structure is entirely unknown prior to querying
- **Irreversibility:** Submitted points cannot be retracted or modified
- **Domain:** All inputs must lie in [0, 1]^d

### Scoring

Performance is evaluated on the **best observed value** at the close of Week 13,
not the Week 13 submission itself. This rewards consistent improvement and
penalises regressions caused by over-aggressive acquisition.

### The Core Tension

Every submission must resolve the **exploration-exploitation trade-off**:

- **Explore** too much and you waste evaluations in uninformative regions
- **Exploit** too early and you converge to a local maximum before the global
  structure is understood

Navigating this tension intelligently, with evidence-based reasoning and
pre-committed decision rules, is the primary intellectual challenge of this project.

---

## 5. Technical Approach

### Philosophy

Every submission in this project was treated as a **controlled experiment**. Before
observing each result, a hypothesis was stated, a surrogate model was justified, and
a falsification condition was defined. This pre-commitment framework prevents
post-hoc rationalisation and ensures that failures are diagnosed structurally rather
than explained away.

This approach is borrowed directly from quantitative research methodology: a
strategy that cannot be falsified cannot be improved.

### Evolution by Phase

**Phase 1: Baseline Exploration (Weeks 1 to 2)**

Uniform random sampling and standard GP-UCB with default hyperparameters across
all functions. Established baseline best-observed values and initial EDA
(correlation, mutual information, partial correlation).

**Phase 2: Structured Surrogate Modelling (Weeks 3 to 5)**

Function-specific surrogate configurations introduced. Key methods:
- Heteroskedastic ARD-GP with per-function beta schedules
- GP + RF ensemble for weakly structured functions (f3, f7)
- Log-transform for f5 (output spans four orders of magnitude)
- REMBO low-dimensional embedding tested for f6

**Phase 3: Ensemble Diversification (Weeks 4 to 6)**

Four-model ensembles (GP, SVR, KNN, XGBoost) with cross-validated inverse-MSE
weighting introduced for f7 and f8. SVM-RFE feature consensus validated for f8
active subspace (X1, X3, X7 confirmed dominant).

**Phase 4: Competition-Validated Methods (Weeks 7 to 8)**

Competition-backed acquisition methods introduced:
- Max Value Entropy Search (MES) for f2 (SigOpt 2021 reference)
- CMA-ES hyperparameter tuning for f1, f4 (BBOB 2013 COCO reference)
- TuRBO-M three trust regions for f7 (Eriksson et al., NeurIPS 2019)
- Differential Evolution for f6 (SAASBO reference)
- Copula rank transformation for f6 (Salinas et al., ICML 2020)

**Phase 5: Structural Diagnosis and Rollback (Weeks 8 to 9)**

CMA-ES and Differential Evolution produced structurally incorrect hyperparameter
basins on f6 and f7. Multi-trust-region TuRBO-M caused a regression on f7 by
leaving the confirmed basin. Strategies rolled back to empirically validated
configurations. Key insight: gradient-free HP tuning solves the wrong problem
when the GP likelihood has a clear basin near the correct configuration.

**Phase 6: Geometry-First and Terminal Exploitation (Weeks 9 to 13)**

Surrogate-guided acquisition replaced by direct structural methods on saturated
functions:

| Function | Terminal Method | Rationale |
|----------|----------------|-----------|
| f1 | Quadratic ridge extrapolation in log10 space | GP acquisition surface flat; ridge is 1D and locally quadratic |
| f2 | GP posterior mean slice + maximin gap fill | GP argmax drifted to boundary artefacts; slice output more honest |
| f3 | KRR + fANOVA structural targeting | KNN/KRR outperformed GP consistently; plateau requires structural not surrogate guidance |
| f4 | Neighbourhood EI + X3 empirical floor | GP gradient near zero; empirical floor prevents catastrophic regression |
| f5 | Corner GP posterior mean + SIR tiebreaker | Calibration error below 1%; pure exploitation optimal under negligible uncertainty |
| f6 | Anchored kernel GP EI + quadratic bracketing | Asymmetric bounds rejected; interpolated X4/X5 peak at (0.744, 0.098) |
| f7 | Trust region ensemble + negative X2 gradient | X2 gradient signal untested until Week 13; produced final improvement |
| f8 | LogEI + inactive dimension fixing + lower-left probe | Active subspace fixing reduced 8D to 3D; probe below incumbent produced new best |

### Key Technical Decisions

**Output Warping**
- f1: Gaussian copula (Week 9) then log10 (Week 10 onward); copula
  produced 400 million times improvement in a single step
- f5: Log transform (Week 3 onward); raw-scale GP not viable
- f3: Box-Cox (Week 6 onward); stabilises weakly negative outputs

**Acquisition Function Selection**
- UCB: default exploratory phase (beta scheduled per function and week)
- EI: once basin identified and improvement target is meaningful
- LogEI: near-ceiling functions (f5, f8) where standard EI suffers
  numerical instability at small improvement margins
- GP posterior mean: terminal exploitation under negligible uncertainty

**Surrogate Selection Criteria**
- LOO cross-validation MSE as primary model quality signal
- ARD length scales as active subspace diagnostic
- fANOVA variance decomposition for dimension importance ranking
- Jackknife correlation stability for signal validation

---

## 6. Results and Performance

### Best Observed Values by Function (Week 0 vs Week 13)

| Function | Week 0 | Best Observed (Week 13) | Improvement Factor |
|----------|--------|------------------------|-------------------|
| f1 | 7.71 x 10^-16 | 6.60 x 10^-6 | ~8.6 billion x |
| f2 | 0.611 | 0.682 | +11.6% |
| f3 | -0.034835 | -0.025630 | +26.4% reduction in magnitude |
| f4 | -4.026 | 0.555 | Positive (was negative) |
| f5 | 1088.86 | 8343.24 | +666% |
| f6 | -0.714 | -0.224 | +68.6% reduction in magnitude |
| f7 | 1.365 | 2.695 | +97.4% |
| f8 | 9.598 | 9.966 | +3.8% (near-ceiling function) |

### Week-by-Week Best Observed Progression

**f1**
```
Week:  0      1      2      3      9      10     11     12     13
Best:  7e-16  7e-16  7e-16  5e-15  2e-9   2e-8   6e-7   7e-6   7e-6
```

**f2**
```
Week:  0      7      13
Best:  0.61   0.68   0.68
```

**f3**
```
Week:  0      11      12      13
Best:  -0.03  -0.027  -0.025  -0.025
```

**f4**
```
Week:  0      1      3      6      10     13
Best:  -4.02  -1.80  0.07   0.34   0.55   0.55
```

**f5**
```
Week:  0     1     2     3     7     10    11    12    13
Best:  1089  1089  2434  4171  6575  6972  7735  8219  8343
```

**f6**
```
Week:  0      1      2      9      10     13
Best:  -0.71  -0.68  -0.56  -0.24  -0.22  -0.22
```

**f7**
```
Week:  0     4     6     10    11    13
Best:  1.36  1.60  2.45  2.56  2.58  2.69
```

**f8**
```
Week:  0     2     3     10    11    13
Best:  9.60  9.84  9.95  9.94  9.94  9.97
```

### Key Performance Milestones

- **f1, Week 9:** Introduction of Gaussian copula transform produced the largest
  single-step improvement of the entire project (400 million x)
- **f4, Week 10:** First positive best observed value after nine weeks; achieved
  by L-BFGS-B gradient refinement within a tight trust region (radius 0.12)
- **f5:** Only function to achieve monotone improvement in every week from Week 7
  to Week 13; seven consecutive improvements
- **f8, Week 13:** New best (9.966) achieved on the final submission by probing
  slightly below the incumbent in both X1 and X3; first improvement since Week 3

---

## 7. Documentation

### Datasheet

A full dataset datasheet is provided, covering:
- Composition: summary statistics, N per function, structural behaviour
- Collection process: sampling strategy, time frame, sequential dependence
- Preprocessing: output transformations applied per function
- Uses and limitations: appropriate use cases and explicit restrictions
- Distribution and maintenance: access terms and retention policy

### Model Cards

Individual model cards for f1 through f8 are provided,
covering for each function:
- Purpose and intended use
- Training data and input specification
- Week-by-week performance table
- Key structural findings from EDA and GP diagnostics
- Limitations and failure mode analysis
- Recommendations for continued optimisation

### Submission Logs

Each weekly submission log in `` contains:
- Pre-committed hypothesis and decision rules
- Surrogate model specification and rationale
- Acquisition function design and parameter settings
- Post-submission reflection (available in the following week's log)

This structure implements the **pre-commitment experimental design framework**:
all hypotheses and failure conditions are stated before results are observed.

---

## 8. Function Summary Cards

| | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---|---|---|---|---|---|---|---|
| **Dimensions** | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 |
| **N at close** | 22 | 22 | 27 | 43 | 33 | 33 | 42 | 52 |
| **Best y** | 6.6e-6 | 0.682 | -0.026 | 0.555 | 8343 | -0.224 | 2.695 | 9.966 |
| **Active dims** | X2 only | X1, X2 | X2, X3 | All 4 | X2, X3 | X4, X5 | X6, X1 | X1, X3, X7 |
| **Key challenge** | 200 order magnitude span | Basin saturation | Near-flat plateau | Catastrophic walls | Heavy-tailed corner | Smooth tilt (not sparse) | Isolated basin | Near-deterministic ceiling |
| **Best surrogate** | Log10 GP + quadratic | GP (RBF + White) | KRR + fANOVA | ARD-GP + EI | Log-GP posterior mean | Anchored ARD-GP + EI | 4-model ensemble | ARD-GP + LogEI |
| **Breakthrough** | Copula transform (Wk 9) | Wk 5 UCB | KRR exploit (Wk 11) | Trust region (Wk 10) | Monotone Wks 7-13 | X4/X5 ridge (Wk 10) | X2 gradient (Wk 13) | Active subspace fix (Wk 13) |

---

## 9. Contact

**Author:** [Hassan Khalid] |
**Programme:** [Professional Certification in Machine Learning and Artificial Intelligence, Imperial Business School] |
**Timeline:** [2025/2026] |
**Email:** [H.Khalid-22@student.lboro.ac.uk] |
**LinkedIn:** [https://www.linkedin.com/in/hassan-khalid-160b36280/] |
**GitHub:** [https://github.com/HassanKhal1d] |

---

### Acknowledgements

This project was completed as part of the Imperial Business School BBO Capstone
Competition. Function evaluations were provided by the course teaching team.
The pre-commitment experimental design framework, surrogate model diagnostics, and
weekly submission logs are original work by the author.

The following academic works directly informed the technical approach:

- Eriksson et al. (2019). Scalable Global Optimisation via Local Bayesian
  Optimisation (TuRBO). NeurIPS.
- Eriksson and Jankowiak (2021). High-Dimensional Bayesian Optimisation with
  Sparse Axis-Aligned Subspaces (SAASBO). UAI.
- Salinas et al. (2020). Quantile Regression for Bayesian Optimisation with
  Copulas. ICML.
- Srinivas et al. (2010). Gaussian Process Optimisation in the Bandit Setting.
  ICML.
- Bull (2011). Convergence Rates of Efficient Global Optimisation Algorithms.
  JRSS-B.
- Hutter, Hoos, Leyton-Brown (2011). Sequential Model-Based Algorithm
  Configuration (SMAC). LION 5.
- Ament et al. (2023). Unexpected Improvements to Expected Improvement for
  Bayesian Optimisation. NeurIPS.

---

### Visuals
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/e2b198fc-00f6-45b7-a2e2-e5e93343a549" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/f1a2e120-f578-44c9-ac01-e61956c7901c" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/c2c06d7a-22f5-4a34-97eb-85fc2bf63d9e" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/702fb8a5-5e85-495f-b639-1338a8f69429" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/fe884e92-371b-45c0-b99b-99c365f020ef" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/6461d8c5-dcc6-41b3-8cdc-85bb2a652d35" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/82215e43-35d9-4b78-8899-8d8fc39f97dd" />
<img width="984" height="584" alt="download" src="https://github.com/user-attachments/assets/384cb316-a1d7-4903-897c-167c70c753bc" />
<img width="1396" height="527" alt="image" src="https://github.com/user-attachments/assets/af6f8d57-4065-45c4-b475-5638070f5d38" />
<img width="921" height="337" alt="image" src="https://github.com/user-attachments/assets/28b68dbd-36a5-40a4-b99c-df65fccdddd0" />
<img width="860" height="332" alt="image" src="https://github.com/user-attachments/assets/d480a8cd-f09a-49d9-ade1-9a2916f660b9" />
<img width="1292" height="332" alt="image" src="https://github.com/user-attachments/assets/3aa2093a-f0b9-48f6-8415-7369e20ba7b7" />
<img width="1732" height="335" alt="image" src="https://github.com/user-attachments/assets/cd7e668d-a5ce-42e1-861c-435715b473c8" />
<img width="1738" height="351" alt="image" src="https://github.com/user-attachments/assets/a7747156-3568-4ce6-adc0-0e9b8b29c209" />
<img width="1606" height="1184" alt="download" src="https://github.com/user-attachments/assets/a807e37a-8583-48b2-86b3-635b1faebb2a" />
<img width="1178" height="584" alt="download" src="https://github.com/user-attachments/assets/c9488cce-7486-4f14-ad03-0ef4453e6cb0" />
<img width="1179" height="884" alt="download" src="https://github.com/user-attachments/assets/aff0a11a-320d-4d86-8ded-cdc166a383f1" />

---

*This README was last updated at the close of Week 13.*
*All performance figures reflect final competition state.*
