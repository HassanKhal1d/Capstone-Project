# Dataset Datasheet | Black Box Optimization Capstone Project


| Category        | Details |
|-----------------|--------|
| **Competition** | Black-Box Optimisation (BBO) — Academic Module Competition |
| **Functions** | f1 (3D), f2 (3D), f3 (4D), f4 (5D), f5 (5D), f6 (6D), f7 (7D), f8 (9D) |
| **Input domain** | \[0, 1\]^d for all functions; all dimensions bounded identically |
| **Objective** | Maximise the unknown scalar output of each function |
| **Query budget** | 13 submissions across all functions; 1 query per function per week |
| **Primary use** | Surrogate modelling, Bayesian optimization, experimental design research |


### Motivation

**What task does this dataset help to solve?**  
This dataset was generated to support sequential black-box optimization across eight unknown functions, f1 through f8, of varying dimensionality (3D to 9D). All inputs are constrained to [0,1]^d. The task is to find the global maximum of each function using only the observed input-output pairs, with no access to gradients, functional form, or noise structure. The dataset records every query submitted to the competition oracle and the corresponding output returned, forming the empirical basis for surrogate model fitting, acquisition function design, and landscape structure discovery.

**Who created the dataset and who funded it?**  
The dataset was provided as part of a Bayesian Optimization competition within the ML and AI certificate programme delivered by Imperial College Business School in collaboration with Emeritus. The underlying function generator used to create the dataset was not publicly disclosed, and therefore cannot be definitively attributed to either institution. All observed data points were generated through the student’s sequential submissions during the competition, under a constrained evaluation budget, forming the final dataset used for analysis.


### Composition

**What does it contain?**  
The dataset comprises eight independently sampled black-box function datasets (f1–f8), collectively containing 271 input–output observations accumulated across thirteen weeks of sequential Bayesian optimization submissions. Each observation consists of a normalised input vector drawn from [0, 1]^d and a scalar oracle output, with dimensionality ranging from two (f1, f2) to eight (f8). All data were generated through active experimentation rather than drawn from a pre-existing corpus, meaning the datasets are incomplete samples of their respective continuous domains and carry no recommended train/test split. Instance counts at the close of Week 13 range from 22 observations (f1, f2) to 52 (f8), reflecting both the per-function submission budget and the varying difficulty of identifying productive regions. No missing values or incomplete instances are present. The datasets exhibit substantial structural heterogeneity: output ranges span from a narrow interval of roughly 0.76 units (f2) to over five orders of magnitude (f5), and the proportion of observations that lie in high-value regions varies from nearly all (f8) to fewer than five percent (f1, f4, f7). Observations within each function are sequentially dependent, as every week's sampling strategy was conditioned on all prior results, making the datasets more informative than equivalent random samples of the same size but also subject to sequential sampling bias toward regions identified as promising in earlier weeks.


| Function | Dimensions | N (Week 13) | Output Range | Best y | Active Dimensions | Primary Challenge |
|----------|-----------|-------------|-------------|-----------------|-------------------|-------------------|
| f1 | 2 | 22 | ~[−3.6×10⁻³, 6.6×10⁻⁶] | 6.6 × 10⁻⁶ | X2 only | Ultra-narrow ridge; 200 order-of-magnitude output span |
| f2 | 2 | 22 | [−0.080, 0.682] | 0.682 | X1 (dominant), X2 (secondary) | Small sample; GP hyperparameter instability |
| f3 | 3 | 27 | [−0.399, −0.026] | −0.026 | X2, X3 (separable) | High noise; flat plateau; KNN > GP throughout |
| f4 | 4 | 43 | [−32.6, 0.555] | 0.555 | All 4 (isotropic) | Extremely narrow positive plateau; catastrophic walls |
| f5 | 4 | 33 | [0.113, 8343.2] | 8343.2 | X2, X3 (dominant) | Heavy-tailed; 5-order magnitude range; monotone corner |
| f6 | 5 | 33 | [−2.757, −0.225] | −0.225 | X4, X5 (94% variance) | Broad smooth tilt; no sparse subspace despite prior belief |
| f7 | 6 | 42 | [0.003, 2.695] | 2.695 | X6 (highly active), X1, X2 | Isolated basin; weak global structure; 91% observations y < 0.7 |
| f8 | 8 | 52 | [5.541, 9.966] | 9.966 | X1, X3, X7 | Near-deterministic; dual-basin structure; inactive dimension fixing critical |



### Collection Process

**How was the data acquired?**  
Queries were generated through a sequential Bayesian optimzation pipeline executed once per week per function. The core loop was: fit a surrogate model to all accumulated (x, y) pairs; optimise an acquisition function over [0,1]^d to select the next candidate; submit to the oracle; append the returned (x, y) pair to the dataset; update EDA. This loop ran for thirteen active weeks (Weeks 1 through 13), with Week 0 providing the initial oracle-supplied point for each function.

**Sampling Strategy**
The sampling strategy was neither purely random nor fully deterministic; it was adaptive
and hypothesis-driven, evolving materially across weeks in response to observed outcomes.
Week 1 used uniform random sampling to establish a baseline. From Week 2 onward, candidate
points were selected by optimizing acquisition functions over surrogate models fitted to all
accumulated observations. The primary acquisition functions employed were:

- **Upper Confidence Bound (UCB)** — used throughout Weeks 2–8 as the default exploratory
  mechanism, with function-specific β parameters adjusted to balance exploration and exploitation
- **Expected Improvement (EI)** — adopted from Week 3 onward for functions with well-identified
  basins, particularly f4 and f5, where the goal shifted to beating a known incumbent
- **Thompson Sampling** — applied selectively in Weeks 5 and 7 to inject stochastic diversity
  and escape confirmed local maxima
- **Max Value Entropy Search (MES)** — trialled on f2 in Week 7 as a competition-backed
  alternative targeting global maximum uncertainty reduction
- **Log Expected Improvement (LogEI)** — introduced in Weeks 10–13 for near-ceiling functions
  (f5, f8) where standard EI suffered numerical instability near strong incumbents

Surrogate models evolved in parallel, progressing from single Gaussian Processes with RBF
or Matérn kernels through increasingly structured configurations:

- **Gaussian Processes with ARD kernels** — the primary global surrogate throughout, with
  asymmetric length-scale bounds introduced from Week 8 to enforce known active subspace
  structure
- **Support Vector Regressors, k-Nearest Neighbours, and XGBoost** — incorporated as ensemble
  components from Week 4 onward, weighted by cross-validated inverse MSE
- **Kernel Ridge Regression** — adopted for f3 from Week 9 as a numerically stable alternative
  to GP marginal likelihood optimisation at small sample sizes
- **Nadaraya–Watson kernel regression** — used on f2 in Week 11 when GP hyperparameter
  fitting became unreliable

Candidate generation relied on quasi-random **Sobol** and **Halton** sequences to ensure
low-discrepancy domain coverage, consistently filtered by minimum-distance guards
(δ_min ∈ [0.02, 0.07] depending on function and week) to prevent near-duplicate evaluations.
From Week 9 onward, for functions where structural knowledge was sufficient, deterministic
or geometry-first methods replaced acquisition-function-driven selection entirely:

- **Quadratic ridge extrapolation in log₁₀ space** for f1, targeting the estimated peak of
  the X2 ridge directly from three locally informative observations
- **Maximin gap-filling** for f2 and f3, placing queries at the largest unsampled gap within
  the confirmed high-value region when surrogate models were too unstable to guide selection
- **Corner-restricted GP posterior mean maximisation with per-dimension floors** for f5,
  reflecting confirmed monotone structure toward the (1,1,1,1) boundary
- **Active-subspace-fixed LogEI with inactive dimensions anchored to incumbent values** for f8,
  reducing the effective search space from eight to three dimensions

As a result, the accumulated datasets are subject to **sequential sampling bias**: regions
identified as promising early in the process are systematically over-represented relative to
a uniform random sample of the same size. Functions where the productive region was identified
early (f5, f8) have the most concentrated late-stage observations, while functions that
resisted structural identification (f3, f7) retain comparatively more dispersed coverage
across the domain.

**Time frame of data collection**  
Data were collected over fourteen discrete weeks (Week 0 through Week 13) during the Spring 2025 term of the Imperial Business School BBO Capstone Project. Each week constituted one observation per function, yielding a maximum of fourteen observations per function from the submission process. Week 0 observations were provided by the course oracle as fixed starting points and were not chosen by the student. The total active collection period spanned approximately thirteen weeks of sequential decision-making, with one candidate submitted per function per week under a strict budget constraint. No data were collected outside this structured weekly cadence, and no retrospective or batch collection took place.

### Preprocessing / Cleaning / Labelling

**Was any preprocessing or labeling performed?**  
Transformations were applied selectively and only when EDA provided statistical
justification. No global preprocessing pipeline was applied uniformly across functions.
All transformations are pipeline steps applied at model-fitting time; raw (x, y) records
are stored without modification.

The most important and consistently validated transformation was a **log-transform of f5
outputs**, applied from Week 3 onward. f5's output range spans four orders of magnitude
(685 to 8343), making GP fitting in the original scale unreliable due to severe variance
heterogeneity. The log-transform stabilised the variance and enabled reliable uncertainty
estimates throughout the remaining weeks.

For **f1**, a **Gaussian copula transform** was introduced in Week 9 to address the
pathological output scale, where values ranged from effectively zero (10^-248) to the
then-best observed value of approximately 10^-9. The copula transform maps outputs through
their empirical rank to a standard normal distribution, preserving rank order while
removing the extreme scale compression that caused GP uncertainty estimates to collapse.
This transformation produced an immediate and substantial improvement: the Week 9
submission returned y = 1.87 x 10^-9, a jump of approximately 400 million times over the
previous best of 5.31 x 10^-15. As further high-value observations accumulated in the
productive ridge and the dataset grew more concentrated near the peak, the copula
transform was subsequently replaced by a direct **log10 transform** from Week 10 onward.
The log10 transform was preferred at that stage because the productive observations were
sufficiently dense to define stable local curvature, and a quadratic model fitted in
log10 space could directly estimate the ridge peak, a calculation that is not meaningful
in copula-transformed space where the scale carries no geometric interpretation.

A **Yeo-Johnson input warp on X1** was tested for f2 in Week 5. The hypothesis was that
X1 had a non-linear monotonic relationship with f2's outputs. The test falsified this: EDA
confirmed the relationship was moderately linear (Pearson r = 0.424, Spearman rho = 0.361),
meaning the warp distorted the kernel distance metric without correcting any genuine
non-stationarity. The warp was removed in Week 6.

A **Box-Cox output warp** was applied to f3 from Week 6 onward to stabilise GP fitting
on its weakly negative outputs, with the condition that it would be retained only if
surrogate performance remained stable. No other transformations were applied across the
dataset.

**What are the intended uses?**  
The dataset is intended for educational and research use in surrogate-based optimization,
specifically for:

- **Comparing surrogate model families** — including Gaussian Processes, Kernel Ridge
  Regression, k-Nearest Neighbours, Support Vector Regressors, XGBoost, and ensemble
  combinations thereof — as point estimators and uncertainty providers under small-sample
  sequential conditions
- **Active subspace detection** — using ARD length-scales, partial correlation, mutual
  information, and SVM-RFE on functions with confirmed sparse or low-effective-dimensional
  structure (f6, f7, f8)
- **Studying the pre-commitment experimental design framework** — where hypotheses,
  decision rules, and failure conditions are specified in full before results are observed,
  enabling honest evaluation of surrogate-guided strategies against their stated assumptions

**What are the inappropriate uses?**
 The dataset should not be used to:

- **Claim statistically validated global optima** — with N approximately 9 to 13 active
  submissions per function, no guarantee exists that the best-observed value is near the
  true global maximum; the sequential sampling process may have missed high-value regions
  entirely, particularly for functions with narrow productive subspaces (f1, f4) or
  diffuse weakly-structured landscapes (f3, f7)
- **Generalise structural findings across functions** — findings such as X3 dominating f5
  or X1 and X3 defining the active subspace of f8 are specific to those functions and
  cannot be extrapolated to other black-box problems
- **Support high-stakes production decisions** — the dataset was generated under strict
  educational budget constraints and is not a substitute for rigorous optimisation studies
  with adequate evaluation budgets
- **Treat diagnostic pass conditions as validation of landscape accuracy** — a confidence
  interval pass or a favourable z-score means the surrogate's uncertainty was appropriately
  sized at the one queried point; it does not imply that the surrogate's reconstruction
  of the broader response surface is correct

**Are there risks or limitations for future use?**  
The most consequential limitation of the dataset is **acquisition bias**: after Week 1,
all queries were drawn from regions the surrogate identified as promising, producing a
highly non-uniform design in which under-performing areas of the domain are systematically
undersampled. This means the dataset cannot support claims about function behaviour outside
the explored regions, and observed best values may reflect the limits of the search
strategy rather than the limits of the function itself.

A related limitation is **confirmation bias in structural hypothesis formation**. Structural
beliefs established in early weeks — for example, that X3 dominates f5, or that the
productive region of f4 is centred at mid-range values across all dimensions — directly
influenced where later queries were placed. If those early beliefs were partially incorrect,
the subsequent sampling pattern would have systematically reinforced an incomplete picture
of the response surface rather than testing it.

A third persistent limitation is **strategy conflation**. Each week typically introduced
multiple simultaneous changes, including the surrogate model family, the acquisition
function, the hyperparameter tuning method, and the candidate generation strategy. Because
only one observation was returned per function per week, it is generally impossible to
attribute an improvement or regression to any single component. The dataset therefore
supports qualitative learning about which combinations of methods tended to work, but
does not support controlled causal attribution of performance to individual design choices.


### Distribution

**How has the dataset been distributed?**

The dataset is stored within the project knowledge repository as two structured files:
inputs_13.txt and outputs_13.txt, which contain the complete sequence of submitted
candidate vectors and oracle returns across all thirteen weeks and all eight functions.
The accompanying submission logs document strategies and reasoning but do not contain
raw data. No public repository, API, or physical media distribution has been established.

**When will the dataset be made available?**

The dataset is currently available within the academic assessment context for which it
was produced. Any future distribution beyond this context would require explicit approval
from the course administrators responsible for the competition oracle.

**What are the terms of use?**

The dataset was generated within an academic module competition at Imperial Business
School. Sharing is limited to academic assessment contexts. Any use should credit the
competition framework from which the oracle evaluations were obtained. The functions
f1 through f8 are proprietary to the competition and cannot be reverse-engineered from
this dataset alone. Query points must not be submitted to any active or future iteration
of the same competition. No fees or royalties are associated with legitimate academic use.

**Additional comments**

The dataset is best used as eight (X, y) DataFrames reconstructed from inputs_13.txt
and outputs_13.txt. Researchers should note that observations are sequentially dependent
and the sampling design is heavily non-uniform due to acquisition bias, making the
dataset well suited for studying sequential surrogate-based decision-making but not for
tasks assuming independent and identically distributed samples.


### Maintenance

The dataset was created and maintained by the student researcher throughout the duration
of the project. It is append-only: each weekly submission adds exactly one record per
function and no prior records are modified. Corrections discovered after submission are
documented in the following week's log rather than applied retroactively, preserving the
integrity of the sequential experimental record.

Versioning is implicit through week numbering. The dataset is considered complete and
frozen at Week 13, with the input and output files and submission logs together
constituting the final archive stored as part of the academic coursework submission.
Retention follows Imperial Business School's institutional data retention policies for
coursework.
