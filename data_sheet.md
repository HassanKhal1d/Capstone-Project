# Dataset Datasheet — Black Box Optimisation Capstone Project

## Motivation

**For what purpose was the dataset created?**  
The dataset was created to support an educational capstone project focused on black-box optimisation and Bayesian optimisation techniques. Its purpose is to simulate realistic machine learning and optimisation problems where objective functions are unknown, evaluations are expensive, and only limited data is available. The dataset enables learners to practice exploration–exploitation trade-offs, surrogate modelling, and iterative decision-making under uncertainty.

**Who created the dataset and who funded it?**  
The dataset was created by the course development team of the programme delivering the Black Box Optimisation capstone. It was developed on behalf of the educational institution or platform hosting the programme. Funding and development were internal to the programme; no external commercial or public funding sources are disclosed.

---

## Composition

**What do the instances represent?**  
Each instance represents a single evaluation of a synthetic black-box function:
- The input is a vector of continuous numerical parameters (ranging from 2 to 8 dimensions).
- The output is a single scalar value representing the objective function value to be maximised.

**How many instances are there?**  
- Each of the eight functions is initially represented by **10 input–output pairs**.
- Additional instances are generated incrementally by the learner through weekly queries, at a rate of one new instance per function per round.

**Is there any missing data?**  
No. All provided input–output pairs are complete, with no missing values.

**Does the dataset contain confidential or sensitive data?**  
No. The dataset is fully synthetic and does not contain personal, confidential, or sensitive information.

---

## Collection Process

**How was the data acquired?**  
The data was generated programmatically by evaluating predefined synthetic black-box functions at selected input points. These functions are designed to exhibit properties such as noise, non-linearity, local optima, and increasing dimensional complexity.

**Sampling strategy**  
The initial 10 data points per function were selected by the dataset creators, likely using space-filling or random sampling strategies to provide minimal coverage of the input space.

**Time frame of data collection**  
The initial datasets were generated prior to the start of the capstone project. Additional data points are collected iteratively during the course as learners submit new queries.

---

## Preprocessing / Cleaning / Labelling

**Was any preprocessing or labeling performed?**  
Minimal preprocessing was applied. Data is provided in raw numerical form as NumPy `.npy` files:
- Inputs are stored as floating-point arrays.
- Outputs are stored as one-dimensional floating-point arrays.
No labeling, discretisation, or feature extraction was required.

**Was raw data saved?**  
Yes. The provided `.npy` files represent the raw generated data, and all newly acquired data is appended without transformation.

---

## Uses

**What other tasks could the dataset be used for?**  
- Benchmarking optimisation algorithms (e.g. Bayesian optimisation, random search).
- Teaching experimental design and surrogate modelling.
- Demonstrating exploration–exploitation trade-offs.
- Testing regression or response surface modelling methods.

**Are there risks or limitations for future use?**  
Because the dataset is synthetic and small, results should not be overgeneralised to real-world systems without validation. The limited number of observations and unknown noise structure may bias model performance if treated as a standard supervised learning dataset.

**Tasks the dataset should not be used for**  
- Any form of real-world decision-making involving humans or safety-critical systems.
- Fairness, bias, or ethical impact assessments, as the data does not represent real populations.

---

## Distribution

**How has the dataset been distributed?**  
The dataset is distributed privately through the programme’s learning platform as downloadable NumPy `.npy` files and via controlled access to a submission portal for additional evaluations.

**Licensing and terms of use**  
The dataset is subject to the programme’s internal terms of use. It is intended solely for educational purposes and is not licensed for commercial or public redistribution.

---

## Maintenance

**Who maintains the dataset?**  
The dataset and associated black-box functions are maintained by the programme’s instructional and technical team. Updates, evaluation outputs, and system availability are managed centrally through the capstone project portal.


