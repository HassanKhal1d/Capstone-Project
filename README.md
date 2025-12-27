# Black Box Optimisation


## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
This project is about making the best possible decisions when information is limited and expensive to obtain. We are given eight “black box” systems, meaning we can see their inputs and outputs but not how they work internally. Each system represents a real-world problem, such as improving a recipe, tuning a machine learning model, or finding an optimal chemical mixture. We can only test a small number of input combinations, so every test must be chosen carefully. Using intelligent trial-and-error strategies inspired by Bayesian optimisation, the goal is to gradually learn from past results and identify the input settings that produce the best possible outcome for each system. 

## DATA
The data used in this project is provided directly as part of the capstone challenge materials. It consists of eight separate datasets, one for each black-box optimisation task, corresponding to functions with input dimensionalities ranging from 2D to 8D.

For each function, the initial dataset includes:

An input matrix stored as a NumPy .npy file, where each row represents a previously evaluated input point in the parameter space (e.g. 2 to 8 parameters depending on the function).

A corresponding output vector, also stored as a .npy file, containing the scalar objective values returned by the black-box function for each input.

Each dataset starts with ten initial observations, which serve as the seed data for guiding further optimisation. The data is synthetic but designed to emulate real-world optimisation scenarios such as hyperparameter tuning, chemical process optimisation, and resource allocation, incorporating properties like noise, non-linearity and local optima.

The datasets are loaded and inspected using NumPy’s np.load() function in Python. All additional data points are generated iteratively by querying the black-box functions via the capstone project submission portal.

Data source:
Capstone Project: Black Box Optimisation, provided by the programme learning platform (internal course material; no external public dataset).

## MODEL 
A summary of the model you’re using and why you chose it. 

## HYPERPARAMETER OPTIMSATION
Description of which hyperparameters you have and how you chose to optimise them. 

## RESULTS
A summary of your results and what you can learn from your model 

You can include images of plots using the code below:


## (OPTIONAL: CONTACT DETAILS)
If you are planning on making your github repo public you may wish to include some contact information such as a link to your twitter or an email address. 
