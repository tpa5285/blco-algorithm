# blco-algorithm
BLCO: A Bi-level Clustering Optimization algorithm for bias reduction in observational studies. This is a model-free, data-driven algorithm designed to minimize **standardized bias** when selecting matched control groups for treatment effect evaluation. It is especially useful for crash modification factor (CMF) estimation, causal inference, and observational safety studies where randomized control trials are not feasible.

This repository contains a Python implementation of the BLCO algorithm that iteratively replaces control samples to reduce the **sum of squared standardized bias (SSSB)** between treated and matched untreated groups.

---

## 📄 Reference

For more details on the method, please refer to our paper currently under review at *Accident Analysis and Prevention*:

> Ahmed, T., & Gayah, V.V. (2025) A novel bi-level clustering optimization approach to balance treatment of crash data. [Under-review at Accident Analysis and Prevention]

We will update the citation and DOI once the paper is published.

## 💡 Key Features

- Model-free approach: No need to estimate a propensity score model
- Directly minimizes sum of squared standardized bias (SSSB)
- Incorporates a competitive learning framework for adaptive matching
- Customizable inputs: user-defined covariates, learning rate, iteration count, etc.
- Visualizes convergence over time

Required Libraries
pandas
numpy
matplotlib
scikit-learn

blco-algorithm/

├── blco/
│   ├── __init__.py
│   └── blco.py                   ← main algorithm code
├── examples/
│   ├── example_usage.py          ← Basic Python example
├── README.md
├── requirements.txt
├── LICENSE                       



📊 Inputs

Parameter	Description
data	A pandas DataFrame containing both treated and untreated observations
treatment_col	Column name indicating treatment status (1 = treated, 0 = control)
covariates	List of column names to use in matching
num_iterations	Number of optimization iterations (default: 10000)
learning_rate	Step size for adjusting replacement probabilities (default: 0.01)
lower_limit and upper_limit	Initial bounds for replacement probabilities

📈 Output
matched_data: DataFrame with matched treated and control units
convergence: List of (iteration, SSSB) values showing algorithm progress
runtime: Total time in seconds for the algorithm to run

📬 Contact
For questions or collaboration inquiries, please reach out via GitHub Issues or email:
Tanveer Ahmed – First Author - tpa5285@psu.edu
Vikash V. Gayah – Co-author - gayah@engr.psu.edu



