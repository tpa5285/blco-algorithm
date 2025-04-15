"""
BLCO - Bi-Level Clustering Optimization
Author: Tanveer Ahmed & Vikash V. Gayah
Description: Minimizes standardized bias between treated and control groups using an iterative cluster replacement algorithm.
"""

import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def calculate_standardized_bias(treated, untreated, variables):
    means_treated = treated[variables].mean()
    variances_treated = treated[variables].var()
    means_untreated = untreated[variables].mean()
    variances_untreated = untreated[variables].var()

    standardized_biases = (means_treated - means_untreated) / np.sqrt((variances_treated + variances_untreated) / 2) * 100
    return np.sum(standardized_biases ** 2), np.mean(np.abs(standardized_biases))


def blco_algorithm(data, treatment_col, covariates, num_iterations=10000, lower_limit=0.1, upper_limit=0.9, learning_rate=0.01, random_seed=42):
    """
    Perform BLCO matching to reduce standardized bias between treated and untreated groups.

    Parameters:
    - data: DataFrame containing both treated and untreated observations
    - treatment_col: Name of the binary treatment indicator column (1 = treated, 0 = untreated)
    - covariates: List of covariate column names to use in matching
    - num_iterations: Number of optimization iterations
    - lower_limit: Minimum initial replacement probability
    - upper_limit: Maximum initial replacement probability
    - learning_rate: Rate at which probabilities are updated
    - random_seed: Seed for reproducibility

    Returns:
    - matched_data: DataFrame with matched treated and control units
    - convergence: List of tuples (iteration, SSSB) showing bias over iterations
    - runtime: Time taken to run the algorithm (seconds)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    df = data[covariates + [treatment_col]].copy()
    treated = df[df[treatment_col] == 1].copy()
    untreated = df[df[treatment_col] == 0].copy()

    initial_centroid = treated[covariates].mean().to_frame().T
    distances = np.linalg.norm(untreated[covariates].values - initial_centroid.values, axis=1)
    closest_indices = np.argsort(distances)[:len(treated)]
    initial_cluster = untreated.iloc[closest_indices]

    # Initialize probabilities
    cluster_probs = pd.Series(np.linspace(lower_limit, upper_limit, len(initial_cluster)), index=initial_cluster.index)
    unclustered_probs = pd.Series(np.linspace(upper_limit, lower_limit, len(untreated) - len(treated)), 
                                  index=untreated.drop(initial_cluster.index).index)

    current_cluster = initial_cluster.copy()
    current_bias, _ = calculate_standardized_bias(treated, current_cluster, covariates)
    best_cluster = current_cluster
    best_bias = current_bias

    convergence = []

    start_time = time.time()

    for iteration in range(num_iterations):
        out_index = random.choices(unclustered_probs.index, weights=unclustered_probs.values, k=1)[0]
        in_index = random.choices(cluster_probs.index, weights=cluster_probs.values, k=1)[0]

        new_cluster = current_cluster.drop(in_index)._append(untreated.loc[out_index])
        new_bias, _ = calculate_standardized_bias(treated, new_cluster, covariates)

        if new_bias < current_bias:
            convergence.append((iteration, new_bias))
            current_cluster = new_cluster
            current_bias = new_bias

            cluster_probs[in_index] -= learning_rate
            unclustered_probs[out_index] -= learning_rate

            # Save the values before dropping
            cluster_prob_value = cluster_probs[in_index]
            unclustered_prob_value = unclustered_probs[out_index]
            
            # Swap probabilities
            unclustered_probs = unclustered_probs.drop(out_index)._append(pd.Series(cluster_prob_value, index=[in_index]))
            cluster_probs = cluster_probs.drop(in_index)._append(pd.Series(unclustered_prob_value, index=[out_index]))

            if new_bias < best_bias:
                best_cluster = new_cluster
                best_bias = new_bias

    treated[treatment_col] = 1
    best_cluster[treatment_col] = 0
    matched_data = pd.concat([treated, best_cluster])
    runtime = time.time() - start_time

    return matched_data, convergence, runtime


def plot_convergence(convergence, title='BLCO Convergence'):
    iterations, biases = zip(*convergence)
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, biases, label='SSSB')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Standardized Bias (%)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
