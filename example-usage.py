import pandas as pd
from blco.blco import blco_algorithm, plot_convergence

# Load your data (or replace with actual loading logic)
# split_10_data = pd.read_csv("path_to_your_data.csv")

# Example usage
matched_data, convergence, runtime = blco_algorithm(
    data=split_10_data,
    treatment_col="only_cl",
    covariates=["aadt_yr", "length_mi", "lanewidth", "psl_45p", "l_s_pave", "r_s_pave", 
                "rhr_123", "rhr_45", "rhr_67", "degree", "centralang"],
    num_iterations=1000,
    learning_rate=0.01
)

print(f"Runtime: {runtime:.2f} seconds")
plot_convergence(convergence)
