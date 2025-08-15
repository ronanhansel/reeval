import argparse
import pandas as pd
import numpy as np
from catsim.cat import generate_item_bank
from catsim.simulation import Simulator
from catsim.initialization import FixedInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import EAPEstimator
from catsim.stopping import MinErrorStopper

def run_cat_simulation(input_path, output_path, selection, estimator, min_se, max_items):
    """
    Runs a Computerized Adaptive Testing (CAT) simulation based on IRT parameters and response data.

    Args:
        input_path (str): Path to the input CSV file.
                          The file must contain IRT parameters (a, b, g, u) and response data.
        output_path (str): Path to save the resulting theta estimates.
        selection (str): The item selection method (e.g., 'MFI').
        estimator (str): The ability estimation method (e.g., 'EAP').
        min_se (float): The minimum standard error to stop the test.
        max_items (int): The maximum number of items to administer.
    """
    print("Loading data...")
    # Load the data using pandas
    data_df = pd.read_csv(input_path)

    # Separate IRT parameters and response data
    # Assumes columns are named 'a', 'b', 'g', 'u', followed by response columns
    irt_param_cols = ['a', 'b', 'g', 'u']
    response_cols = [col for col in data_df.columns if col not in irt_param_cols]
    
    # The item bank needs to be a numpy array of shape (num_items, 4)
    # The columns must be in the order: [discrimination, difficulty, guessing, slipping]
    item_bank = data_df[irt_param_cols].to_numpy()
    
    # The responses need to be a numpy array of shape (num_respondents, num_items)
    responses_matrix = data_df[response_cols].to_numpy()
    
    num_respondents, num_items = responses_matrix.shape
    print(f"Data loaded: {num_respondents} respondents, {num_items} items.")

    # --- Setup CAT Components ---
    # These components mirror the options in the original R script
    
    # 1. Initializer: How to start the test (e.g., at what ability level)
    initializer = FixedInitializer(0.0) # Start with theta = 0, same as catR default

    # 2. Selector: How to choose the next item
    if selection == 'MFI':
        selector = MaxInfoSelector() # MFI = Maximum Fisher Information
    else:
        raise ValueError(f"Selection method '{selection}' is not supported.")

    # 3. Estimator: How to estimate ability after each response
    if estimator == 'EAP':
         # Use a normal prior distribution for ability, centered at 0 with std dev of 1
        estimator = EAPEstimator(d=1, prior_dist='norm', prior_params=[0, 1])
    else:
        raise ValueError(f"Estimator '{estimator}' is not supported.")

    # 4. Stopper: When to end the test
    # The test stops when the standard error of the estimate is below min_se
    # or when max_items have been administered.
    stopper = MinErrorStopper(min_se)

    # --- Run the Simulation ---
    print("Initializing simulator...")
    # Create the simulator object
    s = Simulator(item_bank, num_respondents, initializer, selector, estimator, stopper)
    
    print(f"Running simulation with SE < {min_se} or max items = {max_items}...")
    # The 'simulate' method can take all responses at once
    # We pass 'max_items' to the simulate function as an override for the stopper
    # We also pass the pre-computed responses matrix
    all_thetas = s.simulate(max_items=max_items, responses=responses_matrix)

    # --- Save Results ---
    print(f"Simulation complete. Saving results to {output_path}...")
    results_df = pd.DataFrame({'theta': all_thetas})
    results_df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python implementation of catR CAT simulation.")
    parser.add_argument('--input', type=str, required=True, help="Path to input data file (CSV).")
    parser.add_argument('--output', type=str, required=True, help="Path to output file.")
    parser.add_argument('--selection', type=str, default='MFI', help="Item selection criteria (e.g., 'MFI').")
    parser.add_argument('--estimator', type=str, default='EAP', help="Ability estimator (e.g., 'EAP').")
    parser.add_argument('--min_se', type=float, default=0.3, help="Stopping criterion: minimum standard error.")
    parser.add_argument('--max_items', type=int, default=50, help="Stopping criterion: maximum number of items.")
    
    args = parser.parse_args()

    run_cat_simulation(
        input_path=args.input,
        output_path=args.output,
        selection=args.selection,
        estimator=args.estimator,
        min_se=args.min_se,
        max_items=args.max_items
    )