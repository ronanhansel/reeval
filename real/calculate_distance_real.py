import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance import calculate_3d_wasserstein_distance, calculate_1d_wasserstein_distance

# 1D Wasserstein Distance
with open("./real/1D_Wasserstein_Distance.txt", "w", encoding="utf-8") as f:
    model_list = ["1PL", "2PL", "3PL"]
    for model in model_list:
        f.write(f"{model}\n")
        base_path = f"model_coef/divided_base_coef_{model}_clean.csv"
        perturb1_path = f"model_coef/divided_perturb1_coef_{model}_clean.csv"
        perturb2_path = f"model_coef/divided_perturb2_coef_{model}_clean.csv"

        para_list =  ['a1', 'd', 'g']
        for para in para_list:
            base_coef = pd.read_csv(base_path, usecols=[para])
            perturb1_coef = pd.read_csv(perturb1_path, usecols=[para])
            perturb2_coef = pd.read_csv(perturb2_path, usecols=[para])

            distance_1_2, distance_1_3, distance_2_3 = calculate_1d_wasserstein_distance(base_coef, perturb1_coef, perturb2_coef)
            f.write(f"Parameter {para}\n")
            f.write(f"Distance between base and perturb1: {distance_1_2}\n")
            f.write(f"Distance between base and perturb2: {distance_1_3}\n")
            f.write(f"Distance between perturb1 and perturb2: {distance_2_3}\n")
            f.write("\n")
            


# 3D Wasserstein Distance
with open("./real/3D_Wasserstein_Distance.txt", "w", encoding="utf-8") as f:
    for model in model_list:
        f.write(f"{model}\n")

        base_path = f"model_coef/divided_base_coef_{model}_clean.csv"
        perturb1_path = f"model_coef/divided_perturb1_coef_{model}_clean.csv"
        perturb2_path = f"model_coef/divided_perturb2_coef_{model}_clean.csv"

        base_matrix = pd.read_csv(base_path, usecols=[0, 1, 2]).values
        perturb1_matrix = pd.read_csv(perturb1_path, usecols=[0, 1, 2]).values
        perturb2_matrix = pd.read_csv(perturb2_path, usecols=[0, 1, 2]).values

        n_samples = base_matrix.shape[0]
        print(n_samples)

        distance_1_2 = calculate_3d_wasserstein_distance(base_matrix, perturb1_matrix, n_samples)
        distance_1_3 = calculate_3d_wasserstein_distance(base_matrix, perturb2_matrix, n_samples)
        distance_2_3 = calculate_3d_wasserstein_distance(perturb1_matrix, perturb2_matrix, n_samples)

        f.write(f"Distance between base_matrix and perturb1_matrix: {distance_1_2}\n")
        f.write(f"Distance between base_matrix and perturb2_matrix: {distance_1_3}\n")
        f.write(f"Distance between perturb1_matrix and perturb2_matrix: {distance_2_3}\n")
        f.write("\n")

