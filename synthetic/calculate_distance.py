import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance import *
import numpy as np
from scipy.stats import beta, lognorm, norm
from utils import item_response_fn_3PL
import torch

def gen_random_para_matrix(sample_size, distri_diff = 0):
    a = 0.4548891252373536
    b = 3.863851492349928
    loc = 1.8183451675005396e-05
    scale = 0.851578149836764
    z1 = beta.rvs(a+distri_diff, b+distri_diff, loc=loc+distri_diff, scale=scale+distri_diff, size=sample_size)

    shape = 0.21631136362126255
    loc = -0.3077433473021994
    scale = 1.498526380109833

    z2 = lognorm.rvs(shape+distri_diff, loc+distri_diff, scale+distri_diff, size=sample_size)
    
    loc = -0.44651776605570387
    scale = 1.1461954060033157
    z3 = norm.rvs(loc=loc+distri_diff, scale=scale+distri_diff, size=sample_size)
    matrix = np.vstack((z1, z2, z3)).T
    return matrix # (n_questions, 3)

def get_response_list(Z, theta_list):
    n_testtakers = len(theta_list)
    n_questions = Z.shape[0]
    psolve = np.zeros((n_testtakers, n_questions))

    Z_tensor = torch.tensor(Z)
    theta_tensor = torch.tensor(theta_list)

    for i in range(n_testtakers):
        for j in range(n_questions):
            psolve[i, j] = item_response_fn_3PL(*Z_tensor[j], theta_tensor[i]).item()
    response_matrix = (np.random.rand(n_testtakers, n_questions) < psolve).astype(int)
    return response_matrix.tolist()

n_questions = 1000
n_testtakers = 1000
Z_stars = gen_random_para_matrix(n_questions)
theta_stars = np.random.randn(n_testtakers)
Y_stars = get_response_list(Z_stars, theta_stars)

# epsilons = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
# distri_diffs = [0.1]
epsilons = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
distri_diffs = np.arange(0.1, 1.1, 0.1)
with open("./synthetic/Wasserstein_Distance.txt", "w", encoding="utf-8") as f:
    for epsilon in epsilons:
        f.write(f"\n\n\nepsilon = {epsilon}")
        print(f"epsilon = {epsilon}")
        Zs = []
        Ys = []
        for distri_diff in distri_diffs:
            f.write(f"\ndistri_diff = {distri_diff}")
            print(f"distri_diff = {distri_diff}")
            for i in range(2): 
                flag = False           
                Z = gen_random_para_matrix(n_questions, distri_diff=distri_diff)
                d_Z = calculate_3d_wasserstein_distance(Z_stars, Z, n_questions)

                if abs(float(d_Z) - epsilon) < 0.01:
                    flag = True
                    f.write(f"\nSucceed with Z{i} at epsilon = {epsilon}, distri_diff = {distri_diff}")
                    break

                else:
                    flag = False
                    f.write(f"\nFail with Z{i} at epsilon = {epsilon}, distri_diff = {distri_diff}")
                    break

            if flag:
                Zs.append(Z)
                Ys.append(get_response_list(Z, theta_stars))
                d_Y = calculate_1d_wasserstein_distance(Y_stars, Ys[-1])

                f.write(f"\n\nDistance between Z* and Z_{i}: {float(d_Z)}")
                f.write(f"\n\nDistance between Y* and Y_{i}: {float(d_Y)}")

# plot_scatter(Z_stars, *Zs)
