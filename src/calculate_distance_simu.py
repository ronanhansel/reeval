import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance import *
import numpy as np
from scipy.stats import beta, lognorm, norm
from utils import item_response_fn_3PL
import torch
import matplotlib.pyplot as plt

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

def generate_perturb(Z, epsilon, n_samples, sigma=0.1):
    Z_prime = Z + norm.rvs(scale=sigma, size=Z.shape)
    current_distance = calculate_3d_wasserstein_distance(Z, Z_prime, n_samples)
    
    while abs(current_distance - epsilon) > 1e-2:
        scaling_factor = epsilon / current_distance
        Z_prime = Z + scaling_factor * (Z_prime - Z)
        current_distance = calculate_3d_wasserstein_distance(Z, Z_prime, n_samples)
        
    return Z_prime, current_distance

n_questions = 1000
n_testtakers = 1000
Z_star = gen_random_para_matrix(n_questions)
theta_star = np.random.randn(n_testtakers)
Y_star = get_response_list(Z_star, theta_star)

epsilons = [i * 0.001 for i in range(1, 11)]
# epsilons = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
Z1s = []
Z2s = []
d_Z1s = []
d_Z2s = []
d_Y1s = []
d_Y2s = []
with open("./synthetic/Wasserstein_Distance.txt", "w", encoding="utf-8") as f:
    for epsilon in epsilons:
        f.write(f"\n\n\nepsilon = {epsilon}")
        print(f"processing epsilon = {epsilon}")

        Z1, d_Z1 = generate_perturb(Z_star, epsilon, n_questions)
        Y1 = get_response_list(Z1, theta_star)
        d_Y1 = calculate_1d_wasserstein_distance(Y_star, Y1)
        f.write(f"Distance between Z* and Z1: {d_Z1}")
        f.write(f"Distance between Y* and Y1: {d_Y1}")
        Z1s.append(Z1)
        d_Z1s.append(d_Z1)
        d_Y1s.append(d_Y1)

        Z2, d_Z2 = generate_perturb(Z_star, epsilon, n_questions)
        Y2 = get_response_list(Z2, theta_star)
        d_Y2 = calculate_1d_wasserstein_distance(Y_star, Y2)
        f.write(f"Distance between Z* and Z2: {d_Z2}")
        f.write(f"Distance between Y* and Y2: {d_Y2}")
        Z2s.append(Z2)
        d_Z2s.append(d_Z2)
        d_Y2s.append(d_Y2)

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(d_Z1s, d_Y1s, marker='o', linestyle='-', label='Y1s')
ax.plot(d_Z2s, d_Y2s, marker='o', linestyle='-', label='Y2s')
ax.set_title('Line Plot of d_Y1s and d_Y2s vs d_Z1s and d_Z2s')
ax.set_xlabel('d_Z')
ax.set_ylabel('Values of d_Y1s and d_Y2s')
ax.legend()
ax.grid(True)
plt.show()

# plot_scatter(Z_star, Z1s[0], Z2s[0])