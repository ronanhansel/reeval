import torch
import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import ot
import scipy as sp
import matplotlib.pyplot as plt

# overleaf/R_library: z1/g, z2/a1, z3/d
def item_response_fn_3PL(z1, z2, z3, theta):
    return z1 + (1 - z1) / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_2PL(z2, z3, theta):
    return 1 / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_1PL(z3, theta):
    return 1 / (1 + torch.exp(-(theta + z3)))


def calculate_1d_wasserstein_distance(vector1, vector2):
    return wasserstein_distance(vector1, vector2)

# 3D Wasserstein Distance
def calculate_3d_wasserstein_distance(matrix1, matrix2):
    return wasserstein_distance_nd(matrix1, matrix2)

def plot_scatter(base_coef, perturb1_coef, perturb2_coef, axis_lim = False):
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    print(base_coef[:, 0])
    ax.scatter(base_coef[:, 0], base_coef[:, 1], base_coef[:, 2], c='r', label='Base Coef')
    ax.scatter(perturb1_coef[:, 0], perturb1_coef[:, 1], perturb1_coef[:, 2], c='g', label='Perturb1 Coef')
    ax.scatter(perturb2_coef[:, 0], perturb2_coef[:, 1], perturb2_coef[:, 2], c='b', label='Perturb2 Coef')

    ax.set_xlabel('a1')
    ax.set_ylabel('d')
    ax.set_zlabel('g')

    if axis_lim:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    print(calculate_1d_wasserstein_distance([0, 1, 3], [5, 6, 8]))
    print(calculate_1d_wasserstein_distance([1, 1, 3], [5, 6, 8]))
    