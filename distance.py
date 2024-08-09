import numpy as np
from scipy.stats import wasserstein_distance
import ot
import scipy as sp
import matplotlib.pyplot as plt

def calculate_1d_wasserstein_distance(vector1, vector2, bins=30):
   
    hist1, bin_edges1 = np.histogram(vector1, bins=bins, density=True)
    hist2, bin_edges2 = np.histogram(vector2, bins=bins, density=True)

    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2

    distance_1_2 = wasserstein_distance(bin_centers1, bin_centers2, u_weights=hist1, v_weights=hist2)

    return distance_1_2

# 3D Wasserstein Distance
# https://pythonot.github.io/auto_examples/gromov/plot_gromov.html
def calculate_3d_wasserstein_distance(matrix1, matrix2, n_samples):
    """
    Calculates the 3D Wasserstein distance between two matrices.

    Parameters:
    - matrix1 (numpy.ndarray): The first matrix.
    - matrix2 (numpy.ndarray): The second matrix.
    - n_samples (int): The number of samples.

    Returns:
    - str: The calculated Wasserstein distance as a string.
    """
    C1 = sp.spatial.distance.cdist(matrix1, matrix1)
    C2 = sp.spatial.distance.cdist(matrix2, matrix2)

    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    # Conditional Gradient algorithm
    _, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, "square_loss", verbose=True, log=True
    )

    return log0["gw_dist"]

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
    print(calculate_1d_wasserstein_distance([1, 2, 3], [4, 5, 6]))
    # print(wd(np.array([1, 2, 3]), np.array([4, 5, 6])))
    