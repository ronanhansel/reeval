import numpy as np
from scipy.stats import wasserstein_distance
import ot
import scipy as sp


def calculate_1d_wasserstein_distance(vector1, vector2, vector3, bins=30):
    """
    Calculate the 1D-Wasserstein distance between three vectors.

    Parameters:
    vector1 (array-like): The first vector.
    vector2 (array-like): The second vector.
    vector3 (array-like): The third vector.
    bins (int, optional): The number of bins for histogram calculation.
        Default is 30.

    Returns:
    tuple: A tuple containing the Wasserstein distances between vector1 and
        vector2, vector1 and vector3, and vector2 and vector3, respectively.
    """
    hist1, bin_edges1 = np.histogram(vector1, bins=bins, density=True)
    hist2, bin_edges2 = np.histogram(vector2, bins=bins, density=True)
    hist3, bin_edges3 = np.histogram(vector3, bins=bins, density=True)

    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
    bin_centers3 = (bin_edges3[:-1] + bin_edges3[1:]) / 2

    distance_1_2 = wasserstein_distance(
        bin_centers1, bin_centers2, u_weights=hist1, v_weights=hist2
    )
    distance_1_3 = wasserstein_distance(
        bin_centers1, bin_centers3, u_weights=hist1, v_weights=hist3
    )
    distance_2_3 = wasserstein_distance(
        bin_centers2, bin_centers3, u_weights=hist2, v_weights=hist3
    )

    return distance_1_2, distance_1_3, distance_2_3

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

    return str(log0["gw_dist"])


# if __name__ == "__main__":
    