import torch
import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import jax.numpy as jnp

# overleaf/R_library: z1/g, z2/a1, z3/d
def item_response_fn_3PL(z1, z2, z3, theta):
    return z1 + (1 - z1) / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_2PL(z2, z3, theta):
    return 1 / (1 + torch.exp(-(z2 * theta + z3)))

def item_response_fn_1PL(z3, theta, datatype="torch"):
    if datatype == "torch":
        return 1 / (1 + torch.exp(-(theta + z3)))
    elif datatype == "numpy":
        return 1 / (1 + np.exp(-(theta + z3)))
    elif datatype == "jnp":
        return 1 / (1 + jnp.exp(-(theta + z3)))
    else:
        raise ValueError("datatype should be 'torch' or 'numpy' or 'jnp'")
    
def item_response_fn_1PL_cheat(z3, contamination, theta_true, theta_cheat, datatype="torch"):
    # z3, contamination: vector/scalar; theta_true, theta_cheat: scalar
    bool_cheat = (theta_true < theta_cheat)
    
    if datatype == "torch":
        return item_response_fn_1PL(z3, theta_true)**(1-bool_cheat) \
            * ((1-contamination) * item_response_fn_1PL(z3, theta_true) \
            + contamination * item_response_fn_1PL(z3, theta_cheat))**bool_cheat
    elif datatype == "jnp":
        return item_response_fn_1PL(z3, theta_true, datatype="jnp")**(1-bool_cheat) \
            * ((1-contamination) * item_response_fn_1PL(z3, theta_true, datatype="jnp") \
            + contamination * item_response_fn_1PL(z3, theta_cheat, datatype="jnp"))**bool_cheat        

def calculate_1d_wasserstein_distance(vector1, vector2):
    return wasserstein_distance(vector1, vector2)

# 3D Wasserstein Distance
def calculate_3d_wasserstein_distance(matrix1, matrix2):
    return wasserstein_distance_nd(matrix1, matrix2)

if __name__ == "__main__":
    print(calculate_1d_wasserstein_distance([0, 1, 3], [5, 6, 8]))
    print(calculate_1d_wasserstein_distance([1, 1, 3], [5, 6, 8]))
    
    mean_1 = 1
    mean_2 = 3
    size = 10000
    diff = mean_2 - mean_1
    
    samples_1 = np.random.normal(loc=mean_1, scale=1, size=size)
    samples_2 = np.random.normal(loc=mean_2, scale=1, size=size)
    
    wd = calculate_1d_wasserstein_distance(samples_1, samples_2)
    print(f'Wasserstein Distance: {wd}')
    print(f'error rate: {(wd - diff)/diff}')
    