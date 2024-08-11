from utils import calculate_1d_wasserstein_distance, calculate_3d_wasserstein_distance
import numpy as np
from scipy.stats import beta, lognorm, norm
from utils import item_response_fn_1PL
import torch
import matplotlib.pyplot as plt

def get_response(Z, theta_list):
    n_testtakers = len(theta_list)
    n_questions = Z.shape[0]
    psolve = np.zeros((n_testtakers, n_questions))

    Z_tensor = torch.tensor(Z)
    theta_tensor = torch.tensor(theta_list)

    for i in range(n_testtakers):
        for j in range(n_questions):
            psolve[i, j] = item_response_fn_1PL(Z_tensor[j], theta_tensor[i]).item()
    response_matrix = (np.random.rand(n_testtakers, n_questions) < psolve).astype(int)
    
    return response_matrix.flatten()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    plt.rcParams.update({'font.size': 20})
    
    question_num = 500
    base_Z = np.random.normal(loc=0, scale=1, size=question_num)
    
    testtaker_num = 1000
    true_theta = np.random.normal(loc=0, scale=1, size=testtaker_num)
    
    base_Y = get_response(base_Z, true_theta)
    
    wd_Z_list = []
    wd_Y_list = []
    for mean in np.arange(1, 15, 0.25):
        perturb_Z = np.random.normal(loc=mean, scale=1, size=question_num)
        wd_Z = calculate_1d_wasserstein_distance(base_Z, perturb_Z)
        wd_Z_list.append(wd_Z)
        
        perturb_Y = get_response(perturb_Z, true_theta)
        wd_Y = calculate_1d_wasserstein_distance(base_Y, perturb_Y)
        wd_Y_list.append(wd_Y)

    plt.figure(figsize=(20, 20))
    plt.plot(wd_Z_list, wd_Y_list, '-o')
    plt.xlabel('Wasserstein Distance between Z')
    plt.ylabel('Wasserstein Distance between Y')
    plt.title('Wasserstein Distance between Z and Y')
    plt.grid()
    plt.savefig('../plot/synthetic/wd_Y_vs_Z.png')