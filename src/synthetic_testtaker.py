import os
import pandas as pd
import torch
from utils import item_response_fn_3PL, item_response_fn_2PL, item_response_fn_1PL
import numpy as np
from scipy.stats import beta, lognorm, norm
import jax

class SimulatedTestTaker():
    def __init__(self, theta=None, model="3PL"):
        self.model = model
        if theta==None:
            self.ability = torch.normal(mean=0.0, std=1.0, size=(1,))
        else:
            self.ability = theta

    def ask(self, Z, question_index):
        if self.model == "3PL":
            prob = item_response_fn_3PL(
                *Z[question_index, :], self.ability
            )
        elif self.model == "2PL":
            prob = item_response_fn_2PL(
                *Z[question_index, :], self.ability
            )
        elif self.model == "1PL":
            prob = item_response_fn_1PL(
                Z[question_index], self.ability
            )
        return torch.distributions.Bernoulli(prob).sample()
    
    def get_ability(self):
        return self.ability

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    question_num = 500
    testtaker_num = 1000
    
    # a = 0.4548891252373536
    # b = 3.863851492349928
    # loc = 1.8183451675005396e-05
    # scale = 0.851578149836764
    # z1 = beta.rvs(a, b, loc=loc, scale=scale, size=question_num)

    # shape = 0.21631136362126255
    # loc = -0.3077433473021994
    # scale = 1.498526380109833
    # z2 = lognorm.rvs(shape, loc, scale, size=question_num)
    
    # loc = -0.44651776605570387
    # scale = 1.1461954060033157
    # z3 = norm.rvs(loc=loc, scale=scale, size=question_num)
    
    # Z = torch.vstack((
    #     torch.tensor(z1, dtype=torch.float32),
    #     torch.tensor(z2, dtype=torch.float32),
    #     torch.tensor(z3, dtype=torch.float32)
    #     )).T  # (n_questions, 3)
    
    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    
    response_matrix = np.zeros((testtaker_num, question_num))
    true_theta = np.zeros(testtaker_num)
    for i in range(testtaker_num):
        new_testtaker = SimulatedTestTaker(model="1PL")
        true_theta[i] = new_testtaker.get_ability().item()
        
        for j in range(question_num):
            response_matrix[i, j] = new_testtaker.ask(z3, j).item()
            
    save_dir = "../data/synthetic/response_matrix/"
    os.makedirs(save_dir, exist_ok=True)

    response_df = pd.DataFrame(response_matrix.astype(int))
    response_df.insert(0, '', [f'testtaker_{i}' for i in range(testtaker_num)])
    response_df.to_csv(os.path.join(save_dir, "synthetic_matrix.csv"), index=False)

    true_theta_df = pd.DataFrame(true_theta, columns=["true_theta"])
    true_theta_df.to_csv(os.path.join(save_dir, "true_theta.csv"), index=False)
    
    Z_df = pd.DataFrame(z3.numpy(), columns=["z3"])
    Z_df.to_csv(os.path.join(save_dir, "true_Z.csv"), index=False)
    