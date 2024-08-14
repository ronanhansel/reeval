import torch
from synthetic_testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp

def inverse_item_response_fn_1PL(y,theta):
    return -theta - torch.log((1 - y) / y)

def beta_params_from_mode(mode, concentration=10):
    alpha = mode * (concentration - 2) + 1
    beta_param = (1 - mode) * (concentration - 2) + 1
    return alpha, beta_param

if __name__ == "__main__":
    torch.manual_seed(10)
    
    Y_bar = 0.7
    theta_1 = 1
    theta_2 = 2
    question_num = 1000

    alpha, beta = beta_params_from_mode(Y_bar)
    beta_dist = torch.distributions.Beta(alpha, beta)
    Y = beta_dist.sample((question_num,))

    Z_1 = inverse_item_response_fn_1PL(Y, theta_1)
    Z_2 = inverse_item_response_fn_1PL(Y, theta_2)

    testtaker1 = SimulatedTestTaker(theta=theta_1, model="1PL")
    testtaker2 = SimulatedTestTaker(theta=theta_2, model="1PL")
    
    asked_question_list = list(range(question_num))
    
    asked_answer_list_1 = []
    for i in range(question_num):
        asked_answer_list_1.append(testtaker1.ask(Z_1, i))
    
    asked_answer_list_2 = []
    for i in range(question_num):
        asked_answer_list_2.append(testtaker2.ask(Z_2, i))
    
    # MCMC
    asked_question_list = jnp.array(asked_question_list)
    
    asked_answer_list_1 = jnp.array(asked_answer_list_1)
    Z_1 = jnp.array(Z_1)
    mean_theta_1, std_theta_1 = fit_theta_mcmc(Z_1, asked_question_list, asked_answer_list_1)
    print(f"mean_theta_1: {mean_theta_1}")
    print(f"std_theta_1: {std_theta_1}")
    
    asked_answer_list_2 = jnp.array(asked_answer_list_2)
    Z_2 = jnp.array(Z_2)
    mean_theta_2, std_theta_2 = fit_theta_mcmc(Z_2, asked_question_list, asked_answer_list_2)
    print(f"mean_theta_2: {mean_theta_2}")
    print(f"std_theta_2: {std_theta_2}")