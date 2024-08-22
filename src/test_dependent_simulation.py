from argparse import ArgumentParser
import torch
from synthetic_testtaker import SimulatedTestTaker
from fit_theta import fit_theta_mcmc
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import item_response_fn_1PL, set_seed

def inverse_item_response_fn_1PL(y,theta):
    return -theta - torch.log((1 - y) / y)

def beta_params_from_mode(mode, concentration=10):
    alpha = mode * (concentration - 2) + 1
    beta_param = (1 - mode) * (concentration - 2) + 1
    return alpha, beta_param

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--Y_bar", type=float, default=0.7)
    parser.add_argument("--theta_1", type=float, default=1)
    parser.add_argument("--theta_2", type=float, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    alpha, beta = beta_params_from_mode(args.Y_bar)
    beta_dist = torch.distributions.Beta(alpha, beta)
    Y = beta_dist.sample((args.question_num,))

    Z_1 = inverse_item_response_fn_1PL(Y, args.theta_1)
    Z_2 = inverse_item_response_fn_1PL(Y, args.theta_2)

    testtaker1 = SimulatedTestTaker(theta=args.theta_1, model="1PL")
    testtaker2 = SimulatedTestTaker(theta=args.theta_2, model="1PL")
    
    asked_question_list = list(range(args.question_num))
    
    asked_answer_list_1 = []
    for i in range(args.question_num):
        asked_answer_list_1.append(testtaker1.ask(Z_1, i))
    
    asked_answer_list_2 = []
    for i in range(args.question_num):
        asked_answer_list_2.append(testtaker2.ask(Z_2, i))
    
    # MCMC
    asked_question_list = jnp.array(asked_question_list)
    
    asked_answer_list_1 = jnp.array(asked_answer_list_1)
    Z_1 = jnp.array(Z_1)
    mean_theta_1, std_theta_1, theta_1_samples = fit_theta_mcmc(Z_1, asked_question_list, asked_answer_list_1)
    print(f"mean_theta_1: {mean_theta_1}")
    print(f"std_theta_1: {std_theta_1}")
    
    asked_answer_list_2 = jnp.array(asked_answer_list_2)
    Z_2 = jnp.array(Z_2)
    mean_theta_2, std_theta_2, theta_2_samples= fit_theta_mcmc(Z_2, asked_question_list, asked_answer_list_2)
    print(f"mean_theta_2: {mean_theta_2}")
    print(f"std_theta_2: {std_theta_2}")
    
    
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(theta_1_samples, bins=30, density=True, alpha=0.6, color='blue', label='theta_1_samples')
    plt.hist(theta_2_samples, bins=30, density=True, alpha=0.6, color='green', label='theta_2_samples')
    plt.title('Histogram of Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(Z_1, bins=30, density=True, alpha=0.6, color='red', label='Z_1')
    plt.hist(Z_2, bins=30, density=True, alpha=0.6, color='orange', label='Z_2')
    plt.title('Histogram of Z')
    plt.xlabel('Z')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig('../plot/synthetic/test_dependent_simulation.png')
    
    
    
    # CTT
    probs = item_response_fn_1PL(Z_1, args.theta_1, datatype="jnp")
    response_1_list = []
    for prob in probs:
        prob_tensor = torch.tensor(prob.tolist())
        bernoulli = torch.distributions.Bernoulli(prob_tensor)
        response_1 = bernoulli.sample()
        response_1_list.append(response_1)
    response_1_mean = sum(response_1_list) / len(response_1_list)
    response_1_std = torch.std(torch.stack(response_1_list))
    print(f"CTT theta_1 mean: {response_1_mean}")
    print(f"CTT theta_1 std: {response_1_std}")
    
    probs = item_response_fn_1PL(Z_2, args.theta_2, datatype="jnp")
    response_2_list = []
    for prob in probs:
        prob_tensor = torch.tensor(prob.tolist())
        bernoulli = torch.distributions.Bernoulli(prob_tensor)
        response_2 = bernoulli.sample()
        response_2_list.append(response_2)
    response_2_mean = sum(response_2_list) / len(response_2_list)
    response_2_std = torch.std(torch.stack(response_2_list))
    print(f"CTT theta_2 mean: {response_2_mean}")
    print(f"CTT theta_2 std: {response_2_std}")