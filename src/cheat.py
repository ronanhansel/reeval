from argparse import ArgumentParser
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from numpyro.diagnostics import hpdi
import torch
import torch.optim as optim
from utils import item_response_fn_1PL_cheat, set_seed
from synthetic_testtaker import CheatingTestTaker
import matplotlib.pyplot as plt

def model(Z, asked_question_list, asked_answer_list, contamination):
    theta_hat_true = numpyro.sample("theta_hat_true", dist.Normal(0.0, 1.0)) # prior_true
    theta_hat_cheat = numpyro.sample("theta_hat_cheat", dist.Normal(0.0, 1.0)) # prior_cheat
    Z_asked = Z[asked_question_list]
    contamination_asked = contamination[asked_question_list]
    probs = item_response_fn_1PL_cheat(
        Z_asked, contamination_asked, theta_hat_true, theta_hat_cheat, datatype="jnp"
        )
    numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_answer_list)

def fit_cheat_theta_mcmc(
    Z, asked_question_list, asked_answer_list, contamination, num_samples=9000, num_warmup=1000
    ):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_,
        Z=Z,
        asked_question_list=asked_question_list,
        asked_answer_list=asked_answer_list,
        contamination=contamination,
    )
    mcmc.print_summary()
    
    theta_true_samples = mcmc.get_samples()["theta_hat_true"]
    mean_theta_true = jnp.mean(theta_true_samples)
    std_theta_true = jnp.std(theta_true_samples)
    
    theta_cheat_samples = mcmc.get_samples()["theta_hat_cheat"]
    mean_theta_cheat = jnp.mean(theta_cheat_samples)
    std_theta_cheat = jnp.std(theta_cheat_samples)
    
    return mean_theta_true, std_theta_true, theta_true_samples, \
        mean_theta_cheat, std_theta_cheat, theta_cheat_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--question_num", type=int, default=1000)
    parser.add_argument("--contamination_percent", type=float, default=0.8)
    parser.add_argument("--testtaker_true_theta", type=float, default=0)
    parser.add_argument("--testtaker_cheat_gain", type=float, default=1.5)
    args = parser.parse_args()

    set_seed(args.seed)

    z3 = torch.normal(mean=0.0, std=1.0, size=(args.question_num,))
    num_ones = int(args.question_num * args.contamination_percent)
    contamination = torch.cat([
        torch.ones(num_ones), 
        torch.zeros(args.question_num - num_ones)
        ])
    contamination = contamination[torch.randperm(args.question_num)]
    
    testtaker1 = CheatingTestTaker(
        true_theta=args.testtaker_true_theta, cheat_gain=args.testtaker_cheat_gain, model="1PL"
        )
    true_theta, cheat_theta = testtaker1.get_ability()
    print(f"True theta: {true_theta}")
    print(f"Cheat theta: {cheat_theta}")

    asked_question_list = list(range(args.question_num))
    asked_answer_list = []
    for i in range(args.question_num):
        asked_answer_list.append(testtaker1.ask(z3, contamination, i))
        
    # MCMC
    contamination = jnp.array(contamination)
    z3 = jnp.array(z3)
    
    asked_question_list = jnp.array(asked_question_list)
    asked_answer_list = jnp.array(asked_answer_list)
    
    mean_theta_true, std_theta_true, theta_true_samples, \
        mean_theta_cheat, std_theta_cheat, theta_cheat_samples \
            = fit_cheat_theta_mcmc(
                z3, asked_question_list, asked_answer_list, contamination
                )

    print(f"mean_theta_true: {mean_theta_true}")
    print(f"std_theta_true: {std_theta_true}")
    print(f"mean_theta_cheat: {mean_theta_cheat}")
    print(f"std_theta_cheat: {std_theta_cheat}")
    
    # CTT
    probs = item_response_fn_1PL_cheat(
        z3, contamination, true_theta, cheat_theta, datatype="jnp"
        )
    response_list = []
    for prob in probs:
        prob_tensor = torch.tensor(prob.tolist())
        bernoulli = torch.distributions.Bernoulli(prob_tensor)
        response = bernoulli.sample()
        response_list.append(response)
    response_1_std = torch.std(torch.stack(response_list))
    response_1_mean = sum(response_list) / len(response_list)
    print(f"CTT theta_1 mean: {response_1_mean}")
    print(f"CTT theta_1 std: {response_1_std}")
    
    plt.figure(figsize=(12, 6))

    # Plotting samples for Test Taker 1
    plt.subplot(1, 2, 1)
    plt.hist(theta_true_samples, bins=30, density=True, alpha=0.6, color='blue', label='Theta True')
    plt.hist(theta_cheat_samples, bins=30, density=True, alpha=0.6, color='red', label='Theta Cheat')
    plt.title('Histogram of Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig('../plot/synthetic/cheat.png')