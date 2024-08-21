import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from numpyro.diagnostics import hpdi
import torch
import torch.optim as optim
from utils import item_response_fn_1PL_cheat
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
    torch.manual_seed(10)
    question_num = 1000

    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    contamination_percent = 0.8
    num_ones = int(question_num * contamination_percent)
    contamination = torch.cat([
        torch.ones(num_ones), 
        torch.zeros(question_num - num_ones)
        ])
    contamination = contamination[torch.randperm(question_num)]
    
    testtaker1 = CheatingTestTaker(true_theta=0.5, cheat_gain=1, model="1PL")
    true_theta_1, cheat_theta_1 = testtaker1.get_ability()
    print(f"True theta 1: {true_theta_1}")
    print(f"Cheat theta 1: {cheat_theta_1}")

    asked_question_list_1 = list(range(question_num))
    asked_answer_list_1 = []
    for i in range(question_num):
        asked_answer_list_1.append(testtaker1.ask(z3, contamination, i))
        
    testtaker2 = CheatingTestTaker(true_theta=1.5, cheat_gain=0, model="1PL")
    true_theta_2, cheat_theta_2= testtaker2.get_ability()
    print(f"True theta 2: {true_theta_2}")
    print(f"Cheat theta 2: {cheat_theta_2}")

    asked_question_list_2 = list(range(question_num))
    asked_answer_list_2 = []
    for i in range(question_num):
        asked_answer_list_2.append(testtaker2.ask(z3, contamination, i))

    # MCMC
    contamination = jnp.array(contamination)
    z3 = jnp.array(z3)
    
    asked_question_list_1 = jnp.array(asked_question_list_1)
    asked_answer_list_1 = jnp.array(asked_answer_list_1)

    mean_theta_true_1, std_theta_true_1, theta_true_samples_1, \
        mean_theta_cheat_1, std_theta_cheat_1, theta_cheat_samples_1 \
            = fit_cheat_theta_mcmc(
                z3, asked_question_list_1, asked_answer_list_1, contamination
                )
    
    print(f"mean_theta_true_1: {mean_theta_true_1}")
    print(f"std_theta_true_1: {std_theta_true_1}")
    print(f"mean_theta_cheat_1: {mean_theta_cheat_1}")
    print(f"std_theta_cheat_1: {std_theta_cheat_1}")
    
    asked_question_list_2 = jnp.array(asked_question_list_2)
    asked_answer_list_2 = jnp.array(asked_answer_list_2)
    mean_theta_true_2, std_theta_true_2, theta_true_samples_2, \
        mean_theta_cheat_2, std_theta_cheat_2, theta_cheat_samples_2 \
            = fit_cheat_theta_mcmc(
                z3, asked_question_list_2, asked_answer_list_2, contamination
                )
    
    print(f"mean_theta_true_2: {mean_theta_true_2}")
    print(f"std_theta_true_2: {std_theta_true_2}")
    print(f"mean_theta_cheat_2: {mean_theta_cheat_2}")
    print(f"std_theta_cheat_2: {std_theta_cheat_2}")
    
    # CTT
    probs = item_response_fn_1PL_cheat(
        z3, contamination, true_theta_1, cheat_theta_1, datatype="jnp"
        )
    response_1_list = []
    for prob in probs:
        prob_tensor = torch.tensor(prob.tolist())
        bernoulli = torch.distributions.Bernoulli(prob_tensor)
        response_1 = bernoulli.sample()
        response_1_list.append(response_1)
    response_1_std = torch.std(torch.stack(response_1_list))
    response_1_mean = sum(response_1_list) / len(response_1_list)
    print(f"CTT theta_1 mean: {response_1_mean}")
    print(f"CTT theta_1 std: {response_1_std}")
    
    probs = item_response_fn_1PL_cheat(
        z3, contamination, true_theta_2, cheat_theta_2, datatype="jnp"
        )
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

    plt.figure(figsize=(12, 6))

    # Plotting theta_hat_true samples
    plt.subplot(1, 2, 1)
    plt.hist(theta_true_samples_1, bins=30, density=True, alpha=0.6, color='blue', label='Theta True - Test Taker 1')
    plt.hist(theta_true_samples_2, bins=30, density=True, alpha=0.6, color='green', label='Theta True - Test Taker 2')
    plt.title('Histogram of True Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()

    # Plotting theta_hat_cheat samples
    plt.subplot(1, 2, 2)
    plt.hist(theta_cheat_samples_1, bins=30, density=True, alpha=0.6, color='blue', label='Theta Cheat - Test Taker 1')
    plt.hist(theta_cheat_samples_2, bins=30, density=True, alpha=0.6, color='green', label='Theta Cheat - Test Taker 2')
    plt.title('Histogram of Cheat Theta Samples')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig('../plot/synthetic/cheat.png')