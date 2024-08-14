import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from utils import item_response_fn_1PL
from numpyro.diagnostics import hpdi
import torch
import torch.optim as optim
from utils import item_response_fn_1PL_cheat
from synthetic_testtaker import CheatingTestTaker

def model(Z, asked_question_list, asked_answer_list, contamination):
    theta_hat_true = numpyro.sample("theta_hat_true", dist.Normal(0.0, 1.0)) # prior_true
    theta_hat_cheat = numpyro.sample("theta_hat_cheat", dist.Normal(0.0, 1.0)) # prior_cheat
    Z_asked = Z[asked_question_list]
    probs = item_response_fn_1PL_cheat(
        Z_asked, contamination, theta_hat_true, theta_hat_cheat, datatype="jnp"
        )
    numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_answer_list)
    
def fit_theta_mcmc(
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
    
    return mean_theta_true, std_theta_true, mean_theta_cheat, std_theta_cheat

if __name__ == "__main__":
    torch.manual_seed(10)
    question_num = 1000

    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    new_testtaker = CheatingTestTaker(true_theta=0.5, cheat_gain=1, model="1PL")
    true_theta, cheat_theta = new_testtaker.get_ability()
    print(f"True theta: {true_theta}")
    print(f"Cheat theta: {cheat_theta}")

    contamination_percent = 0.99
    num_ones = int(question_num * contamination_percent)
    contamination = torch.cat([
        torch.ones(num_ones), 
        torch.zeros(question_num - num_ones)
        ])
    contamination = contamination[torch.randperm(question_num)]

    asked_question_list = list(range(question_num))
    asked_answer_list = []
    for i in range(question_num):
        asked_answer_list.append(new_testtaker.ask(z3, contamination, i))

    # MCMC
    asked_question_list = jnp.array(asked_question_list)
    asked_answer_list = jnp.array(asked_answer_list)
    contamination = jnp.array(contamination)
    z3 = jnp.array(z3)

    mean_theta_true, std_theta_true, mean_theta_cheat, std_theta_cheat = fit_theta_mcmc(
        z3, asked_question_list, asked_answer_list, contamination
        )
    print(f"mean_theta_true: {mean_theta_true}")
    print(f"std_theta_true: {std_theta_true}")
    print(f"mean_theta_cheat: {mean_theta_cheat}")
    print(f"std_theta_cheat: {std_theta_cheat}")
    
    
   
    
