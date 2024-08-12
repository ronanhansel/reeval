import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from utils import item_response_fn_1PL
from synthetic_testtaker import SimulatedTestTaker
from numpyro.diagnostics import hpdi

def model(Z, asked_question_list, asked_answer_list):
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0)) # prior
    
    Z_asked = Z[asked_question_list]
    probs = item_response_fn_1PL(Z_asked, theta_hat, datatype="jnp")
    
    numpyro.sample("obs", dist.Bernoulli(probs), obs=asked_answer_list)

def fit_theta_with_mcmc(Z, asked_question_list, asked_answer_list, num_samples=2000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_, Z=Z, asked_question_list=asked_question_list, asked_answer_list=asked_answer_list
        )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    mean_theta = jnp.mean(theta_samples)
    variance_theta = jnp.var(theta_samples)
    hpdi_theta = hpdi(theta_samples, 0.9)

    return mean_theta, variance_theta, hpdi_theta

if __name__ == "__main__":
    random_key = random.PRNGKey(42)
    question_num = 10000

    z3 = random.normal(random_key, shape=(question_num,))
    new_testtaker = SimulatedTestTaker(z3, model="1PL", datatype="jnp")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    asked_question_list = list(range(question_num))
    asked_answer_list = []
    for i in range(question_num):
        asked_answer_list.append(new_testtaker.ask(i))
    asked_question_list = jnp.array(asked_question_list)
    asked_answer_list = jnp.array(asked_answer_list)

    mean_theta, variance_theta, hpdi_theta = fit_theta_with_mcmc(z3, asked_question_list, asked_answer_list)

    print(f"Estimated theta mean: {mean_theta}")
    print(f"Estimated theta variance: {variance_theta}")
