import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from utils import item_response_fn_1PL
from synthetic_testtaker import SimulatedTestTaker

def model(Z, asked_question_list, asked_answer_list):
    theta_hat = numpyro.sample("theta_hat", dist.Normal(0.0, 1.0)) # prior
    for asked_question_index in asked_question_list:
        prob = item_response_fn_1PL(Z[asked_question_index], theta_hat, datatype="jnp")
        numpyro.sample(
            f"obs_{asked_question_index}", 
            dist.Bernoulli(prob), 
            obs=asked_answer_list[asked_question_index].astype(jnp.float32)
            )

def fit_theta_with_mcmc(Z, asked_question_list, asked_answer_list, num_samples=2000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_, Z=Z, asked_question_list=asked_question_list, asked_answer_list=asked_answer_list
        )
    mcmc.print_summary()
    samples_1 = mcmc.get_samples()
    
    # jnp.expand_dims(?, -1): ?.shape = (n_samples,) -> (n_samples, 1)
    posterior_mu = (
        jnp.expand_dims(samples_1, -1)
        + jnp.expand_dims(samples_1["bM"], -1) * dset.MarriageScaled.values
    )

    mean_mu = jnp.mean(posterior_mu, axis=0)
    hpdi_mu = hpdi(posterior_mu, 0.9)
        
    return mean_theta, variance_theta

if __name__ == "__main__":
    random_key = random.PRNGKey(42)
    question_num = 2

    z3 = random.normal(random_key, shape=(question_num,))
    new_testtaker = SimulatedTestTaker(z3, model="1PL", datatype="jnp")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    asked_question_list = list(range(question_num))
    
    asked_answer_list = []
    for i in range(question_num):
        asked_answer_list.append(new_testtaker.ask(i))
    asked_answer_list = jnp.array(asked_answer_list)

    mean_theta, variance_theta = fit_theta_with_mcmc(z3, asked_question_list, asked_answer_list)

    print(f"Estimated theta mean: {mean_theta}")
    print(f"Estimated theta variance: {variance_theta}")
