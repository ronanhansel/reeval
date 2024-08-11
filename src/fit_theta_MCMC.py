import torch
import torch.optim as optim
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from utils import item_response_fn_1PL
from synthetic_testtaker import SimulatedTestTaker

def fit_theta(Z, asked_question_list, asked_answer_list, epoch=300):
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(1,), requires_grad=True)
    optimizer = optim.Adam([theta_hat], lr=0.01)
    for _ in range(epoch):
        log_prob = 0
        for asked_question_index in asked_question_list:
            prob = item_response_fn_1PL(Z[asked_question_index], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            log_prob += bernoulli.log_prob(asked_answer_list[asked_question_index].float())
        
        loss = -log_prob / len(asked_question_list)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return theta_hat.item()

def model(Z, asked_question_list, asked_answer_list, theta_init):
    theta_hat = numpyro.sample("theta_hat", dist.Normal(theta_init, 0.1))  # 使用优化的 theta_init 作为初始值
    
    Z_asked = Z[jnp.array(asked_question_list)]
    probs = item_response_fn_1PL(Z_asked, theta_hat, datatype="jnp")
    
    numpyro.sample(
        "obs", 
        dist.Bernoulli(probs), 
        obs=asked_answer_list.astype(jnp.float32)
    )

def fit_theta_with_mcmc(Z, asked_question_list, asked_answer_list, theta_init, num_samples=2000, num_warmup=1000):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    nuts_kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values={"theta_hat": theta_init}))
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(
        rng_key_, Z=Z, asked_question_list=asked_question_list, asked_answer_list=asked_answer_list, theta_init=theta_init
        )
    mcmc.print_summary()
    
    theta_samples = mcmc.get_samples()["theta_hat"]
    mean_theta = jnp.mean(theta_samples)
    variance_theta = jnp.var(theta_samples)

    return mean_theta, variance_theta

if __name__ == "__main__":
    torch.manual_seed(42)
    random_key = random.PRNGKey(42)
    question_num = 1000

    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    new_testtaker = SimulatedTestTaker(z3, model="1PL", datatype="torch")
    theta_star = new_testtaker.get_ability()
    print(f"True theta: {theta_star}")

    asked_question_list = list(range(question_num))
    
    asked_answer_list = []
    for i in range(question_num):
        asked_answer_list.append(new_testtaker.ask(i))
    asked_answer_list = torch.tensor(asked_answer_list)

    theta_init = fit_theta(z3, asked_question_list, asked_answer_list)
    print(f"Optimized theta: {theta_init}")

    mean_theta, variance_theta = fit_theta_with_mcmc(jnp.array(z3), asked_question_list, jnp.array(asked_answer_list), theta_init=theta_init)

    print(f"Estimated theta mean: {mean_theta}")
    print(f"Estimated theta variance: {variance_theta}")
