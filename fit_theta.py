import torch
import torch.optim as optim
from utils import item_response_fn_3PL
from simulator import SimulatedTestTaker

def fit_theta(Z, asked_question_list, asked_answer_list, epoch=200):
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(1,), requires_grad=True)
    optimizer = optim.Adam([theta_hat], lr=0.01)
    for _ in range(epoch):
        log_prob = 0
        for i, asked_question_index in enumerate(asked_question_list):
            prob = item_response_fn_3PL(*Z[:, asked_question_index], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            log_prob = log_prob + bernoulli.log_prob(asked_answer_list[i].float())
        (-log_prob).backward()
        optimizer.step()
        optimizer.zero_grad()
    return theta_hat

if __name__ == "__main__":
    torch.manual_seed(30)
    sample_size = 1000

    a = torch.distributions.Uniform(0, 2).sample((sample_size,))
    b = torch.distributions.Uniform(-3, 3).sample((sample_size,))
    c = torch.distributions.Uniform(0, 1).sample((sample_size,))
    Z = torch.stack([a,b,c])

    new_testtaker = SimulatedTestTaker(Z)
    theta_star = new_testtaker.get_ability()
    print(theta_star)

    asked_answer_list = []
    for i in range(sample_size):
        asked_answer_list.append(new_testtaker.ask(i))

    asked_question_list = list(range(sample_size))

    theta_hat = fit_theta(Z, asked_question_list, asked_answer_list, epoch=100)

    if abs(theta_hat - theta_star) > 0.1:
        print("Test failed")
    else:
        print("Test passed")