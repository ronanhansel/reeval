import torch
import torch.optim as optim
from utils import item_response_fn_3PL
from synthetic_testtaker import SimulatedTestTaker

def fit_theta(Z, asked_question_list, asked_answer_list, epoch=500):
    theta_hat = torch.normal(mean=0.0, std=1.0, size=(1,), requires_grad=True)
    optimizer = optim.Adam([theta_hat], lr=0.01)
    for _ in range(epoch):
        log_prob = 0
        for i, asked_question_index in enumerate(asked_question_list):
            prob = item_response_fn_3PL(*Z[asked_question_index, :], theta_hat)
            bernoulli = torch.distributions.Bernoulli(prob)
            print(bernoulli.log_prob(asked_answer_list[i].float()))
            log_prob = log_prob + bernoulli.log_prob(asked_answer_list[i].float())
            
        (-log_prob/len(asked_question_list)).backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(theta_hat)
    return theta_hat

if __name__ == "__main__":
    torch.manual_seed(42)
    question_num = 1000

    z1 = torch.distributions.Beta(0.5, 4).sample((1000,))
    z2 = torch.distributions.LogNormal(loc=0.0, scale=0.25).sample((1000,))
    z3 = torch.normal(mean=0.0, std=1.0, size=(question_num,))
    Z = torch.stack([z1,z2,z3]).T

    new_testtaker = SimulatedTestTaker(Z)
    theta_star = new_testtaker.get_ability()
    print(theta_star)

    asked_question_list = list(range(question_num))
    
    asked_answer_list = []
    for i in range(question_num):
        asked_answer_list.append(new_testtaker.ask(i))

    theta_hat = fit_theta(Z, asked_question_list, asked_answer_list, epoch=500)
