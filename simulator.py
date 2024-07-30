import torch
from utils import item_response_fn_3PL


class SimulatedTestTaker():
    def __init__(self, Z):
        self.ability = torch.normal(mean=0.0, std=1.0, size=(1,))
        self.Z = Z

    def ask(self, question_index):
        prob = item_response_fn_3PL(
            *self.Z[:,question_index], self.ability
        )
        bernoulli = torch.distributions.Bernoulli(prob)
        return bernoulli.sample()
    
    def get_ability(self):
        return self.ability

if __name__ == "__main__":
    
    torch.manual_seed(30)
    sample_size = 1000

    a = torch.distributions.Uniform(0, 3).sample((sample_size,))
    b = torch.distributions.Uniform(-3, 3).sample((sample_size,))
    c = torch.distributions.Uniform(0.49, ).sample((sample_size,))
    Z = torch.stack([a,b,c])

    new_testtaker = SimulatedTestTaker(Z)
    