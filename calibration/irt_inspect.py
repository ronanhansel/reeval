from tqdm import tqdm
import pandas as pd
import torch
from torchmetrics.functional import spearman_corrcoef
from torch.distributions import Bernoulli

torch.manual_seed(0)
data = pd.read_csv("Subset.csv")
data = torch.tensor(data.values, dtype=torch.float32)
n_test_takers, n_items = data.shape

z = torch.tensor(torch.zeros(n_items), requires_grad=True)
optim = torch.optim.Adam([z], lr=0.001)

thetas = torch.randn(500, n_test_takers)
# >>> 500 x n_test_takers

for _ in tqdm(range(1000)):
    optim.zero_grad()
    probs = torch.sigmoid(thetas[:, :, None] + z[None, None, :])
    # >>> 500 x n_test_takers x n_items

    (-Bernoulli(probs=probs).log_prob(data).mean()).backward()
    optim.step()

# compute the spearman correlation
print("Spearman: ", spearman_corrcoef(z, data.mean(0)))
