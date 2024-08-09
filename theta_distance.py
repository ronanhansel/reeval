import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_theta_df = pd.read_csv('theta/divided_base_theta_1PL.csv', usecols=[1])
base_theta = base_theta_df.iloc[:,0].values

perturb1_theta_df = pd.read_csv('theta/divided_perturb1_theta_1PL.csv', usecols=[1])
perturb1_theta = perturb1_theta_df.iloc[:,0].values

perturb2_theta_df = pd.read_csv('theta/divided_perturb2_theta_1PL.csv', usecols=[1])
perturb2_theta = perturb2_theta_df.iloc[:,0].values

diff1 = base_theta - perturb1_theta
diff2 = base_theta - perturb2_theta
diff3 = perturb1_theta - perturb2_theta

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.hist(diff1, bins=50, density=True, alpha=0.75, color='blue')
plt.title('base_theta - perturb1_theta')

plt.subplot(3, 1, 2)
plt.hist(diff2, bins=50, density=True, alpha=0.75, color='green')
plt.title('base_theta - perturb2_theta')

plt.subplot(3, 1, 3)
plt.hist(diff3, bins=50, density=True, alpha=0.75, color='red')
plt.title('perturb1_theta - perturb2_theta')

plt.tight_layout()
plt.show()

