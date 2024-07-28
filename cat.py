import pandas as pd
import numpy as np

def fn_1PL(theta, d):
    return 1 / (1 + np.exp(-(theta + d)))

