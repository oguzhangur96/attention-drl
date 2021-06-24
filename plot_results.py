import os
import numpy as np
import pandas as pd
import glob
from plotter import Plotter

log_dir = "COMPLEX_MOV_RESULTS/"
df_dict = {
    "DRQN": [pd.DataFrame({"episodes": range(len(np.load(f"{log_dir}eval_Mario_DRQN.npy"))), "reward": np.load(f"{log_dir}eval_Mario_DRQN.npy")})],
    "DARQN": [pd.DataFrame({"episodes": range(len(np.load(f"{log_dir}eval_Mario_DARQN.npy"))),
                           "reward": np.load(f"{log_dir}eval_Mario_DARQN.npy")})],
    "DCBAMRQN": [pd.DataFrame({"episodes": range(len(np.load(f"{log_dir}eval_Mario_DCBAMRQN.npy"))),
                           "reward": np.load(f"{log_dir}eval_Mario_DCBAMRQN.npy")})],
}

print(df_dict["DCBAMRQN"][0])

pltter = Plotter(df_dict)
pltter()