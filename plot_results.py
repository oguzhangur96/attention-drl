import os
import numpy as np
import pandas as pd
import glob
from plotter import Plotter

log_dir = "COMPLEX_MOV_RESULTS/*"
df_dict = {
    "DRQN": [pd.DataFrame({"episodes": range(len(np.load(file))), "reward": np.load(file)}) for file in glob.glob(log_dir) if file.endswith("eval_Mario_DRQN.npy")],
    "DARQN": [pd.DataFrame({"episodes": range(len(np.load(file))), "reward": np.load(file)}) for file in glob.glob(log_dir) if file.endswith("eval_Mario_DARQN.npy")],
    "DCBAMRQN": [pd.DataFrame({"episodes": range(len(np.load(file))), "reward": np.load(file)}) for file in glob.glob(log_dir) if file.endswith("eval_Mario_DCBAMRQN.npy")]
}

pltter = Plotter(df_dict)
pltter()