import os
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
log_dir = "Train_Historys/"

drqn_rewards = np.load(f"{log_dir}Breakout_DRQN.npy")
darqn_rewards = np.load(f"{log_dir}Breakout_DARQN.npy")

max_len = max(drqn_rewards.size,darqn_rewards.size)

plt.plot(range(drqn_rewards.size), drqn_rewards, label = "DRQN Rewards")
plt.plot(range(darqn_rewards.size), darqn_rewards, label = "DARQN Rewards")
plt.legend()
plt.show()

