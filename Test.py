import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from Environment import CreateBreakout
from Network import ActorCriticNet, QNet, QNet_LSTM
from Train_DRQN import init_hidden

import time
import numpy as np

model_path = './Models/Breakout_PPO.model'

def test():
    env = CreateBreakout(stack=True) # False if DRQN

    Net = ActorCriticNet() # PPO, A3C
    #Net = QNet() # DQN
    #Net = QNet_LSTM() # DRQN
    
    Net.load_state_dict(torch.load(model_path))

    score = 0
    state = env.reset()
    #h, c = init_hidden() # DRQN
    done = False
    
    while not done:
        # PPO, A3C
        prob, _ = Net(torch.FloatTensor([state])) 
        action = torch.argmax(prob).item()

        # DQN, DRQN
        #Q = Net(torch.FloatTensor([state])) # DQN
        #Q, (next_h, next_c) = Net(torch.FloatTensor([state])) # DRQN
        #action = torch.argmax(Q).item()

        next_state, reward, done, info = env.step(action)
        env.render()

        score += reward

        state = next_state

        # DRQN
        #h = next_h 
        #c = next_c
        time.sleep(0.03)
    
    print('score : {}'.format(score))

if __name__ == "__main__":
    test()