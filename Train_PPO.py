import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from Environment import MultipleBreakout
from Network import ActorCriticNet

# settings
Train_max_step         = 4000000
learning_rate          = 1e-4
gamma                  = 0.99
lambd                  = 0.95
eps_clip               = 0.1
K_epoch                = 10
N_worker               = 8
T_horizon              = 16
save_interval          = 1000
model_path             = './Models/Breakout_PPO.model'
history_path           = './Train_Historys/Breakout_PPO'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(Net, optimizer, states, actions, rewards, next_states, dones, old_probs):
    states = torch.FloatTensor(states).view(-1, 4, 84, 84).to(device) # (T*N, 4, 84, 84)
    actions = torch.LongTensor(actions).view(-1, 1).to(device) # (T*N, 1)
    rewards = torch.FloatTensor(rewards).view(-1, 1).to(device) # (T*N, 1)
    next_states = torch.FloatTensor(next_states).view(-1, 4, 84, 84).to(device) # (T*N, 4, 84, 84)
    dones = torch.FloatTensor(dones).view(-1, 1).to(device) # (T*N, 1)
    old_probs = torch.FloatTensor(old_probs).view(-1, 1).to(device) # (T*N, 1)

    for _ in range(K_epoch):
        probs, values = Net(states) # (T*N, num_action), (T*N, 1)
        _, next_values = Net(next_states) # (T*N, 1)

        td_targets = rewards + gamma * next_values * dones #(T*N, 1)
        deltas = td_targets - values # (T*N, 1)

        # calculate GAE
        deltas = deltas.view(T_horizon, N_worker, 1).cpu().detach().numpy() #(T, N, 1)
        masks = dones.view(T_horizon, N_worker, 1).cpu().numpy()
        advantages = []
        advantage = 0
        for delta, mask in zip(deltas[::-1], masks[::-1]):
            advantage = gamma * lambd * advantage * mask + delta
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device) # (T*N, 1)

        probs_a = probs.gather(1, actions) #(T*N, 1)

        m = Categorical(probs)
        entropy = m.entropy()

        ratio = torch.exp(torch.log(probs_a) - torch.log(old_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

        actor_loss = -torch.mean(torch.min(surr1, surr2))
        critic_loss = F.smooth_l1_loss(values, td_targets.detach())
        entropy_loss = torch.mean(entropy)

        loss = actor_loss + critic_loss - 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = MultipleBreakout(N_worker)
    Net = ActorCriticNet().to(device)
    #Net.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(Net.parameters(), learning_rate)

    scores = [0.0 for _ in range(N_worker)]
    score_history = []
    train_history = []
    #train_history = np.load(history_path+'.npy').tolist()
    
    step = 0

    state = env.reset() # (N, 4, 84, 84)

    print("Train Start")
    while step <= Train_max_step:
        states, actions, rewards, next_states, dones, old_probs = list(), list(), list(), list(), list(), list()
        for _ in range(T_horizon):
            prob, _ = Net(torch.FloatTensor(state).to(device))
            m = Categorical(prob)

            action = m.sample() # (N,)
            old_prob = prob.gather(1, action.unsqueeze(1)) # (N, 1)

            action = action.cpu().detach().numpy()
            old_prob = old_prob.cpu().detach().numpy()

            next_state, reward, done = env.step(action) #(N, 4, 84, 84), (N,), (N,)

            # save transition
            states.append(state) # (T, N, 4, 84, 84)
            actions.append(action) # (T, N)
            rewards.append(reward/10.0) # (T, N)
            next_states.append(next_state) # (T, N, 4, 84, 84)
            dones.append(1-done) # (T, N)
            old_probs.append(old_prob)# (T, N, 1)


            # record score and check done
            for i, (r, d) in enumerate(zip(reward, done)):
                scores[i] += r

                if d==True:
                    score_history.append(scores[i])
                    scores[i] = 0.0
                    if len(score_history) > 100:
                        del score_history[0]

            state = next_state

            step += 1

            if step % save_interval == 0:
                torch.save(Net.state_dict(), model_path)
                train_history.append(mean(score_history))
                np.save(history_path, np.array(train_history))
                print("step : {}, Average score of last 100 episode : {:.1f}".format(step, mean(score_history)))

        train(Net, optimizer, states, actions, rewards, next_states, dones, old_probs)

    torch.save(Net.state_dict(), model_path)
    np.save(history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

if __name__ == "__main__":
    main()
        