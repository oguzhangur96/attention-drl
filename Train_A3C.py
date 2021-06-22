import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
import torch.multiprocessing as mp

import random
import numpy as np

from Environment import CreateBreakout
from Network import ActorCriticNet

# settings
learning_rate          = 1e-4
gamma                  = 0.99
beta                   = 0.01
max_T                  = 5000
max_t                  = 32
N_worker               = 8
model_path             = './Models/Breakout_A3C.model'

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def local_train(process, global_model, optimizer):
    env = CreateBreakout()
    local_model = ActorCriticNet()
    local_model.load_state_dict(global_model.state_dict())

    total_reward =0
    max_score = 0

    for T in range(max_T):
        state = env.reset()
        done = False
        score = 0

        while not done:
            log_probs, values, entropys, rewards= [], [], [], []
            for t in range(max_t):
                prob, value = local_model(torch.FloatTensor([state]))

                m = Categorical(prob)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = m.entropy()

                next_state, reward, done, _ = env.step(action.item())
                score += reward

                log_probs.append(log_prob)
                values.append(value)
                entropys.append(entropy)
                rewards.append(reward)

                state = next_state
                if done:
                    break
            
            state_final = torch.FloatTensor([next_state])

            R = 0.0
            if not done:
                _, R = local_model(state_final)
                R = R.item()

            td_target_lst = []
            for reward in rewards[::-1]:
                R = reward + R * gamma
                td_target_lst.append([R])
            td_target_lst.reverse()

            log_probs = torch.stack(log_probs)
            values = torch.cat(values)
            entropys = torch.stack(entropys)
            td_targets = torch.FloatTensor(td_target_lst)
            advantages = (td_targets - values).detach()

            actor_loss = -torch.mean(log_probs * advantages)
            critic_loss = F.smooth_l1_loss(values, td_targets.detach())
            entropy_loss = torch.mean(entropys)
            
            total_loss = actor_loss + critic_loss - beta * entropy_loss
            
            optimizer.zero_grad()
            local_model.zero_grad()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5)

            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad
            
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

        total_reward += score
        if score > max_score:
            max_score = score

        if (T+1) % 10 == 0 :
            print('Process {} of episode {}, avg score : {}, max score : {}'.format(process, T+1, total_reward/10, max_score))
            total_reward = 0
    
    env.close()

def main():
    global_model = ActorCriticNet()
    #global_model.load_state_dict(torch.load(model_path))
    global_model.share_memory()

    optimizer = GlobalAdam(global_model.parameters(), learning_rate)
    
    processes = []

    for process in range(N_worker):
        p = mp.Process(target=local_train, args=(process, global_model, optimizer,))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    
    torch.save(global_model.state_dict(), model_path)

if __name__ == "__main__":
    main()