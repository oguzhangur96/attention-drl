import torch
import torch.nn as nn
import torch.nn.functional as F 

import random
import numpy as np
from statistics import mean

from Environment import CreateBreakout
from Network import QNet

# settings
Train_max_step         = 4000000
learning_rate          = 1e-4
gamma                  = 0.99
buffer_capacity        = 1000000
batch_size             = 32
replay_start_size      = 50000
final_exploration_step = 1000000
update_interval        = 10000 # target net
update_frequency       = 4  # the number of actions selected by the agent between successive SGD updates
save_interval          = 1000
model_path             = './Models/Breakout_DQN.model'
history_path           = './Train_Historys/Breakout_DQN'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.Buffer = []
        self.position = 0

    def push(self, transition):
        """
        push transition data to Beffer

        input:
          transition -- list of [s, a, r, s_prime, t]
        """
        if len(self.Buffer) < self.capacity:
            self.Buffer.append(None)
        self.Buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        mini_batch = random.sample(self.Buffer, batch_size)

        s_batch, a_batch, r_batch, s_prime_batch, t_batch = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, t = transition

            s_batch.append(s)
            a_batch.append([a])
            r_batch.append([r])
            s_prime_batch.append(s_prime)
            t_batch.append([t])
        
        return s_batch, a_batch, r_batch, s_prime_batch, t_batch

    def size(self):
        return len(self.Buffer)

def train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, s_prime_batch, done_batch):
        s_batch = torch.FloatTensor(s_batch).to(device)
        a_batch = torch.LongTensor(a_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).to(device)
        s_prime_batch = torch.FloatTensor(s_prime_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device)

        Q = behaviourNet(s_batch)
        Q_a = Q.gather(1, a_batch)

        next_Q = targetNet(s_prime_batch)
        max_next_Q = next_Q.max(1, keepdims=True)[0]

        TD_target = r_batch + gamma * max_next_Q * done_batch

        loss = F.smooth_l1_loss(Q_a, TD_target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = CreateBreakout()
    buffer = ReplayBuffer(buffer_capacity)
    behaviourNet = QNet().to(device)
    #behaviourNet.load_state_dict(torch.load(model_path))
    targetNet = QNet().to(device)
    targetNet.load_state_dict(behaviourNet.state_dict())
    optimizer = torch.optim.Adam(behaviourNet.parameters(), learning_rate)
    
    score_history = []
    train_history = []
    #train_history = np.load(history_path+'.npy').tolist()

    step = 0
    score = 0

    state = env.reset()

    print("Train start")
    while step < Train_max_step:
        epsilon = max(0.1, 1.0 - (0.9/final_exploration_step) * step)

        action_value = behaviourNet(torch.FloatTensor([state]).to(device))

        # epsilon greedy
        coin = random.random()
        if coin < epsilon:
            action = random.randrange(4)
        else:
            action = action_value.argmax().item()
        
        next_state, reward, done, info = env.step(action)
        buffer.push((state, action, reward, next_state, 1-done))

        score += reward
        step += 1

        if done:
            next_state = env.reset()
            score_history.append(score)
            score = 0
            if len(score_history)> 100:
                del score_history[0]
        
        state = next_state

        if step%update_frequency==0 and buffer.size() > replay_start_size:
            s_batch, a_batch, r_batch, s_prime_batch, done_batch = buffer.sample(batch_size)
            train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, s_prime_batch, done_batch)

        if step % update_interval==0 and buffer.size() > replay_start_size:
            targetNet.load_state_dict(behaviourNet.state_dict())

        if step % save_interval == 0:
            train_history.append(mean(score_history))
            torch.save(behaviourNet.state_dict(), model_path)
            np.save(history_path, np.array(train_history))
            print("step : {}, Average score of last 100 episode : {:.1f}".format(step, mean(score_history)))
    
    torch.save(behaviourNet.state_dict(), model_path)
    np.save(history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

if __name__ == "__main__":
    main()