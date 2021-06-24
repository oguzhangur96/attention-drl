import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import pickle
import argparse
import numpy as np
from statistics import mean
from Buffer import ReplayBuffer
from Environment import CreateBreakout
from Network import QNet_DARQN

parser = argparse.ArgumentParser(description='PyTorch RL trainer')

parser.add_argument('--train_max_step', default=4000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--buffer_capacity', default=500000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--replay_start_size', default=50000, type=int)
parser.add_argument('--final_exploration_step', default=100000, type=int)
parser.add_argument('--update_interval', default=10000, type=int)
parser.add_argument('--update_frequency', default=4, type=int)
parser.add_argument('--save_interval', default=10000, type=int)
parser.add_argument('--model_path', default='./Models/Breakout_DARQN.model', type=str)
parser.add_argument('--history_path', default='./Train_Historys/Breakout_DARQN', type=str)
parser.add_argument('--epsilon_start', default=1, type=int)
parser.add_argument('--epsilon_min', default=0.99, type=float)
parser.add_argument('--model_save_every', default=1000, type=int)
parser.add_argument('--load', default="False", type=str)

args = parser.parse_args()

# settings
"""
Train_max_step         = 4000000
learning_rate          = 1e-4
gamma                  = 0.99
buffer_capacity        = 500000
batch_size             = 32
replay_start_size      = 50000
final_exploration_step = 100000
update_interval        = 10000 # target net
update_frequency       = 4  # the number of actions selected by the agent between successive SGD updates
save_interval          = 10000
model_path = './Models/Breakout_DARQN.model'
buffer_path = './Models/Breakout_DARQN.buffer'
history_path = './Train_Historys/Breakout_DARQN'
eval_history_path = './Train_Historys/eval_Breakout_DARQN'

epsilon_start = 1
epsilon_min = 0.1
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def init_hidden():
    h, c = torch.zeros([1, 512], dtype=torch.float).to(device), torch.zeros([1, 512], dtype=torch.float).to(device)
    return h, c


def train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, done_batch):
    s_batch = torch.FloatTensor(s_batch).to(device)
    a_batch = torch.LongTensor(a_batch[:-1]).to(device)
    r_batch = torch.FloatTensor(r_batch[:-1]).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device)

    hb, cb = init_hidden()
    ht, ct = init_hidden()
    Q_batch = []
    target_Q_batch = []
    # start = time.time()
    for state, done in zip(s_batch, done_batch):
        Q, (hb, cb) = behaviourNet(state.unsqueeze(0), (hb, cb))
        target_Q, (ht, ct) = targetNet(state.unsqueeze(0), (ht, ct))

        Q_batch.append(Q)
        target_Q_batch.append(target_Q)

        if done.item() == 0:
            hb, cb = init_hidden()
            ht, ct = init_hidden()

    Q_batch = torch.cat(Q_batch[:-1])
    next_Q_batch = torch.cat(target_Q_batch[1:])

    Q_a = Q_batch.gather(1, a_batch)

    max_next_Q = next_Q_batch.max(1, keepdims=True)[0]
    TD_target = r_batch + args.gamma * max_next_Q * done_batch[:-1]

    loss = F.smooth_l1_loss(Q_a, TD_target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main(args):
    env = CreateBreakout(stack=False)
    buffer = ReplayBuffer(args.buffer_capacity)
    behaviourNet = QNet_DARQN().to(device)
    targetNet = QNet_DARQN().to(device)
    targetNet.load_state_dict(behaviourNet.state_dict())
    optimizer = torch.optim.Adam(behaviourNet.parameters(), args.learning_rate)

    if args.load == "True":
        behaviourNet.load_state_dict(torch.load(args.model_path))
        targetNet.load_state_dict(behaviourNet.state_dict())

    score_history = []
    train_history = []

    step = 0
    score = 0

    state = env.reset()
    h, c = init_hidden()
    start = time.time()

    print("Train start")
    while step < args.train_max_step:
        # env.render()
        epsilon = max(args.epsilon_min, args.epsilon_start - ((args.epsilon_start-args.epsilon_min) / args.final_exploration_step) * step)
        action_value, (next_h, next_c) = behaviourNet(torch.FloatTensor([state]).to(device), (h, c))

        # epsilon greedy
        coin = random.random()
        if coin < epsilon:
            action = random.randrange(4)
        else:
            action = action_value.argmax().item()

        next_state, reward, done, info = env.step(action)
        buffer.push((state, action, reward, 1 - done))

        score += reward
        step += 1

        if done:
            next_state = env.reset()
            next_h, next_c = init_hidden()
            score_history.append(score)
            score = 0
            if len(score_history) > 100:
                del score_history[0]

        state = next_state
        h = next_h.detach()
        c = next_c.detach()

        if step % args.update_frequency == 0 and buffer.size() > args.replay_start_size:
            s_batch, a_batch, r_batch, done_batch = buffer.sample(args.batch_size)
            train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, done_batch)

        if step % args.update_interval == 0 and buffer.size() > args.replay_start_size:
            targetNet.load_state_dict(behaviourNet.state_dict())

        if step > 0 and step % args.save_interval == 0:
            train_history.append(mean(score_history))
            torch.save(behaviourNet.state_dict(), args.model_path)
            np.save(args.history_path, np.array(train_history))

            print(f"Step No: {step}, Train average: {mean(score_history)}, epsilon: {epsilon}")
            
    torch.save(behaviourNet.state_dict(), args.model_path)
    np.save(args.history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

if __name__ == "__main__":
    main(args)