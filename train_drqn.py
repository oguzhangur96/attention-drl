import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
from statistics import mean
from Buffer import ReplayBuffer
from Environment import CreateBreakout
from Network import QNet_LSTM

parser = argparse.ArgumentParser(description='PyTorch RL trainer')

parser.add_argument('--train_max_step', default=2000000, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--buffer_capacity', default=60000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--replay_start_size', default=50000, type=int)
parser.add_argument('--final_exploration_step', default=1000000, type=int)
parser.add_argument('--update_interval', default=10000, type=int)
parser.add_argument('--update_frequency', default=4, type=int)
parser.add_argument('--save_interval', default=50000, type=int)
parser.add_argument('--print_every', default=10000, type=int)
parser.add_argument('--model_path', default='./Models/Breakout_DRQN.model', type=str)
parser.add_argument('--param_path', default='./Models/parameters_DRQN.npy', type=str)
parser.add_argument('--history_path', default='./Train_Historys/Breakout_DRQN.npy', type=str)
parser.add_argument('--epsilon_start', default=1, type=int)
parser.add_argument('--epsilon_min', default=0.1, type=float)
parser.add_argument('--load', default="False", type=str)

args = parser.parse_args()

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

    for state, done in zip(s_batch, done_batch):
        Q, (hb, cb) = behaviourNet.forward(state.unsqueeze(0), (hb, cb))
        target_Q, (ht, ct) = targetNet.forward(state.unsqueeze(0), (ht, ct))

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

    behaviourNet = QNet_LSTM().to(device)
    targetNet = QNet_LSTM().to(device)
    targetNet.load_state_dict(behaviourNet.state_dict())
    optimizer = torch.optim.Adam(behaviourNet.parameters(), args.learning_rate)

    if args.load == "True":
        
        behaviourNet.load_state_dict(torch.load(args.model_path))
        targetNet.load_state_dict(behaviourNet.state_dict())
    
    if args.load == "True":
        train_history = list(np.load(args.history_path))
        param_dict = np.load(args.param_path, allow_pickle='TRUE').item()
        step = param_dict["step"]
        print("weights and train_history loaded!")
    else:
        train_history = []
        param_dict = {}
        step = 0

    score_history = []
    score = 0

    state = env.reset()
    h, c = init_hidden()

    print("Train start")
    while step < args.train_max_step:
        # env.render()
        epsilon = max(args.epsilon_min, args.epsilon_start - ((args.epsilon_start-args.epsilon_min) / args.final_exploration_step) * step)
        action_value, (next_h, next_c) = behaviourNet.forward(torch.FloatTensor([state]).to(device), (h, c))

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
            param_dict["step"] = step
            train_history.append(mean(score_history))
            torch.save(behaviourNet.state_dict(), args.model_path)
            np.save(args.param_path, param_dict)
            np.save(args.history_path, np.array(train_history))

        if step > 0 and step % args.print_every == 0:
            print(f"Step No: {step}, Train average: {mean(score_history)}, epsilon: {epsilon}")

    torch.save(behaviourNet.state_dict(), args.model_path)
    np.save(args.+param_path, param_dict)
    np.save(args.history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))


if __name__ == "__main__":
    main(args)