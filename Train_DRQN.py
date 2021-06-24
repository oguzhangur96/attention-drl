import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import pickle
import numpy as np
from statistics import mean
from Buffer import ReplayBuffer
from Environment import CreateBreakout
from Network import QNet_LSTM

# settings
Train_max_step         = 4000000
learning_rate          = 3e-4
gamma                  = 0.99
buffer_capacity        = 250000
batch_size             = 32
replay_start_size      = 50000
final_exploration_step = 1000000
update_interval        = 10000 # target net
update_frequency       = 4  # the number of actions selected by the agent between successive SGD updates
save_interval          = 10000
model_path = './Models/Breakout_DARQN.model'
buffer_path = './Models/Breakout_DARQN.buffer'
history_path = './Train_Historys/Breakout_DARQN'
eval_history_path = './Train_Historys/eval_Breakout_DARQN'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




def init_hidden():
    h, c = torch.zeros([1, 64], dtype=torch.float).to(device), torch.zeros([1, 64], dtype=torch.float).to(device)
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
    TD_target = r_batch + gamma * max_next_Q * done_batch[:-1]

    loss = F.smooth_l1_loss(Q_a, TD_target.detach())
    # print(start - time.time())
    start = time.time()
    optimizer.zero_grad()
    loss.backward()
    # print(start - time.time())
    optimizer.step()


def main():
    env = CreateBreakout(stack=False)
    buffer = ReplayBuffer('buffer_capacity')
    behaviourNet = QNet_LSTM().to(device)
    # behaviourNet.load_state_dict(torch.load(model_path))
    targetNet = QNet_LSTM().to(device)
    targetNet.load_state_dict(behaviourNet.state_dict())
    optimizer = torch.optim.Adam(behaviourNet.parameters(), learning_rate)

    score_history = []
    train_history = []
    eval_history = []
    # train_history = np.load(history_path+'.npy').tolist()

    step = 0
    score = 0

    state = env.reset()
    h, c = init_hidden()
    start = time.time()
    print("Train start")
    while step < Train_max_step:
        # env.render()
        epsilon = max(0.1, 1.0 - (0.9 / final_exploration_step) * step)
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

        if step % update_frequency == 0 and buffer.size() > replay_start_size:
            s_batch, a_batch, r_batch, done_batch = buffer.sample(batch_size)
            train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, done_batch)

        if step % update_interval == 0 and buffer.size() > replay_start_size:
            targetNet.load_state_dict(behaviourNet.state_dict())

        if step > 0 and step % save_interval == 0:
            state = env.reset()
            done = False
            # reset environment and set episodic reward to 0 for each episode start
            episodic_reward = 0
            h, c = init_hidden()
            while not done:
                # take action get next state, rewards and terminal status
                action_value, (next_h, next_c) = behaviourNet.forward(torch.FloatTensor([state]).to(device), (h, c))
                state, reward, done, info = env.step(action_value.argmax().item())
                episodic_reward = episodic_reward + reward
                h, c = next_h, next_c
            h, c = init_hidden()
            state = env.reset()
            score = 0

            train_history.append(mean(score_history))
            eval_history.append(episodic_reward)

            torch.save(behaviourNet.state_dict(), model_path)
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer, f)
            np.save(history_path, np.array(train_history))
            np.save(eval_history_path, np.array(eval_history))

            end = time.time()
            print(
                f"Step No: {step}, Train average: {mean(score_history)}, Eval Average: {episodic_reward}, epsilon: {epsilon}, time = {end - start} ")
            start = end
    with open(buffer_path, 'wb') as f:
        pickle.dump(buffer, f)
    torch.save(behaviourNet.state_dict(), model_path)
    np.save(history_path, np.array(train_history))
    np.save(eval_history_path, np.array(eval_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

def main_v2():
    env = CreateBreakout(stack=False)
    behaviourNet = QNet_LSTM().to(device)
    print(sum(p.numel() for p in behaviourNet.parameters() if p.requires_grad))
    behaviourNet.load_state_dict(torch.load(model_path))
    state = env.reset()
    h, c = init_hidden()
    episodic_reward = 0
    done = False
    while not done:
        # take action get next state, rewards and terminal status
        action_value, (next_h, next_c) = behaviourNet.forward(torch.FloatTensor([state]).to(device), (h, c))
        # print(action_value.argmax().item())
        state, reward, done, info = env.step(action_value.argmax().item())
        episodic_reward = episodic_reward + reward
        h, c = next_h, next_c
        env.render()
        # print(h)
        time.sleep(1/10)

    print(episodic_reward)


if __name__ == "__main__":
    main()