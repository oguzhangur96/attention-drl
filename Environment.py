import random
import cv2
import numpy as np
import gym
from gym import Wrapper

class CustomBreakout(Wrapper):
    def __init__(self, env, size=84, skip=4):
        super(CustomBreakout, self).__init__(env)
        self.skip = skip
        self.size = size
    
    def ProcessFrame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size)) / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        state = self.ProcessFrame(state)
        return state, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.ProcessFrame(state)
        return state
    
class CustomBreakout_stack(Wrapper):
    def __init__(self, env, size=84, skip=4):
        super(CustomBreakout_stack, self).__init__(env)
        self.skip = skip
        self.size = size
        self.history = []

    def ProcessFrame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size)) / 255.0
        return frame
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done==True:
                break

        state = self.ProcessFrame(state)

        self.history.append(state)
        del self.history[0]

        return np.stack(self.history), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.ProcessFrame(state)
        self.history = [state, state, state, state]
        return np.stack(self.history)

def CreateBreakout(stack=True):
    env = gym.make('Breakout-v0')
    if stack:
        env = CustomBreakout_stack(env)
    else:
        env = CustomBreakout(env)
    return env

class MultipleBreakout:
    def __init__(self, N, stack=True):
        self.envs = [CreateBreakout(stack) for _ in range(N)]
    
    def reset(self):
        obs = []
        for env in self.envs:
            ob = env.reset()
            obs.append(ob)
        return np.stack(obs)
    
    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            if done:
                ob = env.reset()
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(obs), np.stack(rewards), np.stack(dones)
    
    def render(self):
        for env in self.envs:
            env.render()


