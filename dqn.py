import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyperparameters 超參數
BATCH_SIZE = 32
LR = 0.01                 # learning rate
EPSILON = 0.9             # 最優選擇動作百分比
GAMMA = 0.9               # 獎勵遞減參數
TARGET_REPLACE_ITER = 100 # Q現實網路的更新頻率
MEMORY_CAPACITY = 2000    # 記憶庫大小
env = gym.make('CartPole-v0')
env = env.unwrapped       # 不做這個會有很多限制
N_ACTIONS = env.action_space.n
N_STATES  = env.observation_space.shape[0]

class Net(nn.Module):
  def __init__(self,):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(N_STATES, 10)
    self.fc1.weight.data.normal_(0, 0.1) # initailization
    self.out = nn.Linear(10, N_ACTIONS)
    self.out.weight.data.normal_(0, 0.1) # initailization

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    actions_value = self.out(x)
    return actions_value

class DQN(object):
  def __init__(self):
    # 建立target net, eval net, and memory
    self.eval_net, self.target_net = Net(), Net()

  def choose_action(self, x):
    return action

  def store_transition(self, s, a, r, s_):

  def learn(self):
    # 更新Target網路，學習記憶庫中的記憶
