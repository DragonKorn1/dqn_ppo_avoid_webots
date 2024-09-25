import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class DQN (nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        value = self.fc4(x)
        return value
         
class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # print(self.memory)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

    def can_sample(self):
        return len(self) >= self.batch_size
        
class DQNAgent(nn.Module):
    def __init__(self, observe_input, n_actions, device, batch_size = 256, gamma = 0.99, epsilon_start = 1.0,
               epsilon_end = 0.01, epsilon_decay = 0.99, learning_rate = 0.00005, memory_size = 100_000,
               training_update = 100): # Change the weight parameters of target neural network here.
        super().__init__()
        self.state_size = observe_input
        self.action_size = n_actions
        self.device = device

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPSILON_START = epsilon_start
        self.EPSILON_END = epsilon_end
        self.EPSILON_DECAY = epsilon_decay
        self.LEARNING_RATE = learning_rate
        self.MEMORY_SIZE = memory_size
        self.TARGET_UPDATE = training_update

        # Networks
        self.policy_net = DQN(observe_input, n_actions).to(self.device)
        self.target_net = DQN(observe_input, n_actions).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Adam optimizer without weight decay
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)

        # Adam Optimizer with weight decay       
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE,
                        weight_decay = 1e-4)
        self.memory = ReplayBuffer(self.MEMORY_SIZE, self.BATCH_SIZE)
        self.epsilon = self.EPSILON_START

    def select_action(self, state): 
        if random.random() > self.epsilon:
            with torch.no_grad(): # Greedy
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state).max(1)[1].item()
        else:
            return random.randrange(self.action_size) # Epsilon

    def optimize_model(self):
        if not self.memory.can_sample():
            return [[],[],[],[]]
        
        state, action, reward, next_state, done = self.memory.sample()
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        expected_q_values = reward + self.GAMMA * next_q_values * (1 - done)
        # MSE_Loss
        loss = nn.SmoothL1Loss()(q_values, expected_q_values.unsqueeze(1))
        loss = torch.clamp(loss, min= 1e-6, max=5)

        # Huber Loss
        # loss = nn.SmoothL1Loss()(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        Q_Va = q_values.cpu().detach().numpy().flatten()
        Next_Q = next_q_values.cpu().detach().numpy().flatten()
        Expect_Q = expected_q_values.cpu().detach().numpy().flatten()
        MSE = loss.cpu().detach().numpy().flatten()
        
        return [Q_Va, Next_Q, Expect_Q, MSE]