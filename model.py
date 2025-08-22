import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linner_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.liner1 = nn.Linear(input_size, hidden_size)
        self.liner2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = F.relu(self.liner1(x))
        x = self.liner2(x)
        return x
    
    def save (self, file_name = 'model.pth'):
        model_folderpath = './model'
        if not os.path.exists(model_folderpath):
            os.makedirs(model_folderpath)

        file_name = os.path.join(model_folderpath, file_name)
        torch.save(self.state_dict(),file_name)



class Qtrainner:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion  = nn.MSELoss()

        
    def train_step(self, state, action, reward, next_stage, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_stage = torch.tensor(np.array(next_stage), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # If it’s a single sample, add batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_stage = next_stage.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # predicted Q values for current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Ensure next_state has correct shape before feeding to model
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_stage[idx].unsqueeze(0)))
        
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    
                   
