import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linner_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.liner1 = nn.Linear(input_size, hidden_size)
        self.liner2 = nn.Linear(input_size, output_size)

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

        
    def train_step (self, state ,action, reward, next_stage, done ):
        state = torch.tensor(state, dtype=torch.float)
        next_stage = torch.tensor(next_stage, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        #handle mulltiple sizes

        if len(state.shape) ==1:
            state  = torch.unsqueeze(state, 0)
            reward = torch.unsqueeze(reward,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )



    
                   
