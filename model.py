import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling=True):
        super(DQN, self).__init__()

        self.enable_dueling = enable_dueling
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        if self.enable_dueling:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        
        self.input_residual = nn.Linear(state_dim, hidden_dim)

    def forward(self, x):
        residual = self.input_residual(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x) + residual
        
        residual = x
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x) + residual
        if self.enable_dueling:
            v = F.leaky_relu(self.fc_value(x))
            V = self.value(v)

            a = F.leaky_relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.fc3(x)
        
        
        return Q
    
if __name__ == '__main__':
    statedim = 12
    actdim = 2
    net = DQN(statedim, actdim)

    state = torch.randn(5, statedim)
    output = net(state)
    print(output)