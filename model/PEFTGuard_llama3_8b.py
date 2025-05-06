import torch
import torch.nn as nn
import torch.nn.functional as F

class PEFTGuard_llama3_8b(nn.Module):
    def __init__(self,device,target_number=2):
        super(PEFTGuard_llama3_8b, self).__init__()
        self.device = device
        
        self.conv1_q = nn.Conv2d(32, 16, kernel_size=8, stride=8, padding=0).to(self.device)
        self.fc1_q = nn.Linear(16 * 512 * 512, 512).to(self.device)
        
        self.conv1_v = nn.Conv2d(32, 16, kernel_size=8, stride=8, padding=0).to(self.device)
        self.fc1_v = nn.Linear(16 * 128 * 512, 512).to(self.device)
        
        self.fc2 = nn.Linear(1024, 128).to(self.device)
        self.fc3 = nn.Linear(128, 2).to(self.device)

    def forward(self, x):
        q = x['q_proj']
        v = x['v_proj']
        
        q = q.view(-1, 32, 4096, 4096)
        q = self.conv1_q(q)
        q = q.view(q.size(0), -1)
        q = F.leaky_relu(self.fc1_q(q))
        
        v = v.view(-1, 32, 1024, 4096)
        v = self.conv1_v(v)
        v = v.view(v.size(0), -1)
        v = F.leaky_relu(self.fc1_v(v))
        
        x = torch.cat((q, v), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
