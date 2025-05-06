import torch
import torch.nn as nn
import torch.nn.functional as F

class PEFTGuard_qwenvl(nn.Module):
    def __init__(self, device, target_number=2):
        super(PEFTGuard_qwenvl, self).__init__()
        self.device = device
        
        self.conv1_q = nn.Conv2d(28, 32, kernel_size=4, stride=4, padding=0)
        self.bn1_q = nn.BatchNorm2d(32)
        self.pool1_q = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_q = nn.Linear(32 * 192 * 192, 512) 
        
        self.conv1_v = nn.Conv2d(28, 32, kernel_size=2, stride=2, padding=0)
        self.bn1_v = nn.BatchNorm2d(32)
        self.pool1_v = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_v = nn.Linear(32 * 64 * 384, 512)
        
        self.fc2 = nn.Linear(1024, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, target_number)

        self.to(self.device)

    def forward(self, x):
        q = x['q_proj']
        v = x['v_proj']
        
        q = q.view(-1, 28, 1536, 1536)
        q = self.conv1_q(q)
        q = self.bn1_q(q)
        q = F.leaky_relu(q)
        q = self.pool1_q(q)
        q = q.view(q.size(0), -1)
        q = F.leaky_relu(self.fc1_q(q))
        
        v = v.view(-1, 28, 256, 1536)
        v = self.conv1_v(v)
        v = self.bn1_v(v)
        v = F.leaky_relu(v)
        v = self.pool1_v(v)
        v = v.view(v.size(0), -1)
        v = F.leaky_relu(self.fc1_v(v))
        
        x = torch.cat((q, v), dim=1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x