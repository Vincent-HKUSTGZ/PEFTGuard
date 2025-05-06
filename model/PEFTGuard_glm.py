import torch
import torch.nn as nn
import torch.nn.functional as F

class PEFTGuard_Glm2_6b(nn.Module):
    def __init__(self,device, target_number=2):
        super(PEFTGuard_Glm2_6b, self).__init__()
        self.device = device
        self.input_channel = target_number * 28
        self.conv1 = nn.Conv2d(self.input_channel, 16, kernel_size=8, stride=8, padding=0).to(self.device)
        self.fc1 = nn.Linear(576*512*16, 512).to(self.device)
        self.fc2 = nn.Linear(512, 128).to(self.device)
        self.fc3 = nn.Linear(128, 2).to(self.device)

    def forward(self, x):
        x = x.view(-1, self.input_channel, 4608, 4096)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
