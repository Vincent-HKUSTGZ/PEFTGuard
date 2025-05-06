import torch
import torch.nn as nn
import torch.nn.functional as F

class PEFTGuard_roberta(nn.Module):
    def __init__(self,device, target_number=2):
        super(PEFTGuard_roberta, self).__init__()
        self.device = device
        self.input_channel = target_number * 12
        self.conv1 = nn.Conv2d(self.input_channel, 24, kernel_size=3, stride=2, padding=1).to(self.device)
        self.fc1 = nn.Linear(384*384*24, 512).to(self.device)
        self.fc2 = nn.Linear(512, 128).to(self.device)
        self.fc3 = nn.Linear(128, 2).to(self.device)

    def forward(self, x):
        x = x.view(-1, self.input_channel, 768, 768)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x