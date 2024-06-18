import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorNet(nn.Module):
    def __init__(self, h, w, outputs, gamma, arch=0):
        super(ActorNet, self).__init__()
        self.height = h
        self.width = w
        self.n_actions = outputs

        self.gamma = gamma
        self.temperature = 1.    # factor 1/(1-gamma) no need after normalizing the cost
        # only change the first layer
        if arch == 0:   # modify first kernel so that total parameters are approximately equal
            if h == 20:
                k0, s0 = (3, 1)
            if h == 40:
                k0, s0 = (5, 2)
            if h == 60:
                k0, s0 = (7, 3)
            k1, s1 = (3, 1)
        elif arch == 1:   # same kernel for all resolutions
            k0, s0 = (3, 1)
            k1, s1 = (3, 1)
        else: 
            pass    # other architectures
        k2, s2 = (3, 1)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=k0, stride=s0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=k1, stride=s1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=k2, stride=s2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, k0, s0), k1, s1), k2, s2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, k0, s0), k1, s1), k2, s2)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        nn.init.constant_(self.head.weight, 1 / linear_input_size)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    def get_score(self, state, action):
        action = F.one_hot(action, self.n_actions)
        output = (self.forward(state) * action).sum(dim=1, keepdim=True)
        return output
        
    def get_action(self, state, eps=0, sample=True):
        x = state
        x = x.reshape(-1, 3, self.height, self.width)
        with torch.no_grad():
            probs = self.forward(x)
            probs = F.softmax(probs/self.temperature, dim=1)
            probs = (1-eps)*probs + eps/self.n_actions
        if sample:
            m = Categorical(probs)
            action = m.sample().detach().cpu().numpy()
            return action
        else:   # TODO
            return None

    def step_temperature(self):
        self.temperature = self.temperature * self.gamma
        
class SingleCritic(nn.Module):     # subnet for each action
    def __init__(self, h, w, outputs, arch=0):
        super(SingleCritic, self).__init__()
        
        if arch == 0:   # modify first kernel so that total parameters are approximately equal
            if h == 20:
                k0, s0 = (3, 1)
            if h == 40:
                k0, s0 = (5, 2)
            if h == 60:
                k0, s0 = (7, 3)
            k1, s1 = (3, 1)
        elif arch == 1:   # same kernel for all resolutions
            k0, s0 = (3, 1)
            k1, s1 = (3, 1)
        else: 
            pass    # other architectures
        k2, s2 = (3, 1)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=k0, stride=s0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=k1, stride=s1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=k2, stride=s2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, k0, s0), k1, s1), k2, s2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, k0, s0), k1, s1), k2, s2)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class CriticNet(nn.Module):
    def __init__(self, h, w, n_actions, arch=0):
        super(CriticNet, self).__init__()
        # self.critics = nn.ModuleList([SingleCritic(h, w, 1) for _ in range(n_actions)])
        self.critics = SingleCritic(h, w, n_actions, arch=arch)
        self.n_actions = n_actions

    def forward(self, state, action):
        action = F.one_hot(action, self.n_actions)
        output = (self.critics(state) * action).sum(dim=1, keepdim=True)
        return output
    
    def get_values(self, state):
        output = self.critics(state)
        return output
