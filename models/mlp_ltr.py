import torch.nn as nn
import torch
from utils.math import *

log_protect = 1e-5
multinomial_protect = 1e-10

class LtrPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='relu', log_std=0, ltr_n=1):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # learn to repeat time range.
        self.ltr_n = ltr_n
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        set_init(self.affine_layers)

        self.repeat_hid = nn.Linear(last_dim, int(last_dim/2))
        self.action_hid = nn.Linear(last_dim, int(last_dim/2))
        set_init([self.repeat_hid])
        set_init([self.action_hid])

        self.action_mean = nn.Linear(int(last_dim/2), action_dim)
        # for output action repeat times.
        self.repeat_head = nn.Linear(int(last_dim/2), self.ltr_n)
        # self.action_mean.weight.data.mul_(0.1)
        # self.action_mean.bias.data.mul_(0.0)
        set_init([self.action_mean])
        set_init([self.repeat_head])

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        ax = self.activation(self.action_hid(x))
        rx = self.activation(self.repeat_hid(x))

        action_mean = self.action_mean(ax)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        repeat_prob = torch.softmax(self.repeat_head(rx), dim=1)

        return action_mean, action_log_std, action_std, repeat_prob

    def select_action(self, x):
        action_mean, _, action_std, repeat_prob = self.forward(x)
        action = torch.normal(action_mean, action_std)
        repeat_prob += multinomial_protect
        repeat = repeat_prob.multinomial(1)
        return action, repeat

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    # modified for output both *actions* and *repeat times*.
    def get_log_prob(self, x, actions, repeats):
        action_mean, action_log_std, action_std, repeat_prob = self.forward(x)
        repeat_prob = repeat_prob.gather(1, repeats.long().unsqueeze(1))+log_protect
        return normal_log_density(actions, action_mean, action_log_std, action_std), torch.log(repeat_prob)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


