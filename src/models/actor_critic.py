import os.path

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = None
        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out


class ResPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ResPolicyNet, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = f.avg_pool1d(x, 12)  # 这里使用平均池化，你也可以根据需求使用其他池化方式
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return f.softmax(x, dim=1)


class ResValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ResValueNet, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class FnnPolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(FnnPolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        # 每个状态对应的动作的概率
        x = f.softmax(x, dim=1)  # [b,n_actions]-->[b,n_actions]
        return x


class FnnValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(FnnValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x.squeeze(1)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        """
        :param state_dim: 特征空间的维数
        :param hidden_dim: 隐藏层的维数
        :param action_dim: 动作空间的维数
        :param actor_lr: actor网络的学习率
        :param critic_lr: critic网络的学习率
        :param gamma: 经验回放参数
        :param device: 用于训练的设备
        """
        # 策略网络
        self.actor = FnnPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = FnnValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, states):
        """
        :param states: nparray, size(state_dim,) 代表无人机的状态
        :return:
        """
        states_np = np.array(states)[np.newaxis, :]  # 直接使用np.array来转换
        states_tensor = torch.tensor(states_np, dtype=torch.float).to(self.device)
        probs = self.actor(states_tensor)
        action_dist = torch.distributions.Categorical(probs)  # TODO ?
        action = action_dist.sample()
        return action, probs

    def update(self, transition_dict):
        """
        :param transition_dict: dict, 包含状态,动作, 单个无人机的奖励, 下一个状态的四元组
        :return: None
        """
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        # actions = actions.long()
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device).squeeze()
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))

        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(f.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss, critic_loss, td_delta

    def save(self, save_dir, epoch_i):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()
        }, os.path.join(save_dir, "actor", 'actor_weights_' + str(epoch_i) + '.pth'))
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()
        }, os.path.join(save_dir, "critic", 'critic_weights_' + str(epoch_i) + '.pth'))

    def load(self, actor_path, critic_path):
        if actor_path and os.path.exists(actor_path):
            checkpoint = torch.load(actor_path)
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if critic_path and os.path.exists(critic_path):
            checkpoint = torch.load(critic_path)
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
