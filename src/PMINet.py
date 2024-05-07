import torch.optim as optim
import collections
import random
import numpy as np
# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F


class PMIReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, communication: (float, float, float, float, int),
            observation: (float, float, float, float),
            sb: (float, float, int)):
        self.buffer.append((communication, observation, sb))

    def sample(self, batch_size_):
        transitions = random.sample(self.buffer, batch_size_)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class PMINetwork(nn.Module):
    def __init__(self, comm_dim=5, obs_dim=4, boundary_state_dim=3, hidden_dim=64):
        super(PMINetwork, self).__init__()
        self.comm_dim = comm_dim
        self.obs_dim = obs_dim
        self.boundary_state_dim = boundary_state_dim
        self.hidden_dim = hidden_dim
        self.fc_comm = nn.Linear(comm_dim, hidden_dim)
        self.fc_obs = nn.Linear(obs_dim, hidden_dim)
        self.fc_boundary_state = nn.Linear(boundary_state_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        
    def forward(self, x):
        comm = x[:, :self.comm_dim]
        obs = x[:, self.comm_dim:self.comm_dim + self.obs_dim]
        boundary_state = x[:, self.comm_dim + self.obs_dim:]
        
        # Process each part
        comm_vec = F.relu(self.fc_comm(comm))
        obs_vec = F.relu(self.fc_obs(obs))
        boundary_state_vec = F.relu(self.fc_boundary_state(boundary_state))

        # Concatenate and process through further layers
        combined = torch.cat((comm_vec, obs_vec, boundary_state_vec), dim=1)
        x = F.relu(self.fc1(combined))
        output = self.fc2(x)
        return output

    def inference(self, single_data):
            if single_data.ndim == 1:
                single_data = single_data.unsqueeze(0)
            output = self.forward(single_data)
            return output.item()  # Extract and return the single scalar value
    
if __name__ == "__main__":
    # 初始化网络

    hidden_dim = 64

    model = PMINetwork(hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()

    # 示例数据 (随机生成的样本)
    torch.manual_seed(0)
    num_samples = 100
    data = torch.randn(num_samples, 12)  # 随机生成100个样本，每个样本由4个数字组成
    num_epochs = 10
    batch_size = 5

    for epoch in range(num_epochs):
        permutation = torch.randperm(data.size()[0])
        for i in range(0, data.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x = data[indices]

            # 计算损失
            outputs = model(batch_x)
            target = torch.zeros_like(outputs)  # 目标是最小化输出，这里假设理想输出为0
            loss = loss_function(outputs, target)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # 检查模型输出
    print("Sample Outputs:", model.inference(data[:1]))
