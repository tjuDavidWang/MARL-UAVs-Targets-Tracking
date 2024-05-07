import torch.optim as optim
import collections
import random
import numpy as np
# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output1, output2):
        loss = torch.mean(torch.log(1 + torch.exp(-output1)) + torch.log(1 + torch.exp(output2)))
        return loss


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
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        comm = x[:, :self.comm_dim]
        obs = x[:, self.comm_dim:self.comm_dim + self.obs_dim]
        boundary_state = x[:, self.comm_dim + self.obs_dim:self.comm_dim + self.obs_dim + self.boundary_state_dim]
        
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
        self.eval()
        if isinstance(single_data, np.ndarray):
            single_data = torch.tensor(single_data, dtype=torch.float32)
            
        if single_data.ndim == 1:
            single_data = single_data.unsqueeze(0)
        output = self.forward(single_data)
        return output.item()  # Extract and return the single scalar value

    def train_pmi(self, train_data, n_uav, bs=16):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_function = CustomLoss()

        i1, i2, i3 = torch.randint(0, n_uav, (3,))
        data_1 = train_data[torch.arange(train_data.size(0)) % n_uav == i1]  # l1, a1
        data_2 = train_data[torch.arange(train_data.size(0)) % n_uav == i2]  # l2, a1
        data_3 = train_data[torch.arange(train_data.size(0)) % n_uav == i3]  # l2_, a2_
        permutation = torch.randperm(data_1.size()[0])
        for i in range(0, data_1.size()[0], bs):
            optimizer.zero_grad()

            indices = permutation[i:i + bs]

            # 计算损失
            input1 = (data_1 * data_2)
            input2 = (data_1 * data_3)
            output_l1_a1_l2_a2 = self.forward(input1[indices])
            output_l1_a1_l3_a3 = self.forward(input2[indices])

            loss = loss_function(output_l1_a1_l2_a2, output_l1_a1_l3_a3)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        print(f'PMI Loss: {loss.item()}')



if __name__ == "__main__":
    # 初始化网络

    h_dim = 64

    model = PMINetwork(hidden_dim=h_dim)

    # 示例数据 (随机生成的样本)
    torch.manual_seed(0)
    num_samples = 100
    data = torch.randn(num_samples, 12)  # 随机生成100个样本，每个样本由4个数字组成
    num_epochs = 10
    batch_size = 5

    model.train_pmi(data, num_epochs, batch_size)

    # 检查模型输出
    print("Sample Outputs:", model.inference(data[:1]))
