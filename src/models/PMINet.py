import torch.optim as optim
import numpy as np
# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as f
import os


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @staticmethod
    def forward(output1, output2):
        loss = torch.mean(torch.log(1 + torch.exp(-output1)) + torch.log(1 + torch.exp(output2)))
        return loss


class PMINetwork(nn.Module):
    def __init__(self, comm_dim=5, obs_dim=4, boundary_state_dim=3, hidden_dim=64, b2_size=3000):
        super(PMINetwork, self).__init__()
        self.comm_dim = comm_dim
        self.obs_dim = obs_dim
        self.boundary_state_dim = boundary_state_dim
        self.hidden_dim = hidden_dim
        self.b2_size = b2_size

        self.fc_comm = nn.Linear(comm_dim, hidden_dim)
        self.bn_comm = nn.BatchNorm1d(hidden_dim) 
        self.fc_obs = nn.Linear(obs_dim, hidden_dim)
        self.bn_obs = nn.BatchNorm1d(hidden_dim) 
        self.fc_boundary_state = nn.Linear(boundary_state_dim, hidden_dim)
        self.bn_boundary_state = nn.BatchNorm1d(hidden_dim)  

        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        comm = x[:, :self.comm_dim]
        obs = x[:, self.comm_dim:self.comm_dim + self.obs_dim]
        boundary_state = x[:, self.comm_dim + self.obs_dim:self.comm_dim + self.obs_dim + self.boundary_state_dim]

        # Process each part with BatchNorm
        comm_vec = self.fc_comm(comm)
        comm_vec = f.relu(self.bn_comm(comm_vec))
        obs_vec = self.fc_obs(obs)
        obs_vec = f.relu(self.bn_obs(obs_vec))
        boundary_state_vec = self.fc_boundary_state(boundary_state)
        boundary_state_vec = f.relu(self.bn_boundary_state(boundary_state_vec))

        # Concatenate and process through further layers with BatchNorm
        combined = torch.cat((comm_vec, obs_vec, boundary_state_vec), dim=1)
        x = self.fc1(combined)
        x = f.relu(self.bn1(x))
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

    def train_pmi(self, config, train_data, n_uav):
        self.train()
        loss_function = CustomLoss()
        # train_data (timesteps*n_uav,12)
        timesteps = train_data.size(0) // n_uav
        train_data = train_data.view(timesteps, n_uav, 12)
        timestep_indices = torch.randint(low=0, high=timesteps, size=(self.b2_size,))
        uav_indices = torch.randint(low=0, high=n_uav, size=(self.b2_size, 2))
        selected_data = torch.zeros((self.b2_size, 2, 12))
        for i in range(self.b2_size):
            selected_data[i] = train_data[timestep_indices[i], uav_indices[i]]

        avg_loss = 0
        for i in range(self.b2_size // config["pmi"]["batch_size"]):
            self.optimizer.zero_grad()
            batch_data = selected_data[i * config["pmi"]["batch_size"]:(i + 1) * config["pmi"]["batch_size"]]
            input_1_2 = batch_data[:, 0].squeeze(1)
            input_1_3 = batch_data[:, 1].squeeze(1)
            output_1_2 = self.forward(input_1_2)
            output_1_3 = self.forward(input_1_3)
            loss = loss_function(output_1_2, output_1_3)
            avg_loss += abs(loss.item())
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
        avg_loss /= (self.b2_size // config["pmi"]["batch_size"])
        return avg_loss

    def save(self, save_dir, epoch_i):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(save_dir, "pmi", 'pmi_weights_' + str(epoch_i) + '.pth'))

    def load(self, path):
        if path and os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
