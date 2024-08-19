import torch
import torch.nn.functional as F
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.linear_transform = nn.Linear(input_size, input_size)
        self.linear_transform1 = nn.Linear(input_size, 1, bias=False)

    def forward(self, embeddings):
        # embeddings: [batch_size, seq_len, hidden_size]
        transformed_embeddings = torch.tanh(self.linear_transform(embeddings))
        attention_weights = torch.softmax(
            self.linear_transform1(transformed_embeddings), dim=1
        )
        attended_embeddings = torch.sum(attention_weights * embeddings, dim=1)
        return attended_embeddings


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_routes, in_dim, out_dim):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.randn(num_capsules, num_routes, in_dim, out_dim))

    def squash(self, s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (s_j_norm**2 / (1 + s_j_norm**2)) * (s_j / s_j_norm)

    def forward(self, x):
        # x: [batch_size, num_capsules, num_routes, in_dim]
        # W: [num_capsules, num_routes, in_dim, out_dim]

        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.unsqueeze(1).expand(batch_size, self.num_capsules, seq_len, self.in_dim)
        u_hat = torch.matmul(x, self.W[:, :seq_len, :, :])
        b_ij = torch.zeros(batch_size, self.num_capsules, seq_len, 1).to(x.device)

        for iteration in range(3):
            c_ij = F.softmax(b_ij, dim=1)  # softmax along num_routes
            s_j = (c_ij * u_hat).sum(dim=2)
            v_j = self.squash(s_j)

            if iteration < 2:
                b_ij = b_ij + torch.matmul(u_hat, v_j.unsqueeze(-1)).squeeze(-1)

        return v_j


class Adapter(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x
