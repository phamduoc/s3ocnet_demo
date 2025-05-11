import torch
import torch.nn as nn
import torch.nn.functional as F

class SOCHHead(nn.Module):
    def __init__(self, input_dim=512, num_clusters=10):
        super(SOCHHead, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_clusters, input_dim))

    def forward(self, features):
        # Chuẩn hóa đặc trưng và prototype
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        # Tính cosine similarity giữa đặc trưng và prototype
        similarity = torch.matmul(features, prototypes.T)

        # Shape: (batch_size, num_clusters)
        return similarity