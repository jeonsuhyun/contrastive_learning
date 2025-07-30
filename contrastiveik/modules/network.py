import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, pose_dim):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = pose_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

class SimNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, instance_dim, cluster_dim):
        super(SimNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.cluster_dim = cluster_dim
        self.instance_dim = instance_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
        self.instance_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.instance_dim),
        )
        # self.cluster_projector = nn.Sequential(
        #     nn.Linear(self.feature_dim, self.feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_dim, self.cluster_dim),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x_i, x_j):

        h_i = self.feature_extractor(x_i)
        h_j = self.feature_extractor(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        # c_i = self.cluster_projector(h_i)
        # c_j = self.cluster_projector(h_j)

        return z_i, z_j #c_i, c_j

    def inference(self, x):
        h = self.feature_extractor(x)
        c = self.instance_projector(h)
        return c
    
    def feature_inference(self, x):
        h = self.feature_extractor(x)
        return h


# Transformer-based feature extractor contained in a Network class
class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, instance_dim, cluster_dim, 
                    num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, seq_len=1):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.instance_dim = instance_dim
        self.cluster_dim = cluster_dim
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        self.instance_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, instance_dim),
        )

    def forward(self, x_i, x_j):
        # x_i, x_j: [batch_size, input_dim] or [batch_size, seq_len, input_dim]
        h_i = self._extract_features(x_i)
        h_j = self._extract_features(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        return z_i, z_j #c_i, c_j

    def _extract_features(self, x):
        # x: [batch_size, input_dim] or [batch_size, seq_len, input_dim]
        if x.dim() == 2:
            # If input is [batch_size, input_dim], add seq_len=1 dimension
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        # Pooling: take the mean over the sequence dimension
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x

    def inference(self, x):
        h = self._extract_features(x)
        c = self.instance_projector(h)
        return c

    def feature_inference(self, x):
        h = self._extract_features(x)
        return h

