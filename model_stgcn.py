"""model_contrastive.py
=======================
ST-GCN based contrastive model for shot detection.

Includes:
• STGCNBlock: spatial graph conv + temporal conv
• STGCN: multi-block ST-GCN with pooling & projection head
• STGCNContrastiveModel: wrapper for (B, T, 34) input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------- Graph Convolution ----------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x, A):
        # x: (N, C, T, V), A: (V, V)
        x = torch.einsum('nctv,vw->nctw', x, A)
        return self.conv(x)

# ---------- ST-GCN Block ----------
class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, V, kernel_size=9, stride=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size // 2, 0)
        self.gcn = GraphConv(in_c, out_c)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_c, out_c, (kernel_size,1), (stride,1), pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        if in_c != out_c or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.res = nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x, A):
        res = self.res(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)

# ---------- Full ST-GCN ----------
class STGCN(nn.Module):
    def __init__(self, in_ch=2, num_joints=17, proj_dim=128):
        super().__init__()
        # Build & normalize adjacency matrix
        A = torch.zeros(num_joints, num_joints)
        edges = [(0,1),(1,3),(0,2),(2,4),(0,5),(5,7),(7,9),(0,6),(6,8),
                 (8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        for i,j in edges:
            A[i,j] = 1; A[j,i] = 1
        for i in range(num_joints): A[i,i] = 1
        D_inv = torch.diag(1.0 / A.sum(1))
        A_norm = D_inv @ A
        self.register_buffer('A', A_norm)
        # Data BN
        self.data_bn = nn.BatchNorm1d(in_ch * num_joints)
        # Layers
        self.layer1 = STGCNBlock(in_ch, 64, num_joints)
        self.layer2 = STGCNBlock(64, 128, num_joints)
        self.layer3 = STGCNBlock(128, 256, num_joints)
        self.layer4 = STGCNBlock(256, 256, num_joints)
        # Pool & projection
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(256, proj_dim)
    def forward(self, x):
        # x: (N, T, V, C)  C=2, V=17
        N, T, V, C = x.shape
        x = x.permute(0,3,1,2)      # → (N, C, T, V)
        x = x.reshape(N, C*V, T)    # → (N, C*V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, T, V)   # → (N, C, T, V)
        x = self.layer1(x, self.A)
        x = self.layer2(x, self.A)
        x = self.layer3(x, self.A)
        x = self.layer4(x, self.A)
        x = self.pool(x)            # → (N, 256, 1, 1)
        x = x.view(N, -1)           # → (N, 256)
        z = self.fc(x)              # → (N, proj_dim)
        return F.normalize(z, dim=1)

# ---------- Contrastive Wrapper ----------
class STGCNContrastiveModel(nn.Module):
    """Reshapes (B, T, 34) → (B, T, 17, 2) → ST-GCN"""
    def __init__(self, proj_dim=128):
        super().__init__()
        self.stgcn = STGCN(in_ch=2, num_joints=17, proj_dim=proj_dim)
    def forward(self, seq):
        # seq: (B, T, 34)  # 34 = 17 joints × (x,y)
        B, T, D = seq.shape
        x = seq.view(B, T, 17, 2)
        return self.stgcn(x)  # → (B, proj_dim)
