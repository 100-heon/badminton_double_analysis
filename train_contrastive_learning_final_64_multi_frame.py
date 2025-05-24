"""
(Simplified - 64d version)

Run:
    python train_contrastive_learning_final_64.py
Requires:
    model_stgcn.py
    front_dataset_contrasitive.npy, back_dataset_contrasitive.npy
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from model_stgcn import STGCNContrastiveModel

# Config
OUTPUT_DIR = "model_stgcn_multi_frame"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJ_DIM = 64; BATCH = 64; LR = 1e-4

# Augmentation
# LEFT, RIGHT = [5,7,9,11,13,15], [6,8,10,12,14,16]
# def augment(seq):
#     seq = seq.copy()
#     if np.random.rand() < 0.3:
#         idx = np.sort(np.random.choice(seq.shape[0], int(seq.shape[0]*0.8), replace=False))
#         seq = np.concatenate([seq[idx], np.repeat(seq[idx[-1:]], seq.shape[0]-len(idx), axis=0)], axis=0)
#     if np.random.rand() < 1:
#         seq[..., 0::2] *= -1
#         for l, r in zip(LEFT, RIGHT):
#             lx, ly, rx, ry = 2*l, 2*l+1, 2*r, 2*r+1
#             seq[..., [lx, rx]] = seq[..., [rx, lx]]
#             seq[..., [ly, ry]] = seq[..., [ry, ly]]
#     seq += np.random.normal(0, 0.5, seq.shape)
#     seq += np.random.normal(0, 0.3, seq.shape)
#     seq[np.random.rand(*seq.shape) < 0.2] = 0
#     return seq

LEFT, RIGHT = [5,7,9,11,13,15], [6,8,10,12,14,16]

def augment_flip(seq):
    seq = seq.copy()
    # 좌우 반전 (x 좌표 부호 반전)
    seq[..., 0::2] *= -1
    # 좌우 관절 위치 교환
    for l, r in zip(LEFT, RIGHT):
        lx, ly = 2*l, 2*l+1
        rx, ry = 2*r, 2*r+1
        seq[..., [lx, rx]] = seq[..., [rx, lx]]
        seq[..., [ly, ry]] = seq[..., [ry, ly]]
    return seq

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, kp, window_size):
        # 원본 + 좌우 반전 데이터 모두 만들기
        flipped_kp = [augment_flip(seq) for seq in kp]
        self.kp = np.concatenate([kp, np.array(flipped_kp)], axis=0)
        self.window_size = window_size

    def __len__(self):
        return len(self.kp)

    def __getitem__(self, idx):
        seq = self.kp[idx]
        window = self.window_size

        center = len(seq) // 2
        half = window // 2

        start = max(center - half, 0)
        end = min(center + half + 1, len(seq))

        seq_window = seq[start:end]

        if len(seq_window) < window:
            pad_shape = (window - len(seq_window),) + seq_window.shape[1:]
            seq_window = np.concatenate([seq_window, np.zeros(pad_shape)], axis=0)


        x1 = seq_window
        x2 = seq_window

        return torch.FloatTensor(x1), torch.FloatTensor(x2)


# Loss
class NTXentLoss(nn.Module):
    def __init__(self, T=0.06):
        super().__init__(); self.T = T
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.mm(z, z.t()) / self.T
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        pos = torch.sum(z1 * z2, dim=1) / self.T
        pos = torch.cat([pos, pos], dim=0)
        loss = -pos + torch.logsumexp(sim, dim=1)
        return loss.mean()

# Training
def run_pipeline(tag, npy_path, window):
    print(f"\n=== {tag} TRAINING (64d, {window}frame) ===")
    dataset = ContrastiveDataset(npy_path, window)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=4)
    model = STGCNContrastiveModel(PROJ_DIM).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
    criterion = NTXentLoss()
    best_loss = float('inf'); patience = 15; counter = 0

    last_loss, last_pos, last_neg = None, None, None
    for ep in range(1000):
        model.train()
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)
            pos_sim = (z1 * z2).sum(1).mean().item()
            neg_sim = torch.mm(z1, z2.T).mean().item()
            optim.zero_grad(); loss.backward(); clip_grad_norm_(model.parameters(), 1.0); optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        sched.step()
        print(f"{tag} Ep {ep+1}: loss {avg_loss:.4f}, pos {pos_sim:.4f}, neg {neg_sim:.4f}")
        last_loss, last_pos, last_neg = avg_loss, pos_sim, neg_sim
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, f"{tag.lower()}_backbone_64d_{window}frame.pt")
            )
            counter = 0
        else:
            counter += 1
        if counter >= patience: break
    # 윈도우별 최종 loss, pos_sim, neg_sim 반환
    return {'window': window, 'loss': last_loss, 'pos': last_pos, 'neg': last_neg}

if __name__ == '__main__':
    front_results = []
    back_results = []
    for window in [7, 9, 11, 13, 15]:
        front_results.append(run_pipeline('Front', 'feature_dataset/front_dataset_contrastive.npy', window))
        back_results.append(run_pipeline('Back',  'feature_dataset/back_dataset_contrastive.npy', window))
    print("\n64차원 대조 학습이 완료되었습니다.")
    # 최종 결과 정리 출력
    print("\n[최종 윈도우별 학습 결과: Front]")
    print("window\tloss\tpos\tneg")
    for res in front_results:
        print(f"{res['window']}\t{res['loss']:.4f}\t{res['pos']:.4f}\t{res['neg']:.4f}")
    print("\n[최종 윈도우별 학습 결과: Back]")
    print("window\tloss\tpos\tneg")
    for res in back_results:
        print(f"{res['window']}\t{res['loss']:.4f}\t{res['pos']:.4f}\t{res['neg']:.4f}")
