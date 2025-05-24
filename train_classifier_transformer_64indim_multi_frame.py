"""
classifier_train_transformer_val.py
===================
원본 모델 구조에 검증 추가 버전 - 트랜스포머 기반 분류기 학습
- 모델 구조 유지: 레이어 2개, 헤드 4개, 원본 그대로
- 검증 세트 분리: 80/20 비율로 훈련/검증 분리
- 검증 기반 최적화: 검증 손실 기준 모델 저장, 조기 종료

Run:
    python train_classifier_transformer_val.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_stgcn import STGCNContrastiveModel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import shutil
import subprocess
import sys
import random

SEED = 42
# ----- Config -----
OUTPUT_DIR = "model_stgcn_multi_frame"
EXTRA_OUTPUT_DIR = "D:\\ADSL_BSH_minton\\shuttleset\\video\\valid_dataset\\model_stgcn_multi_frame"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_CLS = 64
LR_CLS = 1e-4
EPOCHS_CLS = 300
PROJ_DIM = 64

# ----- Dataset for classifier -----
# 64차원 feature만 반환 (window size는 무시)
class ClassifierDataset(Dataset):
    def __init__(self, npy_path):
        data = np.load(npy_path, allow_pickle=True).item()
        self.features = torch.FloatTensor(data['relative_keypoints'])  # shape: (N, 64)
        self.labels = torch.LongTensor(data['label'])                  # shape: (N,)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----- Transformer Classifier (64차원 입력) -----
class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = PROJ_DIM
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim*4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(1)               # [batch, 1, PROJ_DIM]
        x = self.transformer_encoder(x)  # [batch, 1, PROJ_DIM]
        x = x.squeeze(1)                 # [batch, PROJ_DIM]
        return self.classifier(x)        # [batch, 2]

# ----- 검증 함수 -----
def validate(loader, backbone, model, criterion, device, get_confusion_matrix=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = backbone(x)
            out = model(feat)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = correct / total * 100
    
    # 항상 F1, precision, recall 계산
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    if get_confusion_matrix:
        return avg_loss, acc, f1, precision, recall, all_preds, all_labels
    return avg_loss, acc, f1, precision, recall

# ----- 신뢰도 점수 생성 함수 -----
def generate_confidence_scores(front_model, front_backbone, back_model, back_backbone, front_val_ds, back_val_ds, output_path):
    print("\n=== 검증 데이터셋 기준 신뢰도 점수 생성 중... ===")
    
    # 결과 저장용 데이터프레임 생성
    results = pd.DataFrame()

    # 모델을 평가 모드로 설정
    front_model.eval()
    back_model.eval()
    front_backbone.eval()
    back_backbone.eval()
    
    # 검증 데이터셋 로더 생성 (랜덤 셔플 없이)
    front_val_loader = DataLoader(front_val_ds, batch_size=1, shuffle=False)
    back_val_loader = DataLoader(back_val_ds, batch_size=1, shuffle=False)
    
    # 전체 front 데이터 수집
    front_data = []
    for x, y in front_val_loader:
        x = x.to(DEVICE)
        feat = front_backbone(x)
        output = front_model(feat)
        softmax = torch.softmax(output, dim=1)
        score = softmax[0, 1].item()
        front_data.append({
            'label': y.item(),
            'confidence': score
        })
    
    # 전체 back 데이터 수집
    back_data = []
    for x, y in back_val_loader:
        x = x.to(DEVICE)
        feat = back_backbone(x)
        output = back_model(feat)
        softmax = torch.softmax(output, dim=1)
        score = softmax[0, 1].item()
        back_data.append({
            'label': y.item(),
            'confidence': score
        })
    
    print(f"수집된 front 데이터: {len(front_data)}개")
    print(f"수집된 back 데이터: {len(back_data)}개")
    
    # front 데이터 라벨 분포 확인
    front_ones = sum(1 for d in front_data if d['label'] == 1)
    front_zeros = len(front_data) - front_ones
    print(f"Front 데이터 분포: 라벨 1 = {front_ones}개, 라벨 0 = {front_zeros}개")
    
    # back 데이터 라벨 분포 확인
    back_ones = sum(1 for d in back_data if d['label'] == 1)
    back_zeros = len(back_data) - back_ones
    print(f"Back 데이터 분포: 라벨 1 = {back_ones}개, 라벨 0 = {back_zeros}개")
    
    # 데이터 랜덤 셔플
    random.seed(42)  # 재현성을 위한 시드 설정
    random.shuffle(front_data)
    random.shuffle(back_data)
    
    # 결과 데이터 준비
    combined_gt_labels = []
    combined_front_scores = []
    combined_back_scores = []
    
    # 패턴별 카운트
    pattern_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    
    # 랜덤하게 매칭된 데이터로 GT 생성 (더 작은 데이터셋 크기만큼 사용)
    pairs_count = min(len(front_data), len(back_data))
    for i in range(pairs_count):
        front_label = front_data[i]['label']
        back_label = back_data[i]['label']
        
        # 패턴 카운트
        pattern = f"{front_label}{back_label}"
        pattern_counts[pattern] += 1
        
        # 둘 다 1인 경우 제외
        if front_label == 1 and back_label == 1:
            continue
        
        # GT 레이블 결정: 둘 중 하나라도 1이면 GT=1, 둘 다 0이면 GT=0
        gt_label = 1 if (front_label == 1 or back_label == 1) else 0
        
        # 결과에 추가
        combined_gt_labels.append(gt_label)
        combined_front_scores.append(front_data[i]['confidence'])
        combined_back_scores.append(back_data[i]['confidence'])
    
    # 패턴 분포 출력
    print(f"랜덤 매칭 패턴 분포:")
    for pattern, count in pattern_counts.items():
        print(f"  패턴 {pattern}: {count}개 ({count/pairs_count*100:.1f}%)")
    
    # 결과 데이터프레임에 저장
    results['is_shot_gt'] = combined_gt_labels
    results['front_confidence'] = combined_front_scores
    results['back_confidence'] = combined_back_scores
    
    # 각 레이블 조합 개수 출력
    print(f"최종 데이터 개수: {len(combined_gt_labels)}개")
    print(f"GT=1인 데이터 개수: {sum(combined_gt_labels)}개")
    print(f"GT=0인 데이터 개수: {len(combined_gt_labels) - sum(combined_gt_labels)}개")
    print(f"제외된 데이터 개수 (front=1, back=1): {pattern_counts['11']}개")
    
    # 결과 저장 (기본 경로)
    results.to_csv(output_path, sep='\t', index=False)
    print(f"검증 데이터셋 기준 신뢰도 점수가 {output_path}에 저장되었습니다.")
    
    # 추가 경로에도 저장
    extra_dir = os.path.dirname(EXTRA_OUTPUT_DIR)
    os.makedirs(extra_dir, exist_ok=True)
    
    extra_output_path = os.path.join(EXTRA_OUTPUT_DIR, os.path.basename(output_path))
    results.to_csv(extra_output_path, sep='\t', index=False)
    print(f"검증 데이터셋 기준 신뢰도 점수가 추가로 {extra_output_path}에 저장되었습니다.")
    
    # 모델 파일도 추가 경로에 복사
    window = output_path.split('_')[-2].replace('frame', '')
    front_model_src = os.path.join(OUTPUT_DIR, f"front_{window}frame_classifier_transformer_val_64_{window}frame_final.pt")
    back_model_src = os.path.join(OUTPUT_DIR, f"back_{window}frame_classifier_transformer_val_64_{window}frame_final.pt")
    
    front_model_dst = os.path.join(EXTRA_OUTPUT_DIR, f"front_{window}frame_classifier_transformer_val_64_{window}frame_final.pt")
    back_model_dst = os.path.join(EXTRA_OUTPUT_DIR, f"back_{window}frame_classifier_transformer_val_64_{window}frame_final.pt")
    
    if os.path.exists(front_model_src):
        shutil.copy2(front_model_src, front_model_dst)
        print(f"Front 모델 파일이 {front_model_dst}에 복사되었습니다.")
    
    if os.path.exists(back_model_src):
        shutil.copy2(back_model_src, back_model_dst)
        print(f"Back 모델 파일이 {back_model_dst}에 복사되었습니다.")

# ----- Training Function -----
def train_classifier(tag, npy_path, backbone_path, window):
    print(f"\n=== {tag} TRANSFORMER CLASSIFIER TRAINING (WITH VALIDATION, {window}frame) ===")
    # Load dataset
    ds = ClassifierDataset(npy_path)
    
    # 훈련/검증 데이터셋 분리 (8:2 비율)
    train_size = int(0.7 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_CLS, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_CLS)
    
    print(f"데이터셋 분할: 훈련 {train_size}개, 검증 {val_size}개")

    # Load pretrained backbone and freeze
    backbone = STGCNContrastiveModel(PROJ_DIM).to(DEVICE)
    backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # Initialize Transformer classifier
    clf = TransformerClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=LR_CLS, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    patience = 15
    wait = 0
    
    # 학습 진행 기록용 리스트
    epoch_nums = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    val_precisions = []
    val_recalls = []

    # 메트릭 저장 파일 경로
    metrics_path = os.path.join(OUTPUT_DIR, f"{tag.lower()}_validation_metrics_{window}frame.csv")
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "val_precision", "val_recall"])

    # 텍스트 파일 형식으로도 저장
    txt_path = os.path.join(OUTPUT_DIR, f"{tag.lower()}_validation_metrics_{window}frame.txt")
    with open(txt_path, 'w') as f:
        f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tval_f1\tval_precision\tval_recall\n")

    for ep in range(EPOCHS_CLS):
        # 훈련 단계
        clf.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                feat = backbone(x)  # (B, 64)
            out = clf(feat)        # (B, 2)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100
        
        # 검증 단계
        val_loss, val_acc, val_f1, val_precision, val_recall = validate(val_loader, backbone, clf, criterion, DEVICE)
        
        print(f"{tag} Ep {ep+1}: train_loss {train_loss:.4f}, train_acc {train_acc:.2f}%, val_loss {val_loss:.4f}, val_acc {val_acc:.2f}%, f1 {val_f1:.4f}, prec {val_precision:.4f}, recall {val_recall:.4f}")
        
        # 진행 상황 기록
        epoch_nums.append(ep+1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        
        # CSV 형식으로 메트릭 저장
        metrics_df.loc[len(metrics_df)] = [
            ep+1, train_loss, train_acc, val_loss, val_acc, val_f1, val_precision, val_recall
        ]
        metrics_df.to_csv(metrics_path, index=False)
        
        # 텍스트 파일 형식으로도 저장
        with open(txt_path, 'a') as f:
            f.write(f"{ep+1}\t{train_loss:.6f}\t{train_acc:.6f}\t{val_loss:.6f}\t{val_acc:.6f}\t{val_f1:.6f}\t{val_precision:.6f}\t{val_recall:.6f}\n")
        
        # 학습률 스케줄러 업데이트 (검증 손실 기준)
        scheduler.step(val_loss)

        # 모델 저장 (검증 손실 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            save_path = os.path.join(OUTPUT_DIR, f"{tag.lower()}_classifier_transformer_val_64_{window}frame_final.pt")
            torch.save(clf.state_dict(), save_path)
            print(f"Saved best {tag} classifier: {save_path} (val_loss: {val_loss:.4f})")
            
            # 추가 경로에도 모델 저장
            os.makedirs(EXTRA_OUTPUT_DIR, exist_ok=True)
            extra_save_path = os.path.join(EXTRA_OUTPUT_DIR, f"{tag.lower()}_classifier_transformer_val_64_{window}frame_final.pt")
            torch.save(clf.state_dict(), extra_save_path)
            print(f"Saved best {tag} classifier to extra path: {extra_save_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping {tag}, no improvement for {patience} epochs.")
                break
    
    # 메트릭 최종 저장 (두 번째 경로)
    extra_metrics_path = os.path.join(EXTRA_OUTPUT_DIR, f"{tag.lower()}_validation_metrics_{window}frame.csv")
    metrics_df.to_csv(extra_metrics_path, index=False)
    
    extra_txt_path = os.path.join(EXTRA_OUTPUT_DIR, f"{tag.lower()}_validation_metrics_{window}frame.txt")
    with open(txt_path, 'r') as src, open(extra_txt_path, 'w') as dst:
        dst.write(src.read())
    
    print(f"검증 메트릭이 저장되었습니다:")
    print(f"- CSV: {metrics_path} 및 {extra_metrics_path}")
    print(f"- TXT: {txt_path} 및 {extra_txt_path}")
    
    # 최종 모델 평가
    print(f"\n{tag} 최종 모델 성능 평가...")
    clf.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{tag.lower()}_classifier_transformer_val_64_{window}frame_final.pt"), map_location=DEVICE))
    val_loss, val_acc, val_f1, val_precision, val_recall, _, _ = validate(val_loader, backbone, clf, criterion, DEVICE, get_confusion_matrix=True)
    print(f"{tag} 최종 모델: val_loss {val_loss:.4f}, val_acc {val_acc:.2f}%, f1 {val_f1:.4f}, precision {val_precision:.4f}, recall {val_recall:.4f}")
    
    # 혼동 행렬 생성 없이 최종 성능 저장
    final_metrics = {
        "model": tag,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall
    }
    final_metrics_path = os.path.join(OUTPUT_DIR, f"{tag.lower()}_final_metrics_{window}frame.csv")
    pd.DataFrame([final_metrics]).to_csv(final_metrics_path, index=False)
    
    extra_final_metrics_path = os.path.join(EXTRA_OUTPUT_DIR, f"{tag.lower()}_final_metrics_{window}frame.csv")
    pd.DataFrame([final_metrics]).to_csv(extra_final_metrics_path, index=False)
    
    return clf, backbone, val_ds

# ----- Main -----
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXTRA_OUTPUT_DIR, exist_ok=True)
    
    """
    for window in [3, 5, 7, 9, 11, 13, 15]:
        train_classifier(
            tag=f'front_{window}frame',
            npy_path='feature_dataset/front_dataset_contrastive.npy',
            backbone_path=os.path.join(OUTPUT_DIR, f'front_backbone_64d_{window}frame.pt'),
            window=window
        )
        train_classifier(
            tag=f'back_{window}frame',
            npy_path='feature_dataset/back_dataset_contrastive.npy',
            backbone_path=os.path.join(OUTPUT_DIR, f'back_backbone_64d_{window}frame.pt'),
            window=window
        )
    """
    # 15프레임만 학습
    window = 15
    front_model, front_backbone, front_val_ds = train_classifier(
        tag=f'front_{window}frame',
        npy_path='feature_dataset/front_dataset_contrastive.npy',
        backbone_path=os.path.join(OUTPUT_DIR, f'front_backbone_64d_{window}frame.pt'),
        window=window
    )
    back_model, back_backbone, back_val_ds = train_classifier(
        tag=f'back_{window}frame',
        npy_path='feature_dataset/back_dataset_contrastive.npy',
        backbone_path=os.path.join(OUTPUT_DIR, f'back_backbone_64d_{window}frame.pt'),
        window=window
    )
    
    # 검증 데이터셋 기준 신뢰도 점수 생성 및 저장
    generate_confidence_scores(
        front_model, front_backbone,
        back_model, back_backbone,
        front_val_ds, back_val_ds,
        os.path.join(OUTPUT_DIR, f'shot_confidence_val_{window}frame_final.tsv')
    )
    
    print("\n64차원 백본 기반 트랜스포머 분류기 15프레임만 학습 완료!")
    
    # 첫 번째 스크립트 실행 완료 후 두 번째 스크립트 실행
    print("\n첫 번째 스크립트 실행 완료. 이제 두 번째 스크립트를 실행합니다...\n")
    validation_script_path = "D:\\ADSL_BSH_minton\\shuttleset\\video\\valid_dataset\\main_stgcn_transformer_validation.py"
    
    try:
        # 파이썬 인터프리터를 사용하여 스크립트 실행
        python_executable = sys.executable  # 현재 실행 중인 파이썬 인터프리터 경로
        subprocess.run([python_executable, validation_script_path], check=True)
        print("\n두 번째 스크립트 실행이 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"\n두 번째 스크립트 실행 중 오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}") 