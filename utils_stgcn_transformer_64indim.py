import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from model_stgcn import STGCNContrastiveModel
from ultralytics import YOLO
import math
import onepose
# ─── ST-GCN 기반 Shot Detector Utility ────────────────────────────────────
class STGCNShotDetector:
    def __init__(
        self,
        model_dir: str = "model_stgcn_backbone_classifier",
        proj_dim: int = 64,
        frame_size: int = 15,
        threshold: float = 0.5,
        device: torch.device = None
    ):
        self.device     = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_size = frame_size
        self.threshold  = threshold

        # 백본 로드
        self.backbone = {
            'front': STGCNContrastiveModel(proj_dim).to(self.device),
            'back':  STGCNContrastiveModel(proj_dim).to(self.device)
        }
        self.classifier = {}

        # TransformerClassifier 구조 정의 
        class TransformerClassifier(nn.Module):
            def __init__(self, proj_dim):
                super().__init__()
                self.hidden_dim = proj_dim
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
            def forward(self, z):
                z = z.unsqueeze(1)                  # [batch, 1, PROJ_DIM]
                z = self.transformer_encoder(z)     # [batch, 1, PROJ_DIM]
                z = z.squeeze(1)                    # [batch, PROJ_DIM]
                return self.classifier(z)           # [batch, 2]

        # front classifier 불러오기
        self.backbone['front'].load_state_dict(
            torch.load(os.path.join(model_dir, "front_backbone_64d.pt"), map_location=self.device)
        )
        self.classifier['front'] = TransformerClassifier(proj_dim).to(self.device)
        self.classifier['front'].load_state_dict(
            torch.load(os.path.join(model_dir, "front_classifier_transformer_val_64.pt"), map_location=self.device)
        )

        # back classifier 불러오기
        self.backbone['back'].load_state_dict(
            torch.load(os.path.join(model_dir, "back_backbone_64d.pt"), map_location=self.device)
        )
        self.classifier['back'] = TransformerClassifier(proj_dim).to(self.device)
        self.classifier['back'].load_state_dict(
            torch.load(os.path.join(model_dir, "back_classifier_transformer_val_64.pt"), map_location=self.device)
        )

        # eval 모드
        for m in self.backbone.values(): m.eval()
        for m in self.classifier.values(): m.eval()

    def detect(self, keypoint_sequences: dict) -> dict:
        results = {}
        for pid, seq in keypoint_sequences.items():
            x = torch.FloatTensor(seq).view(self.frame_size, -1).unsqueeze(0).to(self.device)
            side = 'back' if pid in (1,2) else 'front'
            with torch.no_grad():
                z = self.backbone[side](x)        # (1, proj_dim)
                logits = self.classifier[side](z)  # (1, 2)
            prob = logits.softmax(dim=1)[0,1].item()  # 클래스 1 (shot) 확률
            results[pid] = {
                'is_shot':    prob >= self.threshold,
                'confidence': prob
            }
        return results


# Load models
court_model = YOLO("best_court.pt")  # Court detection model
player_model = YOLO("yolo11x.pt")    # Player/shuttlecock detection model

def extract_keypoints(frame, box, pose_model):
    """
    바운딩 박스 영역에서 포즈 키포인트 추출
    
    Args:
        frame: 입력 프레임
        box: 바운딩 박스 좌표 [x1, y1, x2, y2]
        pose_model: ViTPose 모델
    
    Returns:
        keypoints: (17, 2) 형태의 키포인트 좌표
        confidence: 키포인트 신뢰도 점수
    """
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # 바운딩 박스가 유효한지 확인
        if x2 <= x1 or y2 <= y1:
            return None, None

        # 바운딩 박스 영역 추출
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None, None

        # RGB로 변환
        person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        
        # ViTPose로 포즈 추정
        with torch.no_grad():
            output = pose_model(person_roi)
        
        # 키포인트와 confidence 추출
        if isinstance(output, dict):
            keypoints = np.array(output.get('points', []))
            confidence = output.get('scores')
            
            if len(keypoints) > 0:
                # 키포인트 좌표를 원본 프레임 좌표로 변환
                keypoints[:, 0] += x1
                keypoints[:, 1] += y1
                return keypoints, confidence
    
    except Exception as e:
        print(f"포즈 추정 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None

def calculate_distance_from_center(point, center):
    """점과 중심점 사이의 거리 계산"""
    return math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)

def get_corners_from_bbox(box):
    """바운딩 박스에서 4개의 꼭지점 좌표 추출"""
    x1, y1, x2, y2 = map(int, box)
    return [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

def get_quadrant(point, center):
    """점이 위치한 사분면 반환 (1, 2, 3, 4)"""
    x, y = point
    cx, cy = center
    
    if x >= cx and y < cy:  # 오른쪽 위
        return 1
    elif x < cx and y < cy:  # 왼쪽 위
        return 2
    elif x < cx and y >= cy:  # 왼쪽 아래
        return 3
    else:  # 오른쪽 아래
        return 4


def get_box_quadrant(box, center):
    """바운딩 박스의 중심점이 위치한 사분면 반환"""
    x1, y1, x2, y2 = map(int, box)
    box_center_x = (x1 + x2) // 2
    box_center_y = (y1 + y2) // 2
    return get_quadrant((box_center_x, box_center_y), center)

def get_extreme_corner(box, center, quadrant):
    """박스에서 중심점 기준으로 해당 사분면 방향으로 가장 극단적인 좌표 반환"""
    corners = get_corners_from_bbox(box)
    
    # 각 코너의 사분면 확인
    corner_quadrants = [get_quadrant(corner, center) for corner in corners]
    
    # 지정된 사분면에 있는 코너들 선택
    matching_corners = [corner for i, corner in enumerate(corners) if corner_quadrants[i] == quadrant]
    
    # 해당 사분면에 있는 코너가 없으면 모든 코너 중에서 선택
    if not matching_corners:
        # 대안: 특정 코너 기준으로 선택
        # 1: 우상단, 2: 좌상단, 3: 좌하단, 4: 우하단
        if quadrant == 1:
            # 우상단 방향으로 가장 극단적인 점 (x 최대, y 최소)
            return max(corners, key=lambda p: p[0] - p[1])
        elif quadrant == 2:
            # 좌상단 방향으로 가장 극단적인 점 (x 최소, y 최소)
            return min(corners, key=lambda p: p[0] + p[1])
        elif quadrant == 3:
            # 좌하단 방향으로 가장 극단적인 점 (x 최소, y 최대)
            return min(corners, key=lambda p: p[0] - p[1])
        else:  # quadrant == 4
            # 우하단 방향으로 가장 극단적인 점 (x 최대, y 최대)
            return max(corners, key=lambda p: p[0] + p[1])
    
    # 해당 사분면에 여러 코너가 있는 경우, 중심에서 가장 먼 코너 선택
    if len(matching_corners) > 1:
        return max(matching_corners, key=lambda p: calculate_distance_from_center(p, center))
    
    # 해당 사분면에 딱 하나의 코너만 있는 경우
    return matching_corners[0]

def get_missing_quadrant(quadrants):
    """누락된 사분면 번호 찾기"""
    all_quadrants = set([1, 2, 3, 4])
    return list(all_quadrants - set(quadrants))[0] if len(all_quadrants - set(quadrants)) == 1 else None

def estimate_fourth_point(points, missing_quadrant):
    """3개의 점에서 평행사변형의 원리로 4번째 점 추정"""
    if len(points) != 3:
        return None
    
    # 3개의 점 좌표
    p1, p2, p3 = points
    
    # 벡터 계산: 평행사변형의 원리 사용
    # p4 = p1 + (p3 - p2) 또는 p4 = p2 + (p3 - p1) 또는 p4 = p3 + (p2 - p1)
    # 세 점의 조합에 따라 결과가 달라집니다
    
    # 각 점이 어느 사분면에 있는지 확인
    # 이미 있는 좌표들의 사분면을 확인
    # 이 함수는 3개의 점이 이미 감지되었고, 하나만 누락되었다고 가정합니다
    
    # 누락된 사분면에 따라 적절한 계산 방식 선택
    if missing_quadrant == 1:  # 우상단이 누락
        # 좌상단 + (우하단 - 좌하단)
        # 또는 좌하단 + (좌상단 - 우하단) 등의 계산
        # 각 배열의 인덱스가 어느 사분면인지에 따라 다름
        # 여기서는 간략한 예시로 벡터 더하기 사용
        
        # 가정: p1=좌상단(2), p2=좌하단(3), p3=우하단(4)
        # 그러면 우상단(1) = 좌상단(2) + 우하단(4) - 좌하단(3)
        return (p1[0] + p3[0] - p2[0], p1[1] + p3[1] - p2[1])
    
    elif missing_quadrant == 2:  # 좌상단이 누락
        # 우상단 + (좌하단 - 우하단)
        # 가정: p1=우상단(1), p2=좌하단(3), p3=우하단(4)
        return (p1[0] + p2[0] - p3[0], p1[1] + p2[1] - p3[1])
    
    elif missing_quadrant == 3:  # 좌하단이 누락
        # 좌상단 + (우하단 - 우상단)
        # 가정: p1=좌상단(2), p2=우상단(1), p3=우하단(4)
        return (p1[0] + p3[0] - p2[0], p1[1] + p3[1] - p2[1])
    
    elif missing_quadrant == 4:  # 우하단이 누락
        # 우상단 + (좌하단 - 좌상단)
        # 가정: p1=좌상단(2), p2=우상단(1), p3=좌하단(3)
        return (p2[0] + p3[0] - p1[0], p2[1] + p3[1] - p1[1])
    
    return None

def find_extreme_corners(boxes, center, width, height):
    """
    각 사분면 방향으로 가장 극단적인 코너 포인트 찾기
    """
    if len(boxes) < 2:
        print(f"바운딩 박스가 {len(boxes)}개밖에 감지되지 않아 ROI 설정을 보류합니다 (최소 2개 필요)")
        return None
    
    # 각 사분면별 가장 극단적인 좌표 찾기
    extreme_corners = {1: None, 2: None, 3: None, 4: None}
    
    # 우선 각 박스의 사분면 확인
    box_quadrants = [get_box_quadrant(box, center) for box in boxes]
    unique_quadrants = set(box_quadrants)
    
    # 만약 대각선 위치에 박스가 있으면 해당 박스들에서 극단 코너 찾기
    has_diagonal = (1 in unique_quadrants and 3 in unique_quadrants) or \
                   (2 in unique_quadrants and 4 in unique_quadrants)
    
    # 2개인 경우 반드시 대각선에 위치해야 함
    if len(boxes) == 2 and not has_diagonal:
        print(f"바운딩 박스가 2개 있지만, 대각선에 위치하지 않아 ROI 설정을 보류합니다.")
        print(f"감지된 사분면: {sorted(list(unique_quadrants))}, 대각선 필요(1-3 또는 2-4)")
        return None
    
    # 최소 2개 이상의 서로 다른 위치에 있어야 함    
    if len(unique_quadrants) < 2:
        print(f"바운딩 박스 {len(boxes)}개가 모두 같은 사분면에 있어 ROI 설정을 보류합니다 (최소 2개 이상의 다른 위치 필요)")
        return None
    
    # 각 사분면별로 가장 극단적인 코너 찾기
    for quadrant in range(1, 5):
        # 해당 사분면에 있는 박스들
        boxes_in_quadrant = [box for i, box in enumerate(boxes) if box_quadrants[i] == quadrant]
        
        if boxes_in_quadrant:
            # 해당 사분면에 박스가 있으면, 각 박스에서 해당 방향으로 가장 극단적인 코너 찾기
            extreme_candidates = []
            for box in boxes_in_quadrant:
                extreme_candidates.append(get_extreme_corner(box, center, quadrant))
            
            # 그 중에서 가장 극단적인 코너 선택
            if extreme_candidates:
                if quadrant == 1:  # 우상단: x 최대, y 최소
                    extreme_corners[quadrant] = max(extreme_candidates, key=lambda p: p[0] - p[1])
                elif quadrant == 2:  # 좌상단: x 최소, y 최소
                    extreme_corners[quadrant] = min(extreme_candidates, key=lambda p: p[0] + p[1])
                elif quadrant == 3:  # 좌하단: x 최소, y 최대
                    extreme_corners[quadrant] = min(extreme_candidates, key=lambda p: p[0] - p[1])
                else:  # 우하단: x 최대, y 최대
                    extreme_corners[quadrant] = max(extreme_candidates, key=lambda p: p[0] + p[1])
    
    # 사분면별 바운딩 박스 개수
    filled_quadrants = [q for q in range(1, 5) if extreme_corners[q] is not None]
    filled_count = len(filled_quadrants)
    
    # 상황별 처리
    if filled_count == 4:
        # 4개의 사분면 모두에 바운딩 박스가 있는 이상적인 경우
        print(f"4개의 사분면 모두에 바운딩 박스가 있습니다. 이상적인 ROI를 설정합니다.")
    elif filled_count == 3:
        # 3개의 사분면에만 바운딩 박스가 있는 경우, 4번째 점을 추론
        missing_quadrant = get_missing_quadrant(filled_quadrants)
        
        if missing_quadrant:
            print(f"3개의 사분면에 바운딩 박스가 감지되었습니다. 사분면 {missing_quadrant}의 점을 추론합니다.")
            
            # 기존 3개 점의 좌표
            existing_points = [extreme_corners[q] for q in filled_quadrants]
            
            # 4번째 점 추론
            fourth_point = estimate_fourth_point(existing_points, missing_quadrant)
            
            if fourth_point:
                # 추론된 좌표가 프레임 내에 있도록 조정
                x, y = fourth_point
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                extreme_corners[missing_quadrant] = (int(x), int(y))
                print(f"추론된 좌표 (사분면 {missing_quadrant}): {extreme_corners[missing_quadrant]}")
    elif filled_count == 2:
        # 2개의 사분면에만 바운딩 박스가 있는 경우 - 대각선 확인 (이미 위에서 필터링됨)
        diag1 = (1 in filled_quadrants and 3 in filled_quadrants)  # 1-3 사분면(우상단-좌하단)
        diag2 = (2 in filled_quadrants and 4 in filled_quadrants)  # 2-4 사분면(좌상단-우하단)
        
        if diag1 or diag2:
            print(f"대각선에 위치한 2개의 바운딩 박스로 ROI를 설정합니다. 사분면: {filled_quadrants}")
            
            # 누락된 사분면의 좌표는 프레임 경계를 사용하거나 보간
            # 이 경우에는 상단 좌/우와 하단 좌/우 중 대각선에 해당하지 않는 두 점을 적절히 설정해야 함
            if diag1:  # 1-3 사분면(우상단-좌하단) -> 2(좌상단), 4(우하단) 채우기 필요
                # 좌상단(2)은 우상단(1)의 x좌표와 좌하단(3)의 y좌표 조합
                if 2 not in filled_quadrants:
                    pt1 = extreme_corners[1]  # 우상단
                    pt3 = extreme_corners[3]  # 좌하단
                    extreme_corners[2] = (pt3[0], pt1[1])  # 좌상단 = (좌하단x, 우상단y)
                    
                # 우하단(4)은 우상단(1)의 x좌표와 좌하단(3)의 y좌표 조합
                if 4 not in filled_quadrants:
                    pt1 = extreme_corners[1]  # 우상단
                    pt3 = extreme_corners[3]  # 좌하단
                    extreme_corners[4] = (pt1[0], pt3[1])  # 우하단 = (우상단x, 좌하단y)
            
            elif diag2:  # 2-4 사분면(좌상단-우하단) -> 1(우상단), 3(좌하단) 채우기 필요
                # 우상단(1)은 좌상단(2)의 y좌표와 우하단(4)의 x좌표 조합
                if 1 not in filled_quadrants:
                    pt2 = extreme_corners[2]  # 좌상단
                    pt4 = extreme_corners[4]  # 우하단
                    extreme_corners[1] = (pt4[0], pt2[1])  # 우상단 = (우하단x, 좌상단y)
                    
                # 좌하단(3)은 좌상단(2)의 x좌표와 우하단(4)의 y좌표 조합
                if 3 not in filled_quadrants:
                    pt2 = extreme_corners[2]  # 좌상단
                    pt4 = extreme_corners[4]  # 우하단
                    extreme_corners[3] = (pt2[0], pt4[1])  # 좌하단 = (좌상단x, 우하단y)
        else:
            # 이 경우는 위에서 필터링되었으므로 여기에 오지 않음
            pass
    
    # 여전히 빈 사분면이 있으면 프레임 경계 사용
    for quadrant in range(1, 5):
        if extreme_corners[quadrant] is None:
            if quadrant == 1:
                extreme_corners[quadrant] = (width, 0)  # 우상단
            elif quadrant == 2:
                extreme_corners[quadrant] = (0, 0)  # 좌상단
            elif quadrant == 3:
                extreme_corners[quadrant] = (0, height)  # 좌하단
            else:
                extreme_corners[quadrant] = (width, height)  # 우하단
    
    # 좌표 나열 (시계방향)
    ordered_corners = [
        extreme_corners[2],  # 좌상단
        extreme_corners[1],  # 우상단
        extreme_corners[4],  # 우하단
        extreme_corners[3],  # 좌하단
    ]
    
    return ordered_corners

def is_overlapping(bbox, roi_polygon):
    """바운딩 박스와 ROI 폴리곤이 겹치는지 확인 - 조금이라도 겹치면 True 반환"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # 다각형 포인트를 정수로 변환
    roi_polygon_int = np.array(roi_polygon, dtype=np.int32)
    
    # 바운딩 박스가 이미지 내에 있는지 확인
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False
    
    # ROI 내부에 바운딩 박스가 완전히 포함되는지 확인
    mask = np.zeros((max(y2, roi_polygon_int[:, 1].max()) + 1, max(x2, roi_polygon_int[:, 0].max()) + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon_int], 255)
    
    # 바운딩 박스 영역의 마스크 생성
    bbox_mask = np.zeros_like(mask)
    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
    
    # 두 마스크의 교집합 확인
    intersection = cv2.bitwise_and(mask, bbox_mask)
    if cv2.countNonZero(intersection) > 0:
        return True
    
    return False

def line_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    """두 선분이 교차하는지 확인"""
    def ccw(p1, p2, p3):
        # 세 점의 방향성 확인 (반시계방향이면 양수, 시계방향이면 음수, 일직선이면 0)
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    # 두 선분이 교차하려면 각 선분을 기준으로 다른 선분의 끝점이 반대편에 위치해야 함
    ccw1 = ccw(line1_p1, line1_p2, line2_p1) * ccw(line1_p1, line1_p2, line2_p2)
    ccw2 = ccw(line2_p1, line2_p2, line1_p1) * ccw(line2_p1, line2_p2, line1_p2)
    
    # 둘 다 0 이하(방향이 다르거나 일직선)이면 교차
    return ccw1 <= 0 and ccw2 <= 0

def draw_roi(frame, roi_points, detection_boxes=None):
    """
    Visualize ROI - 트래킹 중인 객체가 ROI와 겹치는 부분은 원래 밝기 유지 (최적화 버전)
    
    Args:
        frame: 원본 프레임
        roi_points: ROI 포인트 배열
        detection_boxes: 트래킹 중인 객체 바운딩 박스 (x1, y1, x2, y2)
    """
    height, width = frame.shape[:2]
    
    # 1. 결과 프레임 준비 (처음에는 완전히 어두운 프레임으로 시작)
    darkened_frame = np.zeros_like(frame)
    
    # 2. ROI 마스크 생성 (ROI 내부는 255, 외부는 0)
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points], 255)
    
    # 3. 어두운 배경 생성 (원본 프레임의 20%만 사용)
    dark_background = cv2.multiply(frame, 0.2).astype(np.uint8)
    
    # 4. ROI 내부는 원본 프레임 사용, 외부는 어두운 배경 사용
    # ROI 영역 마스크 적용
    roi_region = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # 반전된 ROI 마스크 생성 (ROI 외부는 255, 내부는 0)
    inv_roi_mask = cv2.bitwise_not(roi_mask)
    
    # ROI 외부 영역에 어두운 배경 적용
    outside_roi = cv2.bitwise_and(dark_background, dark_background, mask=inv_roi_mask)
    
    # ROI 내부와 외부 합치기
    result = cv2.add(roi_region, outside_roi)
    
    # 5. 객체 바운딩 박스 영역은 원래 밝기로 복원
    if detection_boxes is not None and len(detection_boxes) > 0:
        # 바운딩 박스 영역만 마스크 생성
        box_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 각 바운딩 박스를 마스크에 그리기
        for box in detection_boxes:
            try:
                x1, y1, x2, y2 = map(int, box[:4])  # 첫 4개 값만 사용
                # 이미지 경계 확인
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width-1, x2), min(height-1, y2)
                if x2 > x1 and y2 > y1:  # 유효한 박스만 처리
                    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            except Exception as e:
                # 에러가 있는 경우 무시하고 계속 진행
                continue
        
        # 바운딩 박스와 ROI 외부가 겹치는 부분 찾기
        box_outside_roi = cv2.bitwise_and(box_mask, inv_roi_mask)
        
        # 해당 영역을 원래 밝기로 복원
        box_region = cv2.bitwise_and(frame, frame, mask=box_outside_roi)
        
        # 결과에 합치기 (ROI 외부의 바운딩 박스 영역은 원래 밝기)
        result = cv2.add(result, cv2.subtract(box_region, cv2.bitwise_and(outside_roi, outside_roi, mask=box_outside_roi)))
    
    # 6. ROI 영역에 반투명 녹색 오버레이
    overlay = result.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 200, 0))  # Green fill
    cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)  # 80% transparency
    
    # 7. ROI 경계선 그리기 (굵게)
    cv2.polylines(result, [roi_points], True, (0, 255, 0), 4)
    
    # 8. ROI 꼭짓점 표시
    for i, point in enumerate(roi_points):
        cv2.circle(result, tuple(point), 10, (0, 0, 0), -1)  # 배경
        cv2.circle(result, tuple(point), 7, (0, 0, 255), -1)  # 점
        cv2.putText(result, str(i+1), (point[0]+10, point[1]+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 9. ROI 정보 추가
    area = cv2.contourArea(roi_points)
    cv2.putText(result, f"ROI Area: {area:.0f}px²", (width-280, height-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 10. 중심점 표시
    center = (width // 2, int(height * 2/3))
    cv2.circle(result, center, 5, (255, 255, 255), -1)
    
    for point in roi_points:
        cv2.line(result, center, tuple(point), (100, 100, 255), 1)
    
    return result

def draw_court_detection(frame, boxes, confidences=None, classes=None):
    """Visualize court detection results"""
    result_frame = frame.copy()
    
    # Display number of detected bounding boxes
    cv2.putText(result_frame, f"Detected Court Boxes: {len(boxes)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display center point
    height, width = frame.shape[:2]
    center = (width // 2, int(height * 2/3))  # Adjusted to 2/3 height (1/3 from bottom)
    cv2.circle(result_frame, center, 10, (255, 255, 255), -1)  # Center point
    cv2.line(result_frame, (0, center[1]), (width, center[1]), (100, 100, 100), 1)  # Horizontal line
    cv2.line(result_frame, (center[0], 0), (center[0], height), (100, 100, 100), 1)  # Vertical line
    
    # Add text explaining center point position
    cv2.putText(result_frame, f"Center: 1/3 from bottom", (center[0] - 120, center[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Quadrant labels
    cv2.putText(result_frame, "Q2", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(result_frame, "Q1", (width-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(result_frame, "Q3", (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(result_frame, "Q4", (width-50, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Check quadrant distribution of bounding boxes
    if len(boxes) >= 2:
        center = (width // 2, height // 2)
        box_quadrants = [get_box_quadrant(box, center) for box in boxes]
        unique_quadrants = set(box_quadrants)
        has_diagonal = (1 in unique_quadrants and 3 in unique_quadrants) or \
                      (2 in unique_quadrants and 4 in unique_quadrants)
        
        distribution_text = "Diagonal distribution: "
        distribution_text += "✓" if has_diagonal else "✗"
        distribution_text += f" (Quadrants: {sorted(list(unique_quadrants))})"
        
        color = (0, 255, 0) if has_diagonal else (0, 0, 255)
        cv2.putText(result_frame, distribution_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw each bounding box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Check box quadrant
        box_quadrant = get_box_quadrant(box, center)
        
        # Set color (different for each quadrant)
        quadrant_colors = {
            1: (255, 0, 0),     # Q1: Red
            2: (0, 255, 0),     # Q2: Green
            3: (0, 0, 255),     # Q3: Blue
            4: (255, 255, 0)    # Q4: Yellow
        }
        color = quadrant_colors[box_quadrant]
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Display confidence and quadrant
        if confidences is not None and i < len(confidences):
            conf = confidences[i]
            label = f"Box {i+1}(Q{box_quadrant}): {conf:.2f}"
            if classes is not None and i < len(classes):
                label += f" (Class: {classes[i]})"
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(result_frame, f"Box {i+1}(Q{box_quadrant})", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw corners (emphasize extreme coordinates)
        corners = get_corners_from_bbox(box)
        for j, corner in enumerate(corners):
            corner_quadrant = get_quadrant(corner, center)
            # Check if this is an extreme corner for its quadrant
            extreme_corner = get_extreme_corner(box, center, corner_quadrant)
            is_extreme = corner == extreme_corner
            
            # Emphasize extreme corners
            size = 7 if is_extreme else 4
            thickness = -1 if is_extreme else 1
            
            cv2.circle(result_frame, corner, size, color, thickness)
            
            # Display corner number and quadrant
            if is_extreme:
                cv2.putText(result_frame, f"{j+1}(Q{corner_quadrant})*", (corner[0]+5, corner[1]+5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                cv2.putText(result_frame, f"{j+1}(Q{corner_quadrant})", (corner[0]+5, corner[1]+5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Preview ROI with extreme corners (when at least 2 boxes detected)
    if len(boxes) >= 2:
        # Try to find extreme corners
        corners = find_extreme_corners(boxes, center, width, height)
        if corners is not None:
            # Show preview ROI (thinner and semi-transparent)
            roi_points = np.array(corners, dtype=np.int32)
            cv2.polylines(result_frame, [roi_points], True, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Mark each corner
            for i, point in enumerate(corners):
                cv2.circle(result_frame, point, 5, (0, 255, 255), -1)
                cv2.putText(result_frame, f"ROI{i+1}", (point[0]-20, point[1]-10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return result_frame

def init_pose_model():
    """ViTPose 모델 초기화"""
    try:
        # ViTPose 모델 생성
        pose_model = onepose.create_model('ViTPose_large_simple_coco')
        if torch.cuda.is_available():
            pose_model = pose_model.cuda()
        pose_model.eval()
        print("ViTPose 모델이 성공적으로 초기화되었습니다.")
        return pose_model
    except Exception as e:
        print(f"ViTPose 모델 초기화 중 오류 발생: {e}")
        return None

def draw_pose(frame, keypoints, confidence, color):
    """
    포즈 키포인트와 스켈레톤 시각화 - confidence threshold와 최소 키포인트 개수 적용
    
    Args:
        frame: 입력 프레임
        keypoints: 키포인트 좌표 배열
        confidence: 키포인트 신뢰도 배열
        color: 스켈레톤 색상 (B, G, R)
    """
    # COCO 키포인트 연결 (스켈레톤)
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12],  # 다리
        [11, 12], [5, 11], [6, 12],              # 엉덩이 & 어깨
        [5, 6], [5, 7], [6, 8],                  # 상체 & 팔
        [7, 9], [8, 10],                         # 팔뚝
        [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 얼굴
        [3, 5], [4, 6]                           # 귀-어깨
    ]
    
    # Confidence threshold 설정
    CONF_THRESHOLD = 0.3  # 이 값을 조정하여 threshold 변경 가능
    MIN_KEYPOINTS = 17
    
    try:
        # confidence가 None이 아닌 경우에만 threshold 적용
        valid_keypoints = []
        if confidence is not None and not isinstance(confidence[0], type(None)):
            valid_keypoints = [(i, kp) for i, (kp, conf) in enumerate(zip(keypoints, confidence)) 
                             if conf is not None and conf > CONF_THRESHOLD]
        else:
            # confidence가 없는 경우 모든 키포인트 사용
            valid_keypoints = list(enumerate(keypoints))
        
        # 최소 키포인트 개수 체크
        if len(valid_keypoints) < MIN_KEYPOINTS:
            return
        
        # 스켈레톤 그리기 (관절 연결)
        for pair in skeleton:
            # 양쪽 키포인트의 인덱스가 valid_keypoints에 있는지 확인
            point1 = next((p for i, p in valid_keypoints if i == pair[0]), None)
            point2 = next((p for i, p in valid_keypoints if i == pair[1]), None)
            
            if point1 is not None and point2 is not None:
                pt1 = tuple(map(int, point1[:2]))
                pt2 = tuple(map(int, point2[:2]))
                cv2.line(frame, pt1, pt2, color, 2)
        
        # 키포인트 그리기
        for _, point in valid_keypoints:
            x, y = map(int, point[:2])
            cv2.circle(frame, (x, y), 3, color, -1)
        
        # 감지된 키포인트 수 표시 (디버그용)
        if valid_keypoints:
            first_point = valid_keypoints[0][1]
            cv2.putText(frame, f"KP: {len(valid_keypoints)}", 
                       (int(first_point[0]), int(first_point[1]-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    except Exception as e:
        print(f"포즈 그리기 오류: {e}")
        import traceback
        traceback.print_exc()

def process_pose_estimation(frame, boxes, pose_model, track_colors, id_mapping, shot_threshold=0.1, frame_size=15):
    """
    추적된 선수들의 포즈 추정 및 샷 감지 수행
    """
    if pose_model is None:
        print("포즈 모델이 초기화되지 않았습니다.")
        return frame

    image_shape = frame.shape

    # 버퍼 초기화
    if not hasattr(process_pose_estimation, 'keypoint_buffers'):
        process_pose_estimation.keypoint_buffers = {}
        process_pose_estimation.shot_frames      = {}
        process_pose_estimation.active_shots     = {}
        process_pose_estimation.frame_count      = 0

    process_pose_estimation.frame_count += 1

    # 이번 프레임에 감지된 front/back 선수 ID 수집
    for box, displayed_id in zip(boxes, id_mapping.values()):
        kp, conf = extract_keypoints(frame, box, pose_model)
        if kp is None: 
            continue
        draw_pose(frame, kp, conf, track_colors[displayed_id-1])
        rel_kp = keypoints_to_relative_vector([kp], image_shape)
        if rel_kp is None:
            continue
        process_pose_estimation.keypoint_buffers.setdefault(displayed_id, []).append(rel_kp)

    buffered_front = [pid for pid in (3,4)
                      if len(process_pose_estimation.keypoint_buffers.get(pid, [])) >= frame_size]
    buffered_back  = [pid for pid in (1,2)
                      if len(process_pose_estimation.keypoint_buffers.get(pid, [])) >= frame_size]

    # 모든 선수를 각각 독립적으로 처리
    all_players = buffered_front + buffered_back
    shot_candidates = []
    player_confidences = {}  # 모든 플레이어의 confidence 점수 저장

    if all_players:
        if not hasattr(process_pose_estimation, 'shot_detector'):
            process_pose_estimation.shot_detector = STGCNShotDetector(
                model_dir="model_stgcn_final",
                frame_size=frame_size,
                threshold=shot_threshold
            )
            print("샷 감지기가 초기화되었습니다.")
        
        # player_id → 실제 박스 매핑
        displayed_box_map = {
            disp_id: boxes[i]
            for i, disp_id in enumerate(id_mapping.values())
        }

        for player_id in all_players:
            player_seq = np.stack(process_pose_estimation.keypoint_buffers[player_id][-frame_size:])
            player_type = "Front" if player_id in (3,4) else "Back"
            #print(f"\n프레임 {process_pose_estimation.frame_count}: {player_type} {player_id} 샷 감지")
            
            # 한 명의 선수만 포함하는 딕셔너리 전달
            results = process_pose_estimation.shot_detector.detect({
                player_id: player_seq
            })
            
            # 결과 처리
            for pid, res in results.items():
                # 모든 결과 저장 (threshold와 관계없이)
                box = displayed_box_map.get(pid)
                player_confidences[pid] = res['confidence']
                
                if res['is_shot']:
                    shot_candidates.append((pid, res['confidence'], box))

        # 오래된 키포인트 삭제
        for pid, buf in process_pose_estimation.keypoint_buffers.items():
            if len(buf) > frame_size * 2:
                buf.pop(0)

    # 프레임 당 가장 높은 confidence 하나만 SHOT으로 강조 표시
        if shot_candidates:
            best_pid, best_conf, best_box = max(shot_candidates, key=lambda x: x[1])
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                shot_color = (0, 255, 255)  # 노란색 (BGR)
                # 테두리 효과: 먼저 검은색 두껍게, 그 위에 노란색 얇게
                cv2.putText(frame,
                            f"SHOT! ({best_conf:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 0), 4)  # 검은색 두껍게
                cv2.putText(frame,
                            f"SHOT! ({best_conf:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            shot_color, 2)  # 노란색 얇게
    
    # 모든 플레이어의 confidence 점수 표시
    for pid, conf in player_confidences.items():
        box = displayed_box_map.get(pid)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # 정보 표시 위치 조정 (y1-30으로 위치 조정하여 SHOT! 텍스트와 겹치지 않게)
            cv2.putText(frame,
                        f"Conf: {conf:.2f}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        track_colors[pid-1],
                        2)

    return frame
    
    # 모든 플레이어의 confidence 점수 표시
    for pid, conf in player_confidences.items():
        box = displayed_box_map.get(pid)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # 정보 표시 위치 조정 (y1-30으로 위치 조정하여 SHOT! 텍스트와 겹치지 않게)
            cv2.putText(frame,
                        f"Conf: {conf:.2f}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        track_colors[pid-1],
                        2)

    return frame



def detect_court_and_players(video_path, output_path, debug_mode=True, shot_threshold=0.5, frame_size=15):
    """
    코트를 감지하고 ROI를 설정한 후 선수들을 감지하는 함수
    
    Args:
        video_path: 입력 비디오 파일 경로
        output_path: 출력 비디오 파일 경로
        debug_mode: 디버그 모드 활성화 여부 (기본값: True)
        shot_threshold: 샷 감지를 위한 confidence threshold (기본값: 0.5)
        frame_size: 사용할 프레임 수 (3,5,7,9,11,13,15 중 선택, 기본값: 9)
    """
    # Step 1: Set up ROI from video
    roi_points = find_roi_from_video(video_path, debug_mode)
    
    if roi_points is None:
        print("Failed to set up ROI.")
        return
    
    # Step 2: Detect players from start using the established ROI
    detect_players_with_roi(video_path, output_path, roi_points, shot_threshold=shot_threshold, frame_size=frame_size, debug_mode=debug_mode)
    
    print("Processing complete")

def expand_roi_points(roi_points, expand_px=100):
    """
    ROI 꼭짓점들을 중심점 기준으로 바깥 방향으로 expand_px만큼 확장
    Args:
        roi_points: (4, 2) numpy array 또는 list, 시계방향(좌상단, 우상단, 우하단, 좌하단)
        expand_px: 확장할 픽셀 수
    Returns:
        확장된 roi_points (np.array)
    """
    roi_points = np.array(roi_points)
    # 중심점 계산
    center = np.mean(roi_points, axis=0)
    expanded = []
    for pt in roi_points:
        vec = pt - center
        norm = np.linalg.norm(vec)
        if norm == 0:
            expanded.append(pt)
        else:
            scale = (norm + expand_px) / norm
            new_pt = center + vec * scale
            expanded.append(new_pt.astype(int))
    return np.array(expanded, dtype=np.int32)

def find_roi_from_video(video_path, debug_mode=True):
    """
    Set up ROI from video
    
    Args:
        video_path: Input video path
        debug_mode: Enable debug mode
    
    Returns:
        roi_points: Established ROI points (None if failed)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None
    
    # Video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Frame center point - adjusted to 2/3 height (1/3 from bottom)
    center = (width // 2, int(height * 2/3))
    
    # ROI setup variables
    roi_points = None
    frame_count = 0
    max_frames_to_find_roi = 300  # Maximum frames to search for ROI
    roi_setup_frame = None  # Store frame where ROI is set
    
    # Store best bounding box combinations (by priority)
    best_boxes_4 = []  # Boxes in all 4 quadrants
    best_boxes_3 = []  # Boxes in 3 quadrants
    best_boxes_2_diag = []  # Boxes in diagonal quadrants
    best_frame_4 = -1
    best_frame_3 = -1
    best_frame_2_diag = -1
    best_frame_image = None  # Store best frame image
    
    print("Searching for ROI. Please wait...")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Court detection
        results = court_model(frame, conf=0.1)  # Lower threshold for better sensitivity
        
        # Extract detected box info
        boxes = []
        confidences = []
        classes = []
        
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            print(f"[DEBUG] Detected boxes: {len(results[0].boxes)}")
            for box in results[0].boxes:
                print(f"  Box: {box.xyxy[0].cpu().numpy()}, id: {box.id[0].cpu().numpy() if hasattr(box, 'id') and box.id is not None else None}")
            for box in results[0].boxes:
                try:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    confidences.append(float(box.conf[0].cpu().numpy()))
                    classes.append(int(box.cls[0].cpu().numpy()))
                except:
                    pass
        
        # Display progress (every 30 frames)
        if frame_count % 30 == 0 or frame_count == 1:
            progress = min(100, int(frame_count / max_frames_to_find_roi * 100))
            print(f"ROI search progress... {progress}% ({frame_count}/{max_frames_to_find_roi})")
        
        # Check bounding box quadrant distribution
        box_quadrants = [get_box_quadrant(box, center) for box in boxes]
        unique_quadrants = set(box_quadrants)
        
        # Check diagonal positions
        has_diagonal_1_3 = 1 in unique_quadrants and 3 in unique_quadrants
        has_diagonal_2_4 = 2 in unique_quadrants and 4 in unique_quadrants
        has_diagonal = has_diagonal_1_3 or has_diagonal_2_4
        
        # Evaluate and store current frame's bounding box combination
        if len(unique_quadrants) == 4:
            # Case 1: Boxes in all 4 quadrants (priority 1)
            best_boxes_4 = boxes.copy()
            best_frame_4 = frame_count
            best_frame_image = frame.copy()
        elif len(unique_quadrants) == 3:
            # Case 2: Boxes in 3 quadrants (priority 2)
            if not best_boxes_3:  # Only save the first occurrence
                best_boxes_3 = boxes.copy()
                best_frame_3 = frame_count
                if best_frame_image is None:
                    best_frame_image = frame.copy()
        elif len(unique_quadrants) == 2 and has_diagonal:
            # Case 3: Boxes in diagonal quadrants (priority 3)
            if not best_boxes_2_diag:  # Only save the first occurrence
                best_boxes_2_diag = boxes.copy()
                best_frame_2_diag = frame_count
                if best_frame_image is None:
                    best_frame_image = frame.copy()
        
        # If boxes found in all 4 quadrants, immediately set ROI
        if len(unique_quadrants) == 4:
            corners = find_extreme_corners(boxes, center, width, height)
            if corners is not None:
                roi_points = np.array(corners, dtype=np.int32)
                print(f"\nBoxes found in all 4 quadrants! Setting ROI immediately! (Frame {frame_count})")
                print(f"Selected ROI coordinates: {roi_points.tolist()}")
                
                # Save ROI setup frame
                roi_setup_filename = f"roi_setup_frame_4quads_{frame_count}.jpg"
                roi_setup_frame = frame.copy()
                debug_image = draw_court_detection(roi_setup_frame, boxes, confidences, classes)
                #cv2.imwrite(roi_setup_filename, debug_image)
                print(f"ROI setup frame saved: {roi_setup_filename}")
                break  # Exit loop once ROI is set
        
        # If maximum frames reached, set ROI with best box combination
        if frame_count >= max_frames_to_find_roi:
            print(f"\nReached {max_frames_to_find_roi} frames. Setting ROI with best box combination.")
            
            selected_boxes = None
            roi_type = ""
            used_frame = -1
            
            # Select best bounding box combination by priority
            if best_boxes_4:
                selected_boxes = best_boxes_4
                roi_type = "4_quadrants"
                used_frame = best_frame_4
                print(f"Using 4 quadrant boxes (Frame {best_frame_4})")
            elif best_boxes_3:
                selected_boxes = best_boxes_3
                roi_type = "3_quadrants"
                used_frame = best_frame_3
                print(f"Using 3 quadrant boxes (Frame {best_frame_3})")
            elif best_boxes_2_diag:
                selected_boxes = best_boxes_2_diag
                roi_type = "diagonal"
                used_frame = best_frame_2_diag
                print(f"Using diagonal boxes (Frame {best_frame_2_diag})")
            else:
                # Use entire frame as ROI if no suitable boxes found
                print("No suitable boxes found. Using entire frame as ROI.")
                roi_points = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.int32)
                roi_type = "full_frame"
            
            # Set ROI with selected boxes
            if selected_boxes is not None:
                corners = find_extreme_corners(selected_boxes, center, width, height)
                if corners is not None:
                    roi_points = np.array(corners, dtype=np.int32)
                    print(f"Selected ROI coordinates: {roi_points.tolist()}")
            
            # Save ROI setup frame
            if best_frame_image is not None:
                roi_setup_filename = f"roi_setup_frame_{roi_type}_{used_frame}.jpg"
                debug_image = draw_court_detection(best_frame_image, selected_boxes if selected_boxes is not None else [], 
                                                  confidences=None, classes=None)
                #cv2.imwrite(roi_setup_filename, debug_image)
                #print(f"ROI setup frame saved: {roi_setup_filename}")
            
            break  # Exit loop after setting ROI
    
    # Cleanup
    cap.release()
    
    # Save ROI visualization image
    if roi_points is not None and best_frame_image is not None:
        # ROI 확장 적용
        roi_points = expand_roi_points(roi_points, expand_px=100)
        roi_display = draw_roi(best_frame_image, roi_points)
        #cv2.imwrite("roi_setup_final.jpg", roi_display)
        #print(f"Final ROI image saved: roi_setup_final.jpg")
    
    return roi_points

def assign_player_id(center_x, center_y, track_id, prev_positions, frame_width, frame_height):
    """플레이어 위치에 따라 ID 할당"""
    # 이전 프레임의 위치 정보가 있으면 활용
    if track_id in prev_positions:
        prev_x, prev_y = prev_positions[track_id]
        # 급격한 위치 변화가 없다면 이전 ID 유지
        if abs(center_x - prev_x) < frame_width * 0.2 and abs(center_y - prev_y) < frame_height * 0.2:
            return track_id % 4 + 1  # 1~4 범위로 매핑
    
    # 이전 프레임의 모든 위치와 비교하여 가장 가까운 ID 찾기
    min_distance = float('inf')
    closest_id = None
    
    for prev_id, (prev_x, prev_y) in prev_positions.items():
        distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
        if distance < min_distance and distance < 200:  # 200 픽셀 이내
            min_distance = distance
            closest_id = prev_id
    
    if closest_id is not None:
        return closest_id % 4 + 1
    
    # 코트를 4분할하여 위치 판단
    is_back = center_y < frame_height * 0.5
    is_left = center_x < frame_width * 0.5
    
    # 위치에 따른 ID 결정
    if is_back:
        if is_left:
            return 1  # 뒤쪽 좌측
        else:
            return 2  # 뒤쪽 우측
    else:
        if is_left:
            return 3  # 앞쪽 좌측
        else:
            return 4  # 앞쪽 우측

def detect_players_with_roi(video_path, output_path, roi_points, shot_threshold=0.5, frame_size=15, debug_mode=True):
    """
    ROI 영역 내의 플레이어를 감지하고 추적하는 함수
    
    Args:
        video_path: 입력 비디오 파일 경로
        output_path: 출력 비디오 파일 경로
        roi_points: ROI 영역의 좌표점들
        shot_threshold: 샷 감지를 위한 confidence threshold (기본값: 0.5)
        frame_size: 사용할 프레임 수 (3,5,7,9,11,13,15 중 선택, 기본값: 9)
        debug_mode: 디버그 모드 활성화 여부 (기본값: True)
    """
    print("\nStarting player detection with established ROI from beginning of video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return
    
    # Video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    image_shape = (height, width, 3)
    
    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Frame counter
    frame_count = 0
    
    # Track colors - for consistent ID visualization
    track_colors = [
        (0, 0, 255),    # Red (ID 1: 뒤쪽 왼쪽)
        (255, 0, 0),    # Blue (ID 2: 뒤쪽 오른쪽)
        (0, 255, 0),    # Green (ID 3: 앞쪽 왼쪽)
        (255, 255, 0),  # Cyan (ID 4: 앞쪽 오른쪽)
    ]
    
    # 트래킹 변수들
    last_positions = {}  # ID별 마지막 위치 (x, y)
    predicted_positions = {}  # 사라진 ID의 예측 위치
    id_velocities = {}  # ID별 속도 벡터
    id_mapping = {}  # 트래킹 ID -> 표시 ID 매핑
    stable_ids = set()  # 안정적으로 할당된 ID 집합
    
    # 예측 객체 타임아웃 관리 변수 추가
    predicted_lifetime = {}  # track_id별 예측 유지 시간(프레임)
    
    # 표시 ID 관리를 위한 변수
    missing_start_frame = {}  # 각 표시 ID가 언제부터 사라졌는지 (프레임 번호)
    
    # 샷 감지 관련 변수
    try:
        shot_detector = STGCNShotDetector(model_dir="model_stgcn_final", frame_size=frame_size, threshold=shot_threshold)  # 샷 감지기 초기화
        print(f"샷 감지기가 성공적으로 초기화되었습니다. (frame_size: {frame_size})")
    except Exception as e:
        print(f"샷 감지기 초기화 중 오류 발생: {e}")
        shot_detector = None
        return

    keypoint_buffers = {1: [], 2: [], 3: [], 4: []}  # 각 ID별 키포인트 버퍼
    shot_results = {1: None, 2: None, 3: None, 4: None}  # 각 ID별 최근 샷 감지 결과
    
    # 고급 추적 시스템 변수
    player_memory = {}   # ID별 플레이어 메모리
    memory_timeout = 60  # 플레이어 메모리 유지 시간 (프레임)
    id_swap_prevention = {}  # ID 스왑 방지를 위한 영역 히스토리
    
    # 추적 변수
    stable_positions = {}  # 안정적인 포지션
    court_positions = ["front_left", "front_right", "back_left", "back_right"]
    position_mapping = {}   # 포지션과 ID 매핑
    
    # 모션 추정 변수
    velocity_history = {}  # ID별 속도 이력
    
    # 예측 객체 타임아웃 관리 변수 추가
    predicted_lifetime = {}  # track_id별 예측 유지 시간(프레임)
    
    # 키포인트 버퍼 초기화 (ID별로 관리)
    keypoint_buffers = {}
    
    # OnePose 모델 초기화
    pose_model = init_pose_model()
    if pose_model is None:
        print("포즈 모델 초기화 실패")
        return
    print("포즈 모델 초기화 성공")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        

        # Player detection with tracking
        results = player_model.track(
            frame, 
            persist=True, 
            conf=0.01, 
            classes=[0],  # Only track class 0 (person)
            tracker="botsort.yaml", #bytetrack.yaml
            nms=True,
            iou=0.5,
            verbose=True  # YOLO 추론 로그 끄기
        )
        
        # 현재 프레임의 트래킹 박스 수집
        tracking_boxes = []
        current_positions = {}  # 현재 프레임의 위치 정보
        
        # 현재 감지된 객체 처리
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            for box in boxes:
                try:
                    if hasattr(box, 'id') and box.id is not None:
                        xyxy = box.xyxy[0].cpu().numpy()
                        track_id = int(box.id[0].cpu().numpy())
                        
                        if is_overlapping(xyxy, roi_points):
                            x1, y1, x2, y2 = map(int, xyxy)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # 이미 다른 객체가 사용 중인 ID는 건너뛰기
                            skip = False
                            for existing_id, pos in current_positions.items():
                                if id_mapping.get(existing_id) == id_mapping.get(track_id):
                                    skip = True
                                    break
                            
                            if not skip:
                                tracking_boxes.append(xyxy)
                                current_positions[track_id] = (center_x, center_y)
                except Exception as e:
                    print(f"Error processing box: {e}")
        
        # 불필요한 track_id 정리
        # 1. 현재 프레임에서 감지된 track_id만 유지
        current_track_ids = set(current_positions.keys())
        
        # 사라진 객체 처리 - missing 프레임 시작 기록 (ID 정리 전에 먼저 수행)
        for track_id in list(last_positions.keys()):
            if track_id not in current_positions and track_id in id_mapping:
                displayed_id = id_mapping[track_id]  # 표시 ID (1, 2, 3, 4 중 하나)
                print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id} (track_id {track_id}) 사라짐 감지")
                if track_id not in predicted_positions:
                    # 예측 위치 초기화
                    predicted_positions[track_id] = last_positions[track_id]
                    # 사라짐 시작 프레임 기록 (표시 ID 기준)
                    if displayed_id not in missing_start_frame:
                        missing_start_frame[displayed_id] = frame_count
                        print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id} 사라짐 시작 기록 (시작 프레임: {frame_count})")

        # 현재 예측 중인 track_id 목록 (시각화를 위해 유지할 ID)
        predicted_track_ids = set(predicted_positions.keys())
        
        # 2. id_mapping에서 오래된 track_id 삭제 (예측 중인 ID는 제외)
        for track_id in list(id_mapping.keys()):
            if track_id not in current_track_ids and track_id not in predicted_track_ids:
                # 사라진 객체에 대한 missing_start_frame 기록이 없다면 기록
                if track_id not in missing_start_frame and track_id in last_positions:
                    missing_start_frame[track_id] = frame_count
                    print(f"[DEBUG] Frame {frame_count}: ID {track_id} 삭제 전 사라짐 시작 기록 (시작 프레임: {frame_count})")
                del id_mapping[track_id]
        
        # 3. predicted_positions에서 오래된 track_id 삭제 (20프레임 이상 사라진 경우)
        for track_id in list(predicted_lifetime.keys()):
            if predicted_lifetime[track_id] > 20:
                if track_id in predicted_positions:
                    del predicted_positions[track_id]
                del predicted_lifetime[track_id]
        
        # 4. id_velocities에서 오래된 track_id 삭제 (예측 중인 ID는 제외)
        for track_id in list(id_velocities.keys()):
            if track_id not in current_track_ids and track_id not in predicted_track_ids:
                del id_velocities[track_id]
        
        # 5. last_positions에서 오래된 track_id 삭제 (예측 중인 ID는 제외)
        for track_id in list(last_positions.keys()):
            if track_id not in current_track_ids and track_id not in predicted_track_ids:
                del last_positions[track_id]

        # 예측 위치 업데이트
        for track_id in list(predicted_positions.keys()):
            if track_id in current_positions:
                # 객체가 다시 나타났으면 예측 위치 및 타임아웃 제거
                displayed_id = id_mapping[track_id]  # 표시 ID
                if displayed_id in missing_start_frame:
                    # 사라짐 기록 삭제
                    print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id}가 다시 나타남, 사라짐 기록 삭제")
                    del missing_start_frame[displayed_id]
                
                del predicted_positions[track_id]
                if track_id in predicted_lifetime:
                    del predicted_lifetime[track_id]
            else:
                # 예측 위치 업데이트
                if track_id in id_velocities:
                    prev_pos = predicted_positions[track_id]
                    velocity = id_velocities[track_id]
                    new_x = prev_pos[0] + velocity[0]
                    new_y = prev_pos[1] + velocity[1]
                    predicted_positions[track_id] = (new_x, new_y)
                # 타임아웃 증가
                predicted_lifetime[track_id] = predicted_lifetime.get(track_id, 0) + 1
                # 30프레임 이상 감지 안 되면 삭제
                if predicted_lifetime[track_id] > 20:
                    displayed_id = id_mapping[track_id]  # 표시 ID
                    if displayed_id in missing_start_frame:
                        print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id}가 20프레임 이상 사라짐, 추적 중단")
                        del missing_start_frame[displayed_id]
                    
                    del predicted_positions[track_id]
                    del predicted_lifetime[track_id]
        
        # 디버깅: 추적 상태 출력
        print(f"[DEBUG] frame {frame_count}")
        print(f"current_positions: {current_positions}")
        print(f"id_mapping: {id_mapping}")
        print(f"predicted_positions: {predicted_positions}")
        print(f"missing_start_frame: {missing_start_frame}")
        
        # 현재 표시된 표시 ID 목록 확인
        current_displayed_ids = set()
        for track_id in current_positions.keys():
            if track_id in id_mapping:
                current_displayed_ids.add(id_mapping[track_id])
        
        # 현재 표시된 ID가 missing_start_frame에 있다면 (다른 track_id로 나타난 경우) 통계 업데이트
        for displayed_id in list(missing_start_frame.keys()):
            if displayed_id in current_displayed_ids:
                print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id}가 다른 track_id로 나타남, 사라짐 기록 삭제")
                del missing_start_frame[displayed_id]
        
        # ID 매핑 업데이트
        if len(current_positions) == 4:
            # 4명의 플레이어가 모두 감지된 경우
            # y 좌표로 정렬하여 앞/뒤 구분
            sorted_players = sorted(current_positions.items(), key=lambda x: x[1][1])
            back_players = sorted_players[:2]  # y값이 작은 두 명 (뒤쪽)
            front_players = sorted_players[2:]  # y값이 큰 두 명 (앞쪽)
            
            # 각 그룹 내에서 x 좌표로 정렬
            back_players.sort(key=lambda x: x[1][0])   # x 좌표로 정렬
            front_players.sort(key=lambda x: x[1][0])  # x 좌표로 정렬
            
            # ID 할당 (1,2: 뒤쪽, 3,4: 앞쪽)
            new_mapping = {
                back_players[0][0]: 1,   # 뒤쪽 왼쪽
                back_players[1][0]: 2,   # 뒤쪽 오른쪽
                front_players[0][0]: 3,  # 앞쪽 왼쪽
                front_players[1][0]: 4   # 앞쪽 오른쪽
            }
            
            # 안정적인 ID 매핑 업데이트
            if not stable_ids:  # 아직 안정적인 ID가 설정되지 않은 경우
                id_mapping = new_mapping
                stable_ids = set(current_positions.keys())
        
        # 새로운 객체에 대한 ID 매핑
        for track_id in current_positions:
            if track_id not in id_mapping:
                print(f"[DEBUG] New track_id detected: {track_id}")
                # 예측 위치와 가장 가까운 ID 찾기
                min_dist = float('inf')
                best_id = None
                for pred_id, pred_pos in predicted_positions.items():
                    if pred_id in id_mapping:
                        curr_pos = current_positions[track_id]
                        dist = ((curr_pos[0] - pred_pos[0])**2 + 
                               (curr_pos[1] - pred_pos[1])**2)**0.5
                        if dist < min_dist and dist < 200:  # 허용 범위 확장
                            min_dist = dist
                            best_id = pred_id
                if best_id is not None:
                    # 예측 객체와 매칭되면 해당 ID 재사용
                    id_mapping[track_id] = id_mapping[best_id]
                    del predicted_positions[best_id]
                    if best_id in predicted_lifetime:
                        del predicted_lifetime[best_id]
                else:
                    # 사용되지 않은 ID 중에서 무조건 할당
                    used_ids = set(id_mapping.values())
                    for i in range(1, 5):
                        if i not in used_ids:
                            id_mapping[track_id] = i
                            break
        # === [추가] 3명만 확실히 매칭된 경우, 남은 1명에게 남은 ID 강제 할당 ===
        current_ids = set(current_positions.keys())
        used_display_ids = set(id_mapping.get(tid) for tid in current_ids if tid in id_mapping)
        all_display_ids = set([1, 2, 3, 4])
        unused_display_ids = list(all_display_ids - used_display_ids)
        unmatched_ids = [tid for tid in current_ids if tid not in id_mapping]
        if len(unmatched_ids) == 1 and len(unused_display_ids) == 1:
            id_mapping[unmatched_ids[0]] = unused_display_ids[0]

        # 속도 계산 및 사라진 객체 처리
        for track_id, curr_pos in current_positions.items():
            if track_id in last_positions:
                # 속도 계산
                last_pos = last_positions[track_id]
                velocity = (curr_pos[0] - last_pos[0], curr_pos[1] - last_pos[1])
                id_velocities[track_id] = velocity
        
        # 사라진 객체 처리
        for track_id in list(last_positions.keys()):
            if track_id not in current_positions and track_id in id_mapping:
                displayed_id = id_mapping[track_id]  # 표시 ID (1, 2, 3, 4 중 하나)
                print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id} (track_id {track_id}) 사라짐 감지")
                if track_id not in predicted_positions:
                    # 예측 위치 초기화
                    predicted_positions[track_id] = last_positions[track_id]
                    # 사라짐 시작 프레임 기록 (표시 ID 기준)
                    if displayed_id not in missing_start_frame:
                        missing_start_frame[displayed_id] = frame_count
                        print(f"[DEBUG] Frame {frame_count}: 표시 ID {displayed_id} 사라짐 시작 기록 (시작 프레임: {frame_count})")
        
        # 위치 정보 업데이트
        last_positions = current_positions.copy()
        
        # ROI 시각화
        display_frame = draw_roi(frame.copy(), roi_points, tracking_boxes)
        
        # 트래킹 박스 및 예측 객체 시각화
        all_boxes = []
        all_ids = []
        all_is_predicted = []
        
        # 실제 객체 추가
        for i, box in enumerate(tracking_boxes):
            track_id = list(current_positions.keys())[i]
            if track_id in id_mapping:
                all_boxes.append(box)
                all_ids.append(id_mapping[track_id])
                all_is_predicted.append(False)
        
        # 예측 객체 추가
        for track_id, pred_pos in predicted_positions.items():
            if track_id in id_mapping:
                # 예측 객체의 크기는 마지막으로 본 크기 사용
                x, y = map(int, pred_pos)
                box_size = 100  # 기본 크기
                pred_box = [x - box_size//2, y - box_size//2, x + box_size//2, y + box_size//2]
                all_boxes.append(pred_box)
                all_ids.append(id_mapping[track_id])
                all_is_predicted.append(True)
                print(f"[DEBUG] Frame {frame_count}: 예측 객체 시각화 - ID {track_id}(표시 ID:{id_mapping[track_id]}) 위치: ({x}, {y})")
        
        # 모든 객체 시각화
        for box, displayed_id, is_predicted in zip(all_boxes, all_ids, all_is_predicted):
            try:
                x1, y1, x2, y2 = map(int, box)
                color = track_colors[(displayed_id - 1) % len(track_colors)]
                
                if is_predicted:
                    # 예측 객체는 점선으로 표시
                    for i in range(0, 360, 20):
                        rad = i * np.pi / 180
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        size_x = (x2 - x1) // 2
                        size_y = (y2 - y1) // 2
                        pt1 = (int(center_x + np.cos(rad) * size_x), 
                               int(center_y + np.sin(rad) * size_y))
                        cv2.circle(display_frame, pt1, 1, color, -1)
                    label = f'ID:{displayed_id} (predicted)'
                else:
                    # 실제 객체는 실선으로 표시
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f'ID:{displayed_id}'
                
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 포즈 추정 (실제 객체만)
                if not is_predicted:
                    # 포즈 추정 및 시각화
                    display_frame = process_pose_estimation(display_frame, [box], pose_model, track_colors, {0: displayed_id}, shot_threshold=shot_threshold)
            except Exception as e:
                print(f"Error visualizing box for ID {displayed_id}: {e}")
                continue

        # 디버그 정보에 샷 감지 상태 추가
        if debug_mode:
            # 기존 디버그 정보
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Players: {len(tracking_boxes)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 각 플레이어의 샷 감지 상태 표시
            for pid, result in shot_results.items():
                if result is not None:
                    player_type = "Front" if pid in [3, 4] else "Back"
                    status = f"ID {pid} ({player_type}): {'SHOT' if result['is_shot'] else 'NO SHOT'} ({result['confidence']:.2f})"
                    y_pos = 90 + (pid - 1) * 30
                    cv2.putText(display_frame, status, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_colors[(pid - 1) % len(track_colors)], 2)
            
            # 키포인트 버퍼 상태 표시
            for pid, buffer in keypoint_buffers.items():
                status = f"ID {pid} Buffer: {len(buffer)}/15"
                y_pos = 210 + (pid - 1) * 30
                cv2.putText(display_frame, status, (width - 200, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_colors[(pid - 1) % len(track_colors)], 2)

        # 결과 표시 및 저장
        cv2.imshow('Court ROI Detection', display_frame)
        out.write(display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")

###############################################################################
# Keypoint Processing Functions
###############################################################################

def keypoints_to_relative_vector(keypoints_list, image_shape):
    """
    키포인트 좌표를 상대 좌표로 변환
    
    Args:
        keypoints_list: 키포인트 좌표 리스트 [(N, 17, 2) 형태]
        image_shape: 이미지 크기 (H, W, C)
    
    Returns:
        relative_keypoints: 상대 좌표로 변환된 키포인트 (17, 2) 형태
    """
    try:
        h, w, _ = image_shape
        
        for keypoints in keypoints_list:
            # 키포인트가 없거나 형태가 잘못된 경우 처리
            if keypoints is None or len(keypoints) == 0:
                continue
                
            # 2D 좌표만 사용
            if len(keypoints.shape) == 3:  # (N, 17, 3) 또는 (N, 17, 2)
                keypoints_2d = keypoints[:, :, :2]
            elif len(keypoints.shape) == 2:  # (17, 2) 또는 (17, 3)
                keypoints_2d = keypoints[:, :2]
            else:
                print(f"예상치 못한 키포인트 형태: {keypoints.shape}")
                continue
            
            # 엉덩이 관절 인덱스 확인
            if keypoints_2d.shape[0] < 17:
                print(f"키포인트 수가 부족합니다: {keypoints_2d.shape}")
                continue
            
            try:
                # 1. 이미지 크기로 정규화 (0~1 사이 값으로)
                keypoints_2d = keypoints_2d.astype(np.float32)
                keypoints_2d[:, 0] /= w  # x 좌표를 이미지 너비로 나눔
                keypoints_2d[:, 1] /= h  # y 좌표를 이미지 높이로 나눔
                
                # 2. 엉덩이 중심점(hip center) 계산
                hip_indices = [11, 12]  # COCO 포맷의 엉덩이 관절 인덱스
                valid_hips = [idx for idx in hip_indices if idx < len(keypoints_2d)]
                
                if len(valid_hips) > 0:
                    center_x = np.mean(keypoints_2d[valid_hips, 0])
                    center_y = np.mean(keypoints_2d[valid_hips, 1])
                else:
                    # 엉덩이 관절을 찾을 수 없는 경우 모든 키포인트의 중심점 사용
                    center_x = np.mean(keypoints_2d[:, 0])
                    center_y = np.mean(keypoints_2d[:, 1])
                
                # 3. 중심점 기준 상대좌표 변환
                relative_keypoints = keypoints_2d.copy()
                relative_keypoints[:, 0] -= center_x
                relative_keypoints[:, 1] -= center_y
                
                # 4. 표준화 (평균=0, 표준편차=1)
                # 0으로 나누는 것을 방지하기 위해 epsilon 추가
                eps = 1e-8
                std = np.std(relative_keypoints, axis=0) + eps
                relative_keypoints = (relative_keypoints - np.mean(relative_keypoints, axis=0)) / std
                
                # 5. [17, 2] 형태 그대로 반환 (flatten 제거)
                return relative_keypoints
                
            except Exception as e:
                print(f"키포인트 처리 중 오류 발생: {str(e)}")
                continue
        
        return None
            
    except Exception as e:
        print(f"상대 벡터 변환 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_keypoints_for_shot_detection(keypoint_sequence, image_shape):
    """
    샷 감지를 위한 키포인트 전처리
    
    Args:
        keypoint_sequence: (frame_size, 17, 2) 형태의 키포인트 시퀀스
        image_shape: 이미지 크기 (H, W, C)
    
    Returns:
        numpy.ndarray: 전처리된 키포인트 시퀀스 (frame_size, 17, 2) 형태
    """
    try:
        if keypoint_sequence is None or len(keypoint_sequence) == 0:
            return None
            
        # 시퀀스 길이 확인
        if len(keypoint_sequence) < 15:
            print(f"시퀀스가 너무 짧습니다: {len(keypoint_sequence)} (15 필요)")
            return None
        
        # 최근 15프레임만 사용
        keypoint_sequence = keypoint_sequence[-15:]
        
        # 1. 상대 좌표로 변환
        processed_sequence = []
        for frame_keypoints in keypoint_sequence:
            # 2D 좌표만 사용 (x, y)
            frame_keypoints_2d = frame_keypoints[:, :2] if frame_keypoints.shape[-1] > 2 else frame_keypoints
            
            # 각 프레임의 키포인트를 상대 좌표로 변환
            relative_coords = keypoints_to_relative_vector([frame_keypoints_2d], image_shape)
            if relative_coords is None:
                continue
            processed_sequence.append(relative_coords)
        
        # 처리된 시퀀스 확인
        if len(processed_sequence) < 15:
            print(f"처리된 시퀀스가 부족합니다: {len(processed_sequence)} (15 필요)")
            return None
        
        # 2. 시퀀스를 올바른 형태로 변환 (15, 17, 2)
        processed_sequence = np.array(processed_sequence)
        # 형태 확인
        print(f"Processed sequence shape: {processed_sequence.shape}")
        
        # reshape 제거 - 이미 (15, 17, 2) 형태이므로 유지
        # processed_sequence = processed_sequence.reshape(15, -1)
        
        return processed_sequence
    
    except Exception as e:
        print(f"키포인트 전처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

