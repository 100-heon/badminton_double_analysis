import os
# ─── 항상 수정 확인 코드 수정 기본 값:  utils_stgcn_transformer_64indim ─────────
from utils_stgcn_transformer_64indim import detect_court_and_players
# ─── 코드 수정 ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 기본 설정
    video_path = 'men_double.mp4'  # 여기에 처리할 비디오 파일 경로를 입력하세요
    output_path = 'test2123.mp4'
    threshold = 0.85
    frame_size = 15
    debug_mode = False

    print('=== 시작: ST-GCN Shot Detection ===')
    print(f'- video: {video_path}')
    print(f'- output: {output_path}')
    print(f'- threshold: {threshold}')
    print(f'- frame_size: {frame_size}')
    print(f'- debug_mode: {debug_mode}')

    # 실행
    detect_court_and_players(
        video_path=video_path,
        output_path=output_path,
        debug_mode=debug_mode,
        shot_threshold=threshold,
        frame_size=frame_size
    )
    print('=== 완료 ===')
