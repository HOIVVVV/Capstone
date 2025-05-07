import os
from datetime import datetime
from frame_extractor import extract_frames
from BackEnd import PushImageToModel  # 너가 작성한 이미지 분석 코드

def analyze_video(video_path):
    if not os.path.isfile(video_path):
        print("❌ 유효하지 않은 영상 경로입니다.")
        return

    # 고유한 타임스탬프로 임시 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_output_folder = f"temp_frames_{timestamp}"
    result_output_folder = f"results_{timestamp}"

    # 1단계: 영상 → 프레임 추출
    print("\n📽️ 프레임 추출 중...")
    os.makedirs(frame_output_folder, exist_ok=True)
    extract_frames(video_path, frame_output_folder, seconds_between_frames=1)

    # 2단계: 이미지 → 예측 + Grad-CAM
    print("\n🧠 이미지 분석 중...")
    os.makedirs(result_output_folder, exist_ok=True)
    PushImageToModel(frame_output_folder, result_output_folder)

    print(f"\n✅ 분석 완료! 결과 저장 위치: {os.path.abspath(result_output_folder)}")

if __name__ == "__main__":
    video_path = input("분석할 영상 파일 경로를 입력하세요: ").strip()
    analyze_video(video_path)
