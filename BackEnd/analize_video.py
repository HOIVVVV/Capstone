import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from frame_extractor import extract_frames
from BackEnd.PushImageToModel import predict_images_in_folder

def analyze_video(video_path):
    if not os.path.isfile(video_path):
        print("❌ 유효하지 않은 영상 경로입니다.")
        return

    # 고유한 타임스탬프로 임시 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_output_folder = f"temp_frames_{timestamp}"
    
    # ✅ 결과 폴더는 'Result/' 아래에 생성
    result_root = "Result"
    result_output_folder = os.path.join(result_root, f"results_{timestamp}")
    os.makedirs(result_output_folder, exist_ok=True)

    # 1단계: 영상 → 프레임 추출
    print("\n📽️ 프레임 추출 중...")
    os.makedirs(frame_output_folder, exist_ok=True)
    extract_frames(video_path, frame_output_folder, seconds_between_frames=1)

    # 2단계: 이미지 → 예측 + Grad-CAM
    print("\n🧠 이미지 분석 중...")
    predict_images_in_folder(frame_output_folder, result_output_folder)

    # ✅ 프레임 폴더 삭제
    try:
        shutil.rmtree(frame_output_folder)
        print(f"🧹 프레임 폴더 삭제 완료: {frame_output_folder}")
    except Exception as e:
        print(f"⚠️ 프레임 폴더 삭제 중 오류 발생: {e}")

    print(f"\n✅ 분석 완료! 결과 저장 위치: {os.path.abspath(result_output_folder)}")
    
if __name__ == "__main__":
    input_path = input("🎬 분석할 영상 경로를 입력하세요 (파일 또는 폴더): ").strip()

    if os.path.isfile(input_path):
        # 단일 파일 분석
        analyze_video(input_path)

    elif os.path.isdir(input_path):
        # 폴더 내 영상 파일 전부 탐색
        for file_name in os.listdir(input_path):
            if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                full_path = os.path.join(input_path, file_name)
                print(f"\n====== 🎞️ {file_name} 분석 시작 ======")
                analyze_video(full_path)
    else:
        print("❌ 유효한 파일 또는 폴더 경로가 아닙니다.")