import os
import sys
import shutil
import json
import pytesseract
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from BackEnd.frame_extractor import extract_frames, extract_key_frames_for_text
from BackEnd.PushImageToModel import predict_images_in_folder
from BackEnd.GetTextFromImage import analyze_images_in_folder
from PIL import Image
from db.insert_to_db import insert_analysis_results
from BackEnd import progress
import shutil

def analyze_video(video_path):
    if not os.path.isfile(video_path):
        print("❌ 유효하지 않은 영상 경로입니다.")
        return

    # ✅ 기존 static 결과 삭제
    static_results_dir = os.path.join("static", "predictResults")
    if os.path.exists(static_results_dir):
        shutil.rmtree(static_results_dir)
    os.makedirs(static_results_dir, exist_ok=True)

    progress["step"] = "📂 영상 확인 중..."
    progress["percent"] = 5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_output_folder = f"temp_frames_{timestamp}"
    frame_output_path = os.path.abspath(frame_output_folder)
    os.makedirs(frame_output_path, exist_ok=True)

    result_root = os.path.join("static", "results")  # static/results
    os.makedirs(result_root, exist_ok=True)

    result_output_folder = os.path.join(result_root, f"results_{timestamp}")
    os.makedirs(result_output_folder, exist_ok=True)

    result_output_path = os.path.abspath(result_output_folder)

    # 🔍 텍스트 추출용 프레임 분석
    progress["step"] = "📝 텍스트 추출용 프레임 분석 중..."
    progress["percent"] = 15
    text_frame_paths = extract_key_frames_for_text(video_path, frame_output_folder, max_frames=6)

    if not text_frame_paths:
        progress["step"] = "❌ 텍스트 프레임 추출 실패"
        progress["percent"] = 100
        return

    # 🔍 OCR 분석
    progress["step"] = "🔍 텍스트 분석 중..."
    progress["percent"] = 30
    ocr_result = analyze_images_in_folder(frame_output_path, result_output_path, frame_output_path)

    if not ocr_result or not ocr_result.get('success'):
        progress["step"] = "❌ 텍스트 분석 실패"
        progress["percent"] = 100
        return

    district = ocr_result.get('district')
    recorded_date = ocr_result.get('date')

    # ✅ 본 분석 시작
    progress["step"] = "📽️ 프레임 추출 중..."
    progress["percent"] = 40
    extract_frames(video_path, frame_output_folder, seconds_between_frames = 2)

    # ✅ 이미지 예측
    progress["step"] = "🧠 이미지 분석 중..."
    progress["percent"] = 50
    video_title = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🎥 video_title 추출: {video_title}")

    # ✅ predict_images_in_folder 내부에서 percent 갱신 필요!
    predict_images_in_folder(frame_output_folder, result_output_folder, video_title)

    # ✅ 분석 후
    try:
        shutil.rmtree(frame_output_folder)
    except Exception as e:
        print(f"⚠️ 프레임 폴더 삭제 중 오류 발생: {e}")

    progress["step"] = "🗃️ 결과 이미지 저장 중..."
    progress["percent"] = 95
    
    #DB 삽입
    #insert_analysis_results(
    #    video_path=video_path,
    #    result_dir=result_output_folder,
    #    district=district,
    #    recorded_date=recorded_date
    #)
    
    meta_info = {
        "video_path": video_path,
        "district": district,
        "recorded_date": recorded_date
    }

    meta_path = os.path.join(result_output_folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
        
    # ✅ 분석 끝난 후 영상 삭제
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🧹 업로드 영상 삭제됨: {video_path}")
    except Exception as e:
        print(f"⚠️ 업로드 영상 삭제 실패: {e}")

    progress["step"] = "✅ 분석 완료!"
    progress["percent"] = 100
    progress["done"] = True  # ✅ 명시적 종료
    
    # ✅ 분석 결과를 static 폴더로 복사
    static_results_dir = os.path.join("static", "predictResults")
    os.makedirs(static_results_dir, exist_ok=True)

    timestamped_folder_name = os.path.basename(result_output_folder)
    final_output_path = os.path.join(static_results_dir, timestamped_folder_name)

    if os.path.exists(final_output_path):
        shutil.rmtree(final_output_path)
    shutil.copytree(result_output_folder, final_output_path)

    print(f"\n✅ 분석 완료! 결과 저장 위치: {os.path.abspath(result_output_folder)}")

    # ✅ 결과 복사 후, result_output_folder 삭제 (DB에 저장 전임)
    try:
        shutil.rmtree(result_output_folder)
        print(f"🧹 static/results 내부 분석 결과 삭제됨: {result_output_folder}")
    except Exception as e:
        print(f"⚠️ static/results 분석 결과 삭제 실패: {e}")

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