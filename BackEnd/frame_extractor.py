# frame_extractor.py
import os
import cv2

def extract_frames(video_path, output_folder, seconds_between_frames=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("FPS 정보를 불러올 수 없습니다.")
        return

    frame_interval = int(fps * seconds_between_frames)
    print(f"FPS: {fps}, 프레임 추출 간격: {frame_interval}프레임마다")

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"{saved_count}개의 프레임 저장 완료")
    
def extract_key_frames_for_text(video_path, output_folder, max_frames=6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 영상 열기 실패:", video_path)
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps else 0

    print(f"🎞️ 전체 프레임: {frame_count}, FPS: {fps:.2f}, 길이: {duration:.2f}초")

    timestamps = []
    thirds = [0.0, 0.33, 0.66]
    offset = 1.0  # 각 위치 앞뒤로 ±1초

    for base in thirds:
        t1 = max(0.0, duration * base - offset)
        t2 = min(duration, duration * base + offset)
        timestamps.extend([t1, t2])

    extracted_paths = []

    for idx, t in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()
        if not success:
            print(f"⚠️ {t:.2f}초 프레임 추출 실패")
            continue
        save_path = os.path.join(output_folder, f"text_frame_{idx+1}.jpg")
        cv2.imwrite(save_path, frame)
        extracted_paths.append(save_path)

    cap.release()
    return extracted_paths
