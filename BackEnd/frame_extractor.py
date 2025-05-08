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
    
def extract_key_frames_for_text(video_path, output_folder, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    extracted_paths = []

    for i in range(0, total_frames, step):
        if len(extracted_paths) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        filename = os.path.join(output_folder, f"textframe_{len(extracted_paths):02d}.png")
        cv2.imwrite(filename, frame)
        extracted_paths.append(filename)

    cap.release()
    return extracted_paths
