from flask import Flask, render_template, request, redirect, url_for
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
FRAME_FOLDER = 'frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

def extract_frames(video_path, output_folder, seconds_between_frames=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패")
        return

    # FPS 감지
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
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"{saved_count}개의 프레임 저장 완료")

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file.filename == '':
        return '파일이 없습니다.'

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    extract_frames(video_path, FRAME_FOLDER, seconds_between_frames=1)  # 1초마다 추출

    return f"영상 업로드 및 프레임 추출 완료: {file.filename}"

if __name__ == '__main__':
    app.run(debug=True)
