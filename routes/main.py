# routes/main.py
from flask import Blueprint, render_template, request, current_app
import os
from BackEnd.frame_extractor import extract_frames

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('upload.html')

@main.route('/hello')
def hello():
    return "hello"

@main.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file.filename == '':
        return '파일이 없습니다.'

    upload_folder = current_app.config['UPLOAD_FOLDER']
    frame_folder = current_app.config['FRAME_FOLDER']

    video_path = os.path.join(upload_folder, file.filename)
    file.save(video_path)

    extract_frames(video_path, frame_folder, seconds_between_frames=1)

    return f"영상 업로드 및 프레임 추출 완료: {file.filename}"
