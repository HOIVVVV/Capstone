# routes/main.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from flask import Blueprint, render_template, request, current_app
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from BackEnd.analyze_video import analyze_video  # 기존 함수 import
from BackEnd.progress import progress

main = Blueprint('main', __name__)

#메인 데시보드 페이지지
@main.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")

@main.route("/stats")
def stats():
    return render_template("stats.html", active_page="stats")

@main.route("/result")
def result():
    return render_template("result.html", active_page="result")

@main.route("/map")
def map():
    return render_template("map.html", active_page="map")

#진행도 바
@main.route("/progress")
def get_progress():
    return jsonify(progress)

#영상 업로드 페이지지
@main.route('/upload', methods=['GET', 'POST'])
def upload_video():
    uploaded_filename = None

    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template("upload.html", error="❌ 영상 파일이 없습니다.")

        file = request.files['video']
        if file.filename == '':
            return render_template("upload.html", error="❌ 빈 파일 이름입니다.")

        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        try:
            analyze_video(file_path)
            return render_template("upload.html", uploaded_filename=filename)
        except Exception as e:
            return render_template("upload.html", error=f"❌ 분석 중 오류: {str(e)}")

    # GET 요청 시 기본 화면 렌더링
    return render_template("upload.html")