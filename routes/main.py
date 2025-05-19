# routes/main.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from flask import Blueprint, render_template, request, current_app
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from BackEnd.analyze_video import analyze_video  # 기존 함수 import
from BackEnd import progress
from BackEnd.db.insert_to_db import insert_analysis_results_selected
from BackEnd.db.models import db, DamageImage, Video  # 모델 경로에 맞게 조정

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

@main.route("/api/count_summary")
def count_summary():
    try:
        damage_count = db.session.query(DamageImage).count()
        video_count = db.session.query(Video).count()

        return jsonify({
            "damage_images": damage_count,
            "videos": video_count
        })
    except Exception as e:
        print("📛 /api/count_summary 오류:", e)
        return jsonify({
            "damage_images": 0,
            "videos": 0
        }), 500
        
#진행도 바
@main.route('/progress')
def get_progress():

    response = dict(progress)  # 딕셔너리 복사

    if progress.get("done"):
        # ✅ 다음 요청에서는 polling을 멈추게 하고 progress 초기화
        progress["step"] = "대기 중"
        progress["percent"] = 0
        progress["current_file"] = ""
        progress["done"] = False  # ✅ 리셋

    return jsonify(response)


#영상 업로드 페이지
@main.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")

#영상 업로드
@main.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files and 'video[]' not in request.files:
        return jsonify({'success': False, 'message': 'No video provided'})

    # ✅ 단일 혹은 다중 업로드 지원
    files = request.files.getlist('video[]')
    uploaded_paths = []

    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        uploaded_paths.append(save_path)

    # ✅ 분석은 별도 스레드에서 시작
    from threading import Thread
    for path in uploaded_paths:
        Thread(target=analyze_video, args=(path,)).start()

    return jsonify({'success': True})

#결과 반환환
@main.route("/result_images")
def result_images():
    root_dir = os.path.join("static", "results")
    if not os.path.exists(root_dir):
        return jsonify([])

    # 1) 최신 결과 폴더 선택
    subfolders = sorted(
        [os.path.join(root_dir, d) for d in os.listdir(root_dir)
         if os.path.isdir(os.path.join(root_dir, d))],
        reverse=True
    )
    if not subfolders:
        return jsonify([])

    latest_dir = subfolders[0]

    # 2) 최신 폴더의 모든 하위 디렉토리(영상별) 및 직접 이미지 탐색
    image_files = []
    for entry in os.listdir(latest_dir):
        entry_path = os.path.join(latest_dir, entry)
        if os.path.isdir(entry_path):
            # 영상별 폴더 안 이미지
            for fname in os.listdir(entry_path):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    rel = os.path.relpath(os.path.join(entry_path, fname), "static")
                    image_files.append(rel)
        else:
            # 만약 최신 폴더 바로 아래에도 이미지가 있다면
            if entry.lower().endswith((".jpg", ".png", ".jpeg")):
                rel = os.path.relpath(os.path.join(latest_dir, entry), "static")
                image_files.append(rel)

    return jsonify(image_files)

@main.route('/delete_result_image', methods=['POST'])
def delete_result_image():
    data = request.get_json()
    relative_path = data.get("path")  # 예: results/results_20250519_xxxx/CCTV_/img.jpg

    # ✅ static 경로
    static_path = os.path.join("static", relative_path)

    # ✅ Result 경로: static/results/ → Result/
    parts = relative_path.split(os.sep)
    if parts[0] == "results":
        result_path = os.path.join("Result", *parts[1:])
    else:
        result_path = None  # fallback 처리

    deleted = []

    try:
        if os.path.exists(static_path):
            os.remove(static_path)
            deleted.append("static")
        if result_path and os.path.exists(result_path):
            os.remove(result_path)
            deleted.append("Result")

        if not deleted:
            return jsonify({"error": "파일을 찾을 수 없습니다."}), 404
        else:
            return jsonify({"success": True, "deleted_from": deleted})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@main.route('/save_results_to_db', methods=['POST'])
def save_results_to_db():
    from BackEnd.db.insert_to_db import insert_analysis_results_selected

    data = request.get_json()
    images = data.get("images", [])

    if not images:
        return jsonify({"error": "이미지 없음"}), 400

    # 가장 최근 결과 폴더 이름 추출
    first = images[0]
    parts = first.split(os.sep)
    result_folder = parts[1]  # results_20250519_141239
    meta_path = os.path.join("static", "results", result_folder, "meta.json")

    try:
        insert_analysis_results_selected(images, meta_path)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
