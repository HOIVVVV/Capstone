# routes/main.py
import os
import sys
import re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from flask import Blueprint, render_template, request, current_app
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from BackEnd.analyze_video import analyze_video  # 기존 함수 import
from BackEnd import progress
from BackEnd.db.insert_to_db import insert_analysis_results_selected
from BackEnd.db.models import db, DamageImage, Video  # 모델 경로에 맞게 조정

main = Blueprint('main', __name__)

@main.route('/')
def dashboard():
    # ✅ image_id 기준으로 최신 이미지 3개 가져오기
    recent_images = (
        DamageImage.query
        .order_by(DamageImage.image_id.desc())  # 최신 ID 순
        .limit(3)
        .all()
    )

    image_results = []
    for img in recent_images:
        image_results.append({
            "time": img.timeline.strftime("%H:%M:%S") if img.timeline else "시간 없음",
            "label": img.damage_type or "손상 없음",
            "file": img.image_path.replace("static/", "")  # 템플릿에서는 static/ 제외
        })

    return render_template("dashboard.html", image_results=image_results)

@main.route('/info')
def system_info():
    return render_template('info.html')

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
        
@main.route("/api/recent_images")
def recent_images():
    image_dir = os.path.join(current_app.static_folder, "results")
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in sorted(files, reverse=True):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), current_app.static_folder)
                image_paths.append("/static/" + rel_path.replace("\\", "/"))
            if len(image_paths) >= 12:
                break
        if len(image_paths) >= 12:
            break
    return jsonify({"images": image_paths})
        
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

@main.route("/upload", methods=["POST"])
def upload_video():
    def korean_safe_filename(filename):
        name, ext = os.path.splitext(filename)
        name = re.sub(r'[^\w가-힣\s-]', '', name)      # 한글, 영문, 숫자, 공백, 하이픈만 허용
        name = re.sub(r'\s+', '_', name)              # 공백은 언더스코어로
        return name + ext

    if 'video' not in request.files and 'video[]' not in request.files:
        return jsonify({'success': False, 'message': 'No video provided'})

    files = request.files.getlist('video[]')
    uploaded_paths = []
    renamed_files = []

    for file in files:
        original_filename = file.filename
        filename = korean_safe_filename(original_filename)

        # ✅ 70자 초과 시 자르고 "_short" 추가
        MAX_FILENAME_LENGTH = 70
        name, ext = os.path.splitext(filename)
        if len(filename) > MAX_FILENAME_LENGTH:
            name = name[:MAX_FILENAME_LENGTH - len("_short" + ext)]
            filename = name + "_short" + ext
            renamed_files.append(original_filename)

        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        uploaded_paths.append(save_path)

    # ✅ 분석은 별도 스레드에서 시작
    from threading import Thread
    for path in uploaded_paths:
        Thread(target=analyze_video, args=(path,)).start()

    # ✅ 사용자에게 알림 추가
    message = "업로드 완료되었습니다."
    if renamed_files:
        message += f" 다음 파일명은 길어 간략화되었습니다: {', '.join(renamed_files)}"

    return jsonify({'success': True, 'message': message})

#결과 반환환
@main.route("/result_images")
def result_images():
    root_dir = os.path.join("static", "predictResults")
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
    relative_path = data.get("path")  # 예: predictResults/영상폴더/img.jpg
    static_path = os.path.join("static", relative_path)

    try:
        if os.path.exists(static_path):
            os.remove(static_path)
            return jsonify({"success": True, "deleted": relative_path})
        else:
            return jsonify({"error": "파일을 찾을 수 없습니다."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@main.route('/save_results_to_db', methods=['POST'])
def save_results_to_db():
    from BackEnd.db.insert_to_db import insert_analysis_results_selected

    data = request.get_json()
    images = data.get("images", [])

    if not images:
        return jsonify({"error": "이미지 없음"}), 400

    # ✅ 경로 정규화
    first = images[0].replace("\\", "/")

    # ✅ "results_YYYYMMDD_HHMMSS" 폴더 추출 (정규표현식 사용)
    match = re.search(r"(results_\d{8}_\d{6})", first)
    if not match:
        return jsonify({"error": "결과 폴더명을 찾을 수 없습니다."}), 400

    result_folder = match.group(1)  # 예: results_20250520_142553
    meta_path = os.path.join("static", "predictResults", result_folder, "meta.json")
    
    import shutil

    # 예: static/predictResults/results_타임스탬프 → static/results/results_타임스탬프
    predict_folder = os.path.join("static", "predictResults", result_folder)
    final_folder = os.path.join("static", "results", result_folder)

    if os.path.exists(predict_folder):
        try:
            shutil.copytree(predict_folder, final_folder)
            print(f"📁 결과 폴더 복사 완료: {final_folder}")
        except Exception as e:
            print(f"❌ 결과 복사 실패: {e}")

    try:
        insert_analysis_results_selected(images, meta_path)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
