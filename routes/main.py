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

from flask import Flask, request, jsonify, send_from_directory, render_template
from BackEnd.db.db_config import get_connection
import traceback

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
    
    
@main.route('/edit_image.html')
def edit_image():
    return render_template('edit_image.html')

@main.route('/edit_video.html')
def edit_video_page():
    return render_template('edit_video.html')


@main.route('/api/options')
def get_options():
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT damage_type FROM damage_images ORDER BY damage_type")
            damage_types = [row['damage_type'] for row in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT location FROM videos ORDER BY location")
            locations = [row['location'] for row in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT title FROM videos ORDER BY title")
            titles = [row['title'] for row in cursor.fetchall()]

        return jsonify({
            'damage_types': damage_types,
            'locations': locations,
            'titles': titles
        })
    except Exception as e:
        main.logger.error(f"Error loading options: {str(e)}")
        return jsonify({'error': '옵션 데이터를 불러오는 중 오류가 발생했습니다.'}), 500

    
def extract_static_path(full_path):
    """
    절대 경로에서 'static' 폴더 이후의 경로만 추출하고,
    웹에서 쓸 수 있도록 슬래시(`/`)로 변환함
    """
    match = re.search(r'static[/\\](.+)$', full_path)
    if match:
        return match.group(1).replace('\\', '/')
    return full_path.replace('\\', '/')


@main.route('/api/images', methods=['GET'])
def get_images():
    try:
        damage_type = request.args.get('damage_type')
        location = request.args.get('location')
        title = request.args.get('title')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        query = """
            SELECT d.image_id, d.image_title, d.damage_type, d.timeline, 
                   v.title AS video_title, v.location, v.recorded_date
            FROM damage_images d
            JOIN videos v ON d.video_id = v.video_id
            WHERE 1=1
        """
        params = []

        if damage_type:
            query += " AND d.damage_type = %s"
            params.append(damage_type)
        if location:
            query += " AND v.location = %s"
            params.append(location)
        if title:
            query += " AND v.title = %s"
            params.append(title)
        if start_date and end_date:
            query += " AND v.recorded_date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        elif start_date:
            query += " AND v.recorded_date >= %s"
            params.append(start_date)
        elif end_date:
            query += " AND v.recorded_date <= %s"
            params.append(end_date)

        count_query = f"SELECT COUNT(*) as total FROM ({query}) as subquery"

        query += " ORDER BY v.recorded_date DESC, d.image_title"
        query += " LIMIT %s OFFSET %s"
        offset = (page - 1) * per_page
        params.extend([per_page, offset])

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(count_query, params[:-2])
            total = cursor.fetchone()['total']

            cursor.execute(query, params)
            results = cursor.fetchall()

            for row in results:

                if row['recorded_date']:
                    row['recorded_date'] = row['recorded_date'].strftime('%Y-%m-%d')
                if row['timeline']:
                    row['timeline'] = str(row['timeline'])

        pagination = {
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }

        return jsonify({
            'images': results,
            'pagination': pagination
        })

    except Exception as e:
        main.logger.error(f"Error retrieving images: {str(e)}")
        main.logger.error(traceback.format_exc())
        return jsonify({'error': '이미지 조회 중 오류가 발생했습니다.'}), 500

@main.route('/api/images/<int:image_id>', methods=['GET'])
def get_image(image_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT d.*, v.title AS video_title, v.location, v.recorded_date
                FROM damage_images d
                JOIN videos v ON d.video_id = v.video_id
                WHERE d.image_id = %s
            """, (image_id,))
            image = cursor.fetchone()

            if not image:
                return jsonify({'error': '해당 이미지를 찾을 수 없습니다.'}), 404

            # image['image_exists'] = os.path.exists(os.path.join("static", image['image_path']))

            relative_path = extract_static_path(image['image_path'])
            image['image_exists'] = os.path.exists(os.path.join("static", relative_path))
            # 클라이언트에게 보내는 경로도 처리 (선택사항)
            image['image_path'] = relative_path

            if image['recorded_date']:
                image['recorded_date'] = image['recorded_date'].strftime('%Y-%m-%d')
            if image['timeline']:
                image['timeline'] = str(image['timeline'])

            return jsonify({'image': image})

    except Exception as e:
        main.logger.error(f"Error loading image: {str(e)}")
        return jsonify({'error': '이미지 정보 로딩 중 오류가 발생했습니다.'}), 500

@main.route('/api/images/<int:image_id>', methods=['PUT'])
def update_image(image_id):
    try:
        data = request.json
        image_title = data.get('image_title')
        damage_type = data.get('damage_type')

        if not all([image_title, damage_type]):
            return jsonify({'error': '이미지 제목과 손상 유형을 입력해야 합니다.'}), 400

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM damage_images WHERE image_id = %s", (image_id,))
            if cursor.fetchone()['count'] == 0:
                return jsonify({'error': '해당 이미지 ID가 존재하지 않습니다.'}), 404

            cursor.execute("""
                UPDATE damage_images
                SET image_title = %s, damage_type = %s
                WHERE image_id = %s
            """, (image_title, damage_type, image_id))
            conn.commit()

        return jsonify({'message': '이미지 정보가 성공적으로 수정되었습니다.'})

    except Exception as e:
        main.logger.error(f"Error updating image: {str(e)}")
        return jsonify({'error': '이미지 정보 수정 중 오류가 발생했습니다.'}), 500


@main.route('/api/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT image_path, video_id FROM damage_images WHERE image_id = %s", (image_id,))
            row = cursor.fetchone()
            if not row:
                return jsonify({'error': '해당 이미지가 존재하지 않습니다.'}), 404

            image_path = row['image_path']
            video_id = row['video_id']

            cursor.execute("DELETE FROM damage_images WHERE image_id = %s", (image_id,))
            cursor.execute("""
                UPDATE videos
                SET damage_image_count = damage_image_count - 1
                WHERE video_id = %s AND damage_image_count > 0
            """, (video_id,))


            cursor.execute("SELECT damage_image_count FROM videos WHERE video_id = %s", (video_id,))
            count = cursor.fetchone()
            if count and count['damage_image_count'] == 0:
                cursor.execute("DELETE FROM videos WHERE video_id = %s", (video_id,))

            conn.commit()

        # full_path = os.path.join("static", image_path)
        relative_path = extract_static_path(image_path)
        full_path = os.path.join("static", relative_path)
        if os.path.exists(full_path):
            os.remove(full_path)

        return jsonify({'message': '이미지가 성공적으로 삭제되었습니다.'})

    except Exception as e:
        main.logger.error(f"Error deleting image: {str(e)}")
        return jsonify({'error': '이미지 삭제 중 오류가 발생했습니다.'}), 500



@main.route('/api/videos', methods=['GET'])
def get_videos():
    try:
        title = request.args.get('title')
        location = request.args.get('location')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        query = """
            SELECT video_id, title, location, recorded_date, damage_image_count
            FROM videos
            WHERE 1=1
        """
        params = []

        if title:
            query += " AND title = %s"
            params.append(title)
        if location:
            query += " AND location = %s"
            params.append(location)
        if start_date and end_date:
            query += " AND recorded_date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        elif start_date:
            query += " AND recorded_date >= %s"
            params.append(start_date)
        elif end_date:
            query += " AND recorded_date <= %s"
            params.append(end_date)

        count_query = f"SELECT COUNT(*) AS total FROM ({query}) AS subquery"

        query += " ORDER BY recorded_date DESC"
        query += " LIMIT %s OFFSET %s"
        offset = (page - 1) * per_page
        params.extend([per_page, offset])

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(count_query, params[:-2])
            total = cursor.fetchone()['total']

            cursor.execute(query, params)
            videos = cursor.fetchall()

        pagination = {
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }

        return jsonify({'videos': videos, 'pagination': pagination})

    except Exception as e:
        main.logger.error(f"영상 정보 조회 오류: {str(e)}")
        return jsonify({'error': '영상 조회 중 오류가 발생했습니다.'}), 500

@main.route('/api/videos/<int:video_id>', methods=['PUT'])
def update_video(video_id):
    try:
        data = request.json
        title = data.get('title')
        location = data.get('location')
        recorded_date = data.get('recorded_date')

        if not all([title, location, recorded_date]):
            return jsonify({'error': '영상 제목, 촬영 위치, 촬영 날짜를 모두 입력해야 합니다.'}), 400

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM videos WHERE video_id = %s", (video_id,))
            if cursor.fetchone()['count'] == 0:
                return jsonify({'error': '해당 영상 ID가 존재하지 않습니다.'}), 404

            cursor.execute("""
                UPDATE videos
                SET title = %s, location = %s, recorded_date = %s
                WHERE video_id = %s
            """, (title, location, recorded_date, video_id))
            conn.commit()

        return jsonify({'message': '영상 정보가 성공적으로 수정되었습니다.'})

    except Exception as e:
        main.logger.error(f"Error updating video: {str(e)}")
        return jsonify({'error': '영상 정보 수정 중 오류가 발생했습니다.'}), 500


@main.route('/api/videos/<int:video_id>', methods=['GET'])
def get_video(video_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT video_id, title, location, recorded_date, damage_image_count
                FROM videos
                WHERE video_id = %s
            """, (video_id,))
            video = cursor.fetchone()

            if not video:
                return jsonify({'error': '해당 영상을 찾을 수 없습니다.'}), 404

            if video['recorded_date']:
                video['recorded_date'] = video['recorded_date'].strftime('%Y-%m-%d')

            return jsonify({'video': video})

    except Exception as e:
        main.logger.error(f"영상 정보 조회 오류: {str(e)}")
        return jsonify({'error': '영상 정보를 불러오는데 실패했습니다.'}), 500


@main.route('/api/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # 영상과 연결된 이미지 정보 조회
            cursor.execute("SELECT image_path FROM damage_images WHERE video_id = %s", (video_id,))
            images = cursor.fetchall()

            # 이미지 파일 삭제
            for image in images:
                # image_path = os.path.join("static", image['image_path'])
                relative_path = extract_static_path(image['image_path'])
                image_path = os.path.join("static", relative_path)
                if os.path.exists(image_path):
                    os.remove(image_path)

            # 데이터베이스에서 이미지 삭제
            cursor.execute("DELETE FROM damage_images WHERE video_id = %s", (video_id,))

            # 영상 삭제
            cursor.execute("DELETE FROM videos WHERE video_id = %s", (video_id,))

            conn.commit()

            return jsonify({'message': '영상이 성공적으로 삭제되었습니다.'})

    except Exception as e:
        main.logger.error(f"영상 삭제 오류: {str(e)}")
        main.logger.error(traceback.format_exc())
        return jsonify({'error': '영상 삭제 중 오류가 발생했습니다.'}), 500



