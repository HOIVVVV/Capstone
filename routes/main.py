# routes/main.py
import os
import sys
import re
import shutil
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
from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
import seaborn as sns
import pandas as pd
from datetime import datetime
from sqlalchemy import func
import folium
import json
import geopandas as gpd
from shapely.geometry import shape
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    chart_type = request.form.get('chart_type')
    aggregate_unit = request.form.get('aggregate_unit')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    query = db.session.query(Video.recorded_date, Video.location, DamageImage.damage_type).join(DamageImage)

    # ✅ 날짜 필터링 적용
    if start_date:
        query = query.filter(Video.recorded_date >= start_date)
    if end_date:
        query = query.filter(Video.recorded_date <= end_date)

    results = query.all()

    if results:
        data = [{'recorded_date': rec, 'location': loc, 'damage_type': dtype} for rec, loc, dtype in results]
        df = pd.DataFrame(data)
        df['recorded_date'] = pd.to_datetime(df['recorded_date'], errors='coerce')  # datetime 변환
    else:
        df = pd.DataFrame(columns=['recorded_date', 'location', 'damage_type'])

    # ✅ 날짜 범위 계산
    if not df.empty and df['recorded_date'].notna().any():
        min_date = df['recorded_date'].min().strftime('%Y-%m-%d')
        max_date = df['recorded_date'].max().strftime('%Y-%m-%d')
    else:
        min_date = max_date = ''

    # ✅ 전체 손상 유형과 위치 가져오기
    all_damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    all_locations = db.session.query(Video.location).distinct().all()
    damage_types = [d[0] for d in all_damage_types]
    locations = [l[0] for l in all_locations]

    # ✅ 차트 1: pieChart -> 손상유형통계
    summary = df['damage_type'].value_counts().reset_index()
    summary.columns = ['damage_type', 'count']
    fig1 = px.pie(summary, values='count', names='damage_type', title='손상 유형 분포', hole=0.3)
    chart_html1 = fig1.to_html(full_html=False)

    # ✅ 차트 2: stacked -> 지역별 손상빈도
    pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
    fig2 = go.Figure()
    for damage_type in pivot.columns:
        fig2.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[damage_type],
            name=damage_type
        ))
    fig2.update_layout(
        barmode='stack',
        title='Stacked Bar Chart',
        xaxis_title='위치',
        yaxis_title='건수'
    )
    chart_html2 = fig2.to_html(full_html=False)

    # ✅ 차트 3: line -> 손상발생빈도 (월별)
    if not df.empty and pd.api.types.is_datetime64_any_dtype(df['recorded_date']):
        df_valid = df[df['recorded_date'].notna()].copy()
        df_valid['period'] = df_valid['recorded_date'].dt.to_period('M').astype(str)
        summary = df_valid.groupby('period').size().reset_index(name='count')
        fig3 = px.line(summary, x='period', y='count')
        chart_html3 = fig3.to_html(full_html=False)
    else:
        chart_html3 = "<p style='text-align:center;'>유효한 날짜 데이터가 없습니다.</p>"

    return render_template('dashboard.html',
                           chart_html1=chart_html1,
                           chart_html2=chart_html2,
                           chart_html3=chart_html3,
                           image_results=image_results)



@main.route('/info')
def system_info():
    return render_template('info.html')

@main.route("/result")
def result():
    return render_template("result.html", active_page="result")

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
        
@main.route('/api/recent_images')
def recent_images():
    base_folder = os.path.join("static", "predictResults")

    # results_로 시작하는 하위 폴더 중 가장 최근 것 찾기
    all_result_folders = sorted(
        [f for f in os.listdir(base_folder) if f.startswith("results_")],
        reverse=True
    )

    if not all_result_folders:
        return jsonify({"images": []})

    latest_folder = all_result_folders[0]
    latest_folder_path = os.path.join(base_folder, latest_folder)

    # 하위 모든 이미지 찾기 (.jpg, .png 등)
    image_files = []
    for root, _, files in os.walk(latest_folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), "static")
                image_files.append("/static/" + rel_path.replace("\\", "/"))

    # 최신 이미지 12개만 리턴 (파일 이름 기준 정렬)
    image_files = sorted(image_files, reverse=True)[:12]

    return jsonify({"images": image_files})
        
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

    # ✅ "results_YYYYMMDD_HHMMSS" 폴더 추출
    match = re.search(r"(results_\d{8}_\d{6})", first)
    if not match:
        return jsonify({"error": "결과 폴더명을 찾을 수 없습니다."}), 400

    result_folder = match.group(1)
    meta_path = os.path.join("static", "results", result_folder, "meta.json")

    import shutil
    predict_folder = os.path.join("static", "predictResults", result_folder)
    final_folder = os.path.join("static", "results", result_folder)

    if os.path.exists(predict_folder):
        try:
            shutil.copytree(predict_folder, final_folder)
            print(f"📁 결과 폴더 복사 완료: {final_folder}")
        except Exception as e:
            print(f"❌ 결과 복사 실패: {e}")

    # ✅ 경로를 results 기준으로 교체
    corrected_images = [img.replace("predictResults", "results") for img in images]

    try:
        insert_analysis_results_selected(corrected_images, meta_path)
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

@main.route('/reset_progress', methods=['POST'])
def reset_progress():
    progress["step"] = "대기 중"
    progress["percent"] = 0
    progress["current_file"] = ""
    progress["done"] = False
    return jsonify({"success": True})

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
            # ✅ 해당 영상의 results 폴더 삭제 시도
            cursor.execute("SELECT title FROM videos WHERE video_id = %s", (video_id,))
            video_row = cursor.fetchone()

            if video_row:
                video_title = video_row['title']
                result_root = "static/results"

                for folder_name in os.listdir(result_root):
                    folder_path = os.path.join(result_root, folder_name)
                    video_folder_path = os.path.join(folder_path, video_title)

                    if os.path.exists(video_folder_path) and os.path.isdir(video_folder_path):
                        try:
                            # ✅ 영상 폴더 삭제
                            shutil.rmtree(video_folder_path)
                            print(f"🗑️ 영상 폴더 삭제 완료: {video_folder_path}")

                            # ✅ 결과 상위 폴더도 무조건 삭제
                            shutil.rmtree(folder_path)
                            print(f"🗑️ 상위 결과 폴더도 삭제됨: {folder_path}")

                        except Exception as e:
                            print(f"❌ 삭제 실패: {e}")

        
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
    
    
########################################################### 최규영
    
    # 한글 폰트 설정
if platform.system() == 'Windows':
    
    rc('font', family='Malgun Gothic')
else:
    rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False

#경로설정
path = "stats.html"   

@main.route('/stats', methods=['GET'])
def stats_page():
    damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    locations = db.session.query(Video.location).distinct().all()
    
    min_date = db.session.query(func.min(Video.recorded_date)).scalar()
    max_date = db.session.query(func.max(Video.recorded_date)).scalar()
    return render_template(path,
        damage_types=[d[0] for d in damage_types],
        locations=[l[0] for l in locations],
        min_date = min_date,
        max_date = max_date
    )
    
@main.route('/mapping')
def mapping_page():
    damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    locations = db.session.query(Video.location).distinct().all()
    
    min_date = db.session.query(func.min(Video.recorded_date)).scalar()
    max_date = db.session.query(func.max(Video.recorded_date)).scalar()
    return render_template('mapping.html',
        damage_types=[d[0] for d in damage_types],
        locations=[l[0] for l in locations],
        min_date = min_date,
        max_date = max_date
    )

@main.route('/generate_chart', methods=['POST'])
def generate_chart():
    chart_type = request.form.get('chart_type')
    selected_damage_types = request.form.getlist('damage_type')
    selected_locations = request.form.getlist('location')
    aggregate_unit = request.form.get('aggregate_unit')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    query = db.session.query(Video.recorded_date, Video.location, DamageImage.damage_type).join(DamageImage)

    if selected_damage_types:
        query = query.filter(DamageImage.damage_type.in_(selected_damage_types))
    if selected_locations:
        query = query.filter(Video.location.in_(selected_locations))

    # ✅ 날짜 범위 필터링
    if start_date:
        query = query.filter(Video.recorded_date >= start_date)
    if end_date:
        query = query.filter(Video.recorded_date <= end_date)

    results = query.all()
    data = [{'recorded_date': rec, 'location': loc, 'damage_type': dtype} for rec, loc, dtype in results]
    df = pd.DataFrame(data)
    
    # ✅ 전체 선택 옵션 제공을 위해 전체 리스트도 다시 전달
    # damage_types = df['damage_type'].unique().tolist()
    # locations = df['location'].unique().tolist()
    # ✅ 전체 손상 유형과 위치 가져오기
    all_damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    all_locations = db.session.query(Video.location).distinct().all()
    damage_types = [d[0] for d in all_damage_types]
    locations = [l[0] for l in all_locations]
    # min_date = df['recorded_date'].min().strftime('%Y-%m-%d')
    # max_date = df['recorded_date'].max().strftime('%Y-%m-%d')
    min_date = start_date
    max_date = end_date

    if df.empty:
        return render_template(path, chart_html="<p style='text-align:center;'>선택된 조건에 맞는 데이터가 없습니다.</p>",
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,
                           locations=locations,
                           min_date=min_date,
                           max_date=max_date,
                           chart_type=chart_type)


    fig, ax = plt.subplots(figsize=(12, 6))

    if chart_type == 'line':
        df['recorded_date'] = pd.to_datetime(df['recorded_date'])

        # ✅ 월 또는 분기 집계 단위
        if aggregate_unit == 'quarter':
            df['period'] = df['recorded_date'].dt.to_period('Q').astype(str)
        else:#MONTH 
            df['period'] = df['recorded_date'].dt.to_period('M').astype(str)

        # ✅ 교집합 필터링
        if selected_locations:
            df = df[df['location'].isin(selected_locations)]
        if selected_damage_types:
            df = df[df['damage_type'].isin(selected_damage_types)]

        summary = df.groupby('period').size().reset_index(name='count')
        fig = px.line(summary, x = 'period', y= 'count')
        chart_html =  fig.to_html(full_html=False)
        return render_template(path, chart_html=chart_html,
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)

    elif chart_type == 'bar':
        has_locations = bool(selected_locations)
        has_damage_types = bool(selected_damage_types)

        if has_locations and has_damage_types:
            summary = df.groupby(['location', 'damage_type']).size().reset_index(name='count')
            fig = px.bar(summary, x='location', y='count', color='damage_type', barmode='group', title="손상 건수 통계")
        elif has_locations:
            summary = df.groupby('location').size().reset_index(name='count')
            fig = px.bar(summary, x='location', y='count', title="손상 건수 통계")
        elif has_damage_types:
            summary = df.groupby('damage_type').size().reset_index(name='count')
            fig = px.bar(summary, x='damage_type', y='count', title="손상 건수 통계")
        else:
            return '위치 또는 손상 유형 중 하나 이상 선택해주세요.'
        chart_html =  fig.to_html(full_html=False)
        return render_template(path, chart_html=chart_html,
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)

    elif chart_type == 'pie':
        has_locations = bool(selected_locations)
        has_damage_types = bool(selected_damage_types)

        if has_locations and has_damage_types:
            # ✅ 위치별 손상유형 비율 pie 여러 개
            summary = df.groupby(['location', 'damage_type']).size().reset_index(name='count')
            pie_locations = summary['location'].unique()

            num_charts = len(pie_locations)
            cols = 5  # 한 줄에 최대 5개
            rows = (num_charts + cols - 1) // cols

            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)],
                subplot_titles=[f"{loc} 손상 유형 분포" for loc in pie_locations]
            )

            for i, loc in enumerate(pie_locations):
                row = i // cols + 1
                col = i % cols + 1
                data = summary[summary['location'] == loc]
                fig.add_trace(
                    go.Pie(labels=data['damage_type'], values=data['count'], name=loc, hole=0.3),
                    row=row, col=col
                )

            fig.update_layout(title_text="위치별 손상 유형 비율", height=300 * rows)

        elif has_locations:
            summary = df['location'].value_counts().reset_index()
            summary.columns = ['location', 'count']
            fig = px.pie(summary, values='count', names='location', title='위치별 손상 분포', hole=0.3)

        elif has_damage_types:
            summary = df['damage_type'].value_counts().reset_index()
            summary.columns = ['damage_type', 'count']
            fig = px.pie(summary, values='count', names='damage_type', title='손상 유형 분포', hole=0.3)

        else:
            return render_template(path, chart_html="<p style='text-align:center;'>위치 또는 손상 유형 중 하나 이상 선택해주세요.</p>",
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)

        # ✅ HTML로 변환하여 템플릿에 전달
        chart_html = fig.to_html(full_html=False)
        return render_template(path, chart_html=chart_html,
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)

    elif chart_type == 'heatmap':
        pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)

        if selected_locations:
            pivot = pivot.loc[pivot.index.intersection(selected_locations)]

        if selected_damage_types:
            cols = pivot.columns.intersection(selected_damage_types)
            if cols.empty:
                return render_template(path, chart_html="<p style='text-align:center;'>선택한 손상 유형에 해당하는 데이터가 없습니다.</p>",
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)
            pivot = pivot[cols]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='YlOrRd',
            colorbar=dict(title='건수'),
            hoverongaps=False
        ))

        fig.update_layout(title='위치별 손상유형 히트맵', xaxis_title="손상 유형", yaxis_title="위치")

        chart_html = fig.to_html(full_html=False)
        return render_template(path, chart_html=chart_html,
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)
        
    elif chart_type == 'stacked':
        pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)

        if selected_locations:
            pivot = pivot.loc[pivot.index.intersection(selected_locations)]

        if selected_damage_types:
            cols = pivot.columns.intersection(selected_damage_types)
            if cols.empty:
                return render_template(path, chart_html="<p style='text-align:center;'>선택한 손상 유형에 해당하는 데이터가 없습니다.</p>",
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)
            pivot = pivot[cols]

        fig = go.Figure()

        for damage_type in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot.index,
                y=pivot[damage_type],
                name=damage_type
            ))

        fig.update_layout(
            barmode='stack',
            title='Stacked Bar Chart',
            xaxis_title='위치',
            yaxis_title='건수'
        )

        chart_html = fig.to_html(full_html=False)
        return render_template(path, chart_html=chart_html,
                           start_date=start_date,
                           end_date=end_date,
                           selected_damage_types=selected_damage_types,
                           selected_locations=selected_locations,
                           damage_types=damage_types,   # 전체 리스트
                           locations=locations,         # 전체 리스트
                           min_date=min_date,
                           max_date=max_date,
                           chart_type = chart_type)
    
    elif chart_type == 'map':
        import requests

        # ✅ 사용자 선택 정보 가져오기
        selected_locations = request.form.getlist("location")
        selected_damage_types = request.form.getlist("damage_type")

        # ✅ DB에서 조건에 맞는 데이터 조회 (손상 수 있는 구만 집계됨)
        results = db.session.query(
            Video.location,
            func.count(DamageImage.image_id)
        ).join(DamageImage, Video.video_id == DamageImage.video_id)\
        .filter(Video.location.in_(selected_locations),
                DamageImage.damage_type.in_(selected_damage_types))\
        .group_by(Video.location)\
        .all()

        # ✅ DB 결과를 DataFrame으로 변환
        df_result = pd.DataFrame(results, columns=["location", "count"])

        # ✅ 선택된 location 리스트를 기준으로 0 포함 데이터프레임 생성
        df_all_locations = pd.DataFrame(selected_locations, columns=["location"])

        # ✅ 병합하여 손상 수 없는 location은 count = 0으로 처리
        df_count = pd.merge(df_all_locations, df_result, on="location", how="left").fillna(0)
        df_count["count"] = df_count["count"].astype(int)

        # ✅ 지도 객체 생성
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

        # ✅ 서울시 구 단위 GeoJSON 불러오기 (GitHub에서 직접 요청)
        url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
        response = requests.get(url)
        if response.status_code != 200:
            return "GeoJSON 파일 로드 실패", 500

        seoul_gu_geo = response.json()

        # ✅ 구 이름 -> 중심 좌표 dict 생성
        gu_centers = {}
        for feature in seoul_gu_geo["features"]:
            gu_name = feature["properties"]["name"]
            geometry = shape(feature["geometry"])
            centroid = geometry.centroid
            gu_centers[gu_name] = [centroid.y, centroid.x]  # folium은 [lat, lon]

        # ✅ Choropleth 시각화
        folium.Choropleth(
            geo_data=seoul_gu_geo,
            name="choropleth",
            data=df_count,
            columns=["location", "count"],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="손상 건수"
        ).add_to(m)

        # ✅ GeoJSON 경계선 스타일 + 툴팁 추가
        folium.GeoJson(
            seoul_gu_geo,
            name="구 경계",
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 1.5,
                'dashArray': '5, 5'
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["지역:"]
            )
        ).add_to(m)

        # # ✅ 각 구에 마커 추가 (손상 건수 숫자 표시)
        # for idx, row in df_count.iterrows():
        #     gu_name = row["location"]
        #     count = row["count"]
        #     if gu_name in gu_centers:
        #         lat, lon = gu_centers[gu_name]
        #         folium.Marker(
        #             location=[lat, lon],
        #             icon=folium.DivIcon(
        #                 html=f"""<div style="font-size: 12pt; color: black; font-weight: bold">{count}</div>"""
        #             )
        #         ).add_to(m)
        # ✅ 각 구에 마커 추가 (구 이름 + 손상 건수 숫자 표시, 가로 출력)
        
        for idx, row in df_count.iterrows():
            gu_name = row["location"]
            count = row["count"]
            if gu_name in gu_centers:
                lat, lon = gu_centers[gu_name]
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f"""
                            <div style="text-align: center; white-space: nowrap;">
                                <span style="font-size: 10pt; color: black;">{gu_name}</span><br>
                                <span style="font-size: 12pt; color: black; font-weight: bold;">{count}</span>
                            </div>
                        """
                    )
                ).add_to(m)



        # ✅ chart HTML 렌더링
        chart_html = m._repr_html_()

        # ✅ 전체 damage_types, locations, 날짜 범위 추출
        all_damage_types = db.session.query(DamageImage.damage_type).distinct().all()
        all_locations = db.session.query(Video.location).distinct().all()
        damage_types = [d[0] for d in all_damage_types]
        locations = [l[0] for l in all_locations]

        # ✅ 날짜 범위 계산
        min_date_query = db.session.query(func.min(Video.recorded_date)).scalar()
        max_date_query = db.session.query(func.max(Video.recorded_date)).scalar()
        min_date = min_date_query.strftime('%Y-%m-%d') if min_date_query else ''
        max_date = max_date_query.strftime('%Y-%m-%d') if max_date_query else ''

        return render_template(path,
                               chart_html=chart_html,
                               start_date=start_date,
                               end_date=end_date,
                               selected_damage_types=selected_damage_types,
                               selected_locations=selected_locations,
                               damage_types=damage_types,
                               locations=locations,
                               min_date=min_date,
                               max_date=max_date,
                               chart_type=chart_type)



