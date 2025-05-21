import os
import json
from db.models import db, Video, DamageImage
from flask import Flask
from datetime import datetime, timedelta
from PIL import Image
import re

def insert_analysis_results(video_path, result_dir, district, recorded_date, db_path='mysql+pymysql://root:1234@localhost/capstone'):

    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_path
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    with app.app_context():
        print(f"📥 DB 데이터 삽입 시작")
        db.create_all()

        video_title = os.path.splitext(os.path.basename(video_path))[0]
        recorded_date = datetime.now().date()

        new_video = Video(
            title=video_title,
            location=district or "지역 추출 실패",
            recorded_date=recorded_date if recorded_date and recorded_date != "날짜 추출 안됨" else None
        )
        db.session.add(new_video)
        db.session.commit()

        damage_image_count = 0

        # ✅ result_dir 내부 하위 폴더 탐색 (e.g., 153000/)
        for subfolder in os.listdir(result_dir):
            subfolder_path = os.path.join(result_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            print(f"📂 하위 폴더 탐색: {subfolder_path}")
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(".jpg") and "GradCAM" not in f]
            print(f"🔍 총 이미지 탐색 대상: {len(image_files)}개")

            for file in image_files:
                file_path = os.path.join(subfolder_path, file)
                try:
                    image_title = os.path.splitext(file)[0]
                    parts = image_title.split('_')

                    # ✅ f### 형식 탐색
                    timeline_str = next((p for p in parts if re.match(r"^f\d+$", p)), None)
                    if not timeline_str:
                        raise ValueError("프레임 태그 f### 없음")

                    seconds = int(timeline_str[1:])
                    timecode = (datetime.min + timedelta(seconds=seconds)).time()
                    # top-1 라벨 추출
                    damage_info_str = image_title.split(f"{timeline_str}_")[-1]
                    damage_type = damage_info_str.split(',')[0].split('(')[0]

                except Exception as e:
                    print(f"❌ 파일 이름 분석 실패: {file} → {e}")
                    continue

                damage_image = DamageImage(
                    video_id=new_video.video_id,
                    image_title=image_title,
                    damage_type=damage_type,
                    timeline=timecode,
                    image_path=os.path.abspath(file_path)
                )
                db.session.add(damage_image)
                damage_image_count += 1
                print(f"✅ 손상 이미지 등록 완료: {image_title} | 라벨: {damage_type}, 초: {seconds}, 경로: {file_path}")

        new_video.damage_image_count = damage_image_count
        db.session.commit()
        print(f"📊 최종 등록된 손상 이미지 수: {damage_image_count}")


def insert_analysis_results_selected(image_paths, meta_path):
    if not image_paths:
        print("❌ 저장할 이미지 없음")
        return

    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.json 파일을 찾을 수 없습니다")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # ✅ Flask + DB 설정
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/capstone'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    # ✅ 손상 라벨 맵
    label_map = {
        "균열(Crack,CL),파손(Broken-Pipe,BK)": 0,
        "표면손상(Surface-Damage,SD),토사퇴적(Deposits-Silty,DS)": 1,
        "연결관-돌출(Lateral-Protruding,LP)": 2,
        "이음부-손상(Joint-Faulty,JF)": 3,
        "기타결함(Etc.,ETC)": 4,
    }

    with app.app_context():
        db.create_all()

        video_title = os.path.splitext(os.path.basename(meta["video_path"]))[0]
        recorded_date = meta.get("recorded_date")
        district = meta.get("district")

        new_video = Video(
            title=video_title,
            location=district,
            recorded_date=recorded_date if recorded_date != "날짜 추출 안됨" else None
        )
        db.session.add(new_video)
        db.session.commit()

        damage_image_count = 0

        for rel_path in image_paths:
            try:
                rel_path = rel_path.replace("\\", "/")
                file_name = os.path.basename(rel_path)
                image_title = os.path.splitext(file_name)[0]

                # f### 추출 (시간 코드용)
                timeline_match = re.search(r"f(\d+)", image_title)
                if not timeline_match:
                    print(f"❌ 프레임 태그 없음: {image_title}")
                    continue

                seconds = int(timeline_match.group(1))
                timecode = (datetime.min + timedelta(seconds=seconds)).time()

                # ✅ 손상 라벨 탐색 (포함 여부 + 먼저 매칭되는 순서)
                damage_type = None
                for label in label_map:
                    if label in image_title:
                        damage_type = label
                        print(f"✅ 매칭된 손상 타입: {damage_type}")
                        break

                if not damage_type:
                    print(f"ℹ️ 매칭되는 손상 없음, 저장 제외: {rel_path}")
                    continue

                # ✅ DB 저장
                damage_image = DamageImage(
                    video_id=new_video.video_id,
                    image_title=image_title,
                    damage_type=damage_type,  # 라벨 전체 저장
                    timeline=timecode,
                    image_path=rel_path
                )
                db.session.add(damage_image)
                damage_image_count += 1

            except Exception as e:
                print(f"❌ 이미지 분석 실패: {rel_path} → {e}")
                continue

        new_video.damage_image_count = damage_image_count
        db.session.commit()
        print(f"✅ 선택된 이미지 {damage_image_count}개 저장 완료")