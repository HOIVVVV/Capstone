# clear_db.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask
from db.models import db, Video, DamageImage

#DB 초기화 코드
def clear_tables(db_path='mysql+pymysql://root:1234@localhost/capstone'):
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_path
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    with app.app_context():
        print("🧹 DB 테이블 데이터 삭제 시작...")
        try:
            num_damage = db.session.query(DamageImage).delete()
            num_video = db.session.query(Video).delete()
            db.session.commit()
            print(f"✅ 삭제 완료: DamageImage {num_damage}개, Video {num_video}개")
        except Exception as e:
            db.session.rollback()
            print(f"❌ 삭제 중 오류 발생: {e}")

if __name__ == "__main__":
    clear_tables()