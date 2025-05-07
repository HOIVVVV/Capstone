from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pymysql

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:autoset@localhost/capstone'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 1. MySQL에 접속해서 capstone DB와 테이블이 없으면 생성
def initialize_database():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='autoset',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            # 데이터베이스 생성
            cursor.execute("CREATE DATABASE IF NOT EXISTS capstone")
            connection.commit()

        # capstone DB 접속 후 테이블 생성
        connection.select_db('capstone')
        with connection.cursor() as cursor:
            # videos 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    location VARCHAR(100),
                    recorded_date DATE,
                    damage_image_count INT DEFAULT 0
                )
            """)
            # damage_images 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS damage_images (
                    image_id INT AUTO_INCREMENT PRIMARY KEY,
                    video_id INT,
                    image_title VARCHAR(255),
                    damage_type VARCHAR(100),
                    timeline TIME,
                    image_path TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """)
            connection.commit()
    finally:
        connection.close()

# 2. SQLAlchemy 모델 정의
class Video(db.Model):
    __tablename__ = 'videos'
    video_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(100))
    recorded_date = db.Column(db.Date)
    damage_image_count = db.Column(db.Integer, default=0)

class DamageImage(db.Model):
    __tablename__ = 'damage_images'
    image_id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.video_id', ondelete='CASCADE'))
    image_title = db.Column(db.String(255))
    damage_type = db.Column(db.String(100))
    timeline = db.Column(db.Time)
    image_path = db.Column(db.Text)

# 3. Flask 시작 전 DB 초기화
@app.before_first_request
def setup():
    initialize_database()

if __name__ == '__main__':
    app.run(debug=True)
