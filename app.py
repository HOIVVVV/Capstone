# app.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from flask import Flask
from routes.main import main  # 🔥 Blueprint import
from BackEnd.db.models import db      # 🔥 SQLAlchemy db 객체
from BackEnd.db.init_db import initialize_database  # 🔥 DB 초기화 함수

app = Flask(__name__)

# 폴더 설정
UPLOAD_FOLDER = 'uploads'
FRAME_FOLDER = 'frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAME_FOLDER'] = FRAME_FOLDER

# DB 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/capstone'  #비밀번호 아이디 확인할 것!
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# SQLAlchemy 초기화
db.init_app(app)

# 업로드 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Blueprint 등록
app.register_blueprint(main)

# 첫 요청 전에 DB 생성
@app.before_request
def setup():
    initialize_database()

if __name__ == '__main__':
    app.run(debug=True)
