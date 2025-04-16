from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
import seaborn as sns
import pandas as pd

# 한글 폰트 설정
if platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
else:
    rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:autoset@localhost/capstone'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 모델 정의
class DamageImage(db.Model):
    __tablename__ = 'damage_images'
    image_id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.video_id'))
    image_title = db.Column(db.String(255))
    damage_type = db.Column(db.String(100))
    timeline = db.Column(db.Time)
    image_path = db.Column(db.Text)

class Video(db.Model):
    __tablename__ = 'videos'
    video_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    location = db.Column(db.String(100))
    recorded_date = db.Column(db.Date)
    damage_image_count = db.Column(db.Integer)

# 홈 페이지
@app.route('/', methods=['GET'])
def index():
    damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    locations = db.session.query(Video.location).distinct().all()
    return render_template(
        'visualize.html',
        damage_types=[d[0] for d in damage_types],
        locations=[l[0] for l in locations]
    )

# 그래프 생성
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    chart_type = request.form.get('chart_type')  # bar or pie
    selected_damage_types = request.form.getlist('damage_type')
    selected_locations = request.form.getlist('location')

    # 필터링 쿼리
    query = db.session.query(Video.location, DamageImage.damage_type).join(DamageImage)

    if selected_damage_types:
        query = query.filter(DamageImage.damage_type.in_(selected_damage_types))
    if selected_locations:
        query = query.filter(Video.location.in_(selected_locations))

    results = query.all()

    # 데이터프레임으로 변환
    data = [{'location': loc, 'damage_type': dtype} for loc, dtype in results]
    df = pd.DataFrame(data)

    if df.empty:
        return '선택된 조건에 맞는 데이터가 없습니다.'

    fig, ax = plt.subplots(figsize=(12, 6))

    if chart_type == 'bar':
        summary = df.groupby(['location', 'damage_type']).size().reset_index(name='count')
        sns.barplot(data=summary, x='location', y='count', hue='damage_type', ax=ax)
        ax.set_title('촬영 위치별 손상 유형 건수')
        ax.set_ylabel('건수')
        ax.set_xlabel('촬영 위치')
        plt.xticks(rotation=45)

    elif chart_type == 'pie':
        summary = df['damage_type'].value_counts()
        ax.pie(summary.values, labels=summary.index, autopct='%1.1f%%')
        ax.set_title('손상 유형별 비율')

    # 이미지 전송
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
