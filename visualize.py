from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
import seaborn as sns
import pandas as pd
from datetime import datetime

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

@app.route('/', methods=['GET'])
def index():
    damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    locations = db.session.query(Video.location).distinct().all()
    return render_template('visualize.html',
        damage_types=[d[0] for d in damage_types],
        locations=[l[0] for l in locations]
    )

@app.route('/generate_chart', methods=['POST'])
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

    if df.empty:
        return '선택된 조건에 맞는 데이터가 없습니다.'

    fig, ax = plt.subplots(figsize=(12, 6))

    if chart_type == 'line':
        df['recorded_date'] = pd.to_datetime(df['recorded_date'])

        # ✅ 월 또는 분기 집계 단위
        if aggregate_unit == 'quarter':
            df['period'] = df['recorded_date'].dt.to_period('Q').astype(str)
        else:
            df['period'] = df['recorded_date'].dt.to_period('M').astype(str)

        # ✅ 교집합 필터링
        if selected_locations:
            df = df[df['location'].isin(selected_locations)]
        if selected_damage_types:
            df = df[df['damage_type'].isin(selected_damage_types)]

        summary = df.groupby('period').size().reset_index(name='count')
        ax.plot(summary['period'], summary['count'], marker='o')
        ax.set_title(f"{aggregate_unit.upper()}별 손상 발생 건수")
        ax.set_xlabel(aggregate_unit.upper())
        ax.set_ylabel('건수')
        plt.xticks(rotation=45)

    elif chart_type == 'bar':
        has_locations = bool(selected_locations)
        has_damage_types = bool(selected_damage_types)

        if has_locations and has_damage_types:
            summary = df.groupby(['location', 'damage_type']).size().reset_index(name='count')
            sns.barplot(data=summary, x='location', y='count', hue='damage_type', ax=ax)
        elif has_locations:
            summary = df.groupby('location').size().reset_index(name='count')
            sns.barplot(data=summary, x='location', y='count', ax=ax)
        elif has_damage_types:
            summary = df.groupby('damage_type').size().reset_index(name='count')
            sns.barplot(data=summary, x='damage_type', y='count', ax=ax)
        else:
            return '위치 또는 손상 유형 중 하나 이상 선택해주세요.'

        ax.set_title('손상 건수 통계')
        ax.set_ylabel('건수')
        plt.xticks(rotation=45)

    elif chart_type == 'pie':
        has_locations = bool(selected_locations)
        has_damage_types = bool(selected_damage_types)

        if has_locations and has_damage_types:
            # ✅ 위치별 손상유형 비율 pie 여러 개
            summary = df.groupby(['location', 'damage_type']).size().reset_index(name='count')
            locations = summary['location'].unique()

            num_charts = len(locations)
            cols = 5  # ✅ 한 줄에 최대 5개
            rows = (num_charts + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = axes.flatten()  # 2D → 1D로 평탄화

            for i, loc in enumerate(locations):
                data = summary[summary['location'] == loc]
                axes[i].pie(data['count'], labels=data['damage_type'], autopct='%1.1f%%')
                axes[i].set_title(f"{loc} 손상 유형 분포")

            # 남는 subplot 제거
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(rect=[0, 0, 1, 0.95])

        elif has_locations:
            summary = df['location'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(summary.values, labels=summary.index, autopct='%1.1f%%')
            ax.set_title("위치별 손상 분포")

        elif has_damage_types:
            summary = df['damage_type'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(summary.values, labels=summary.index, autopct='%1.1f%%')
            ax.set_title("손상 유형 분포")

        else:
            return '위치 또는 손상 유형 중 하나 이상 선택해주세요.'


    elif chart_type == 'heatmap':
        pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
        if selected_locations:
            pivot = pivot.loc[pivot.index.intersection(selected_locations)]
        if selected_damage_types:
            pivot = pivot[selected_damage_types]
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('위치별 손상유형 히트맵')

    elif chart_type == 'stacked':
        pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
        if selected_locations:
            pivot = pivot.loc[pivot.index.intersection(selected_locations)]
        if selected_damage_types:
            pivot = pivot[selected_damage_types]
        pivot.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Stacked Bar Chart')
        ax.set_ylabel('건수')

    else:
        return '그래프 종류를 선택해주세요.'

    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
