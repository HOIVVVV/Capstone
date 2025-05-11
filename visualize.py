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
    
    min_date = db.session.query(func.min(Video.recorded_date)).scalar()
    max_date = db.session.query(func.max(Video.recorded_date)).scalar()
    return render_template('visualize.html',
        damage_types=[d[0] for d in damage_types],
        locations=[l[0] for l in locations],
        min_date = min_date,
        max_date = max_date
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


    # elif chart_type == 'heatmap':
    #     pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
    #     if selected_locations:
    #         pivot = pivot.loc[pivot.index.intersection(selected_locations)]
    #     if selected_damage_types:
    #         pivot = pivot[selected_damage_types]
    #     sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    #     ax.set_title('위치별 손상유형 히트맵')
    elif chart_type == 'heatmap':
        pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
        if selected_locations:
            pivot = pivot.loc[pivot.index.intersection(selected_locations)]
        if selected_damage_types:
            cols = pivot.columns.intersection(selected_damage_types)
            if cols.empty:
                return '선택한 손상 유형에 해당하는 데이터가 없습니다.'
            pivot = pivot[cols]
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('위치별 손상유형 히트맵')


    # elif chart_type == 'stacked':
    #     pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
    #     if selected_locations:
    #         pivot = pivot.loc[pivot.index.intersection(selected_locations)]
    #     if selected_damage_types:
    #         pivot = pivot[selected_damage_types]
    #     pivot.plot(kind='bar', stacked=True, ax=ax)
    #     ax.set_title('Stacked Bar Chart')
    #     ax.set_ylabel('건수')
    elif chart_type == 'stacked':
        pivot = df.groupby(['location', 'damage_type']).size().unstack(fill_value=0)
        if selected_locations:
            pivot = pivot.loc[pivot.index.intersection(selected_locations)]
        if selected_damage_types:
            cols = pivot.columns.intersection(selected_damage_types)
            if cols.empty:
                return '선택한 손상 유형에 해당하는 데이터가 없습니다.'
            pivot = pivot[cols]
        pivot.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Stacked Bar Chart')
        ax.set_ylabel('건수')
    
    elif chart_type == 'map':
        # ✅ 사용자 선택 정보 가져오기
        selected_locations = request.form.getlist("location")
        selected_dtypes = request.form.getlist("damage_type")

        # ✅ DB에서 조건에 맞는 데이터 조회
        results = db.session.query(
            Video.location,
            func.count(DamageImage.image_id)
        ).join(DamageImage, Video.video_id == DamageImage.video_id)\
        .filter(Video.location.in_(selected_locations),
                DamageImage.damage_type.in_(selected_dtypes))\
        .group_by(Video.location)\
        .all()

        # ✅ 결과를 Pandas DataFrame으로 변환
        df_count = pd.DataFrame(results, columns=["구", "데이터수"])
        df_count.columns = ["location", "count"]

        # ✅ 지도 객체 생성
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

        # # GeoJSON 파일 경로
        # geojson_path = 'static/data/HangJeongDong_ver20250401.geojson'
        
        # # GeoJSON 파일 불러오기
        # gdf = gpd.read_file('static/data/HangJeongDong_ver20250401.geojson')

        # # '구' 단위로 병합 (예: sggnm 컬럼이 구 이름이라면)
        # gdf_gu = gdf.dissolve(by='sggnm', as_index=False)

        # # 병합된 GeoJSON 저장 (선택)
        # gdf_gu.to_file('static/data/Seoul_Gu_Aggregated.geojson', driver='GeoJSON')

        # 파일 로드
        with open('static/data/Seoul_Gu_Aggregated.geojson', 'r', encoding='utf-8') as f:
            seoul_gu_geo = json.load(f)
            
        # 구 이름 -> 중심 좌표 dict
        gu_centers = {}
        for feature in seoul_gu_geo["features"]:
            gu_name = feature["properties"]["sggnm"]
            geometry = shape(feature["geometry"])
            centroid = geometry.centroid
            gu_centers[gu_name] = [centroid.y, centroid.x]  # folium은 [lat, lon]

        # ✅ Choropleth로 시각화
        folium.Choropleth(
            geo_data=seoul_gu_geo,
            name="choropleth",
            data=df_count,  # pandas DataFrame (예: 구별 손상 개수)
            columns=["location", "count"],  # 'location'은 성북구, 강남구 등과 매칭되어야 함
            key_on="feature.properties.sggnm",  # 여기 수정!
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="손상 건수"
        ).add_to(m)
        
            # ✅ GeoJSON 경계선 스타일 지정
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
                fields=["sggnm"],
                aliases=["지역:"]
            )
        ).add_to(m)
        
        for idx, row in df_count.iterrows():
            gu_name = row["location"]
            count = row["count"]
            if gu_name in gu_centers:
                lat, lon = gu_centers[gu_name]
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-size: 12pt; color: black; font-weight: bold">{count}</div>"""
                    )
                ).add_to(m)



        
        map_html = m._repr_html_()
        return render_template("visualize.html", map_html=map_html)

    else:
        return '그래프 종류를 선택해주세요.'

    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
