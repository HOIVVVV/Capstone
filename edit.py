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

#경로설정
path = "upload.html"

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
    return render_template(path,
        damage_types=[d[0] for d in damage_types],
        locations=[l[0] for l in locations],
        min_date = min_date,
        max_date = max_date
    )
    
@app.route('/mapping')
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
    
    # ✅ 전체 선택 옵션 제공을 위해 전체 리스트도 다시 전달
    # damage_types = df['damage_type'].unique().tolist()
    # locations = df['location'].unique().tolist()
    # ✅ 전체 손상 유형과 위치 가져오기
    all_damage_types = db.session.query(DamageImage.damage_type).distinct().all()
    all_locations = db.session.query(Video.location).distinct().all()
    damage_types = [d[0] for d in all_damage_types]
    locations = [l[0] for l in all_locations]
    min_date = df['recorded_date'].min().strftime('%Y-%m-%d')
    max_date = df['recorded_date'].max().strftime('%Y-%m-%d')

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

        # ✅ 각 구에 마커 추가 (손상 건수 숫자 표시)
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



    else:
        return '그래프 종류를 선택해주세요.'

    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/front')
def front_page():
    return render_template('front.html')



if __name__ == '__main__':
    app.run(debug=True)
