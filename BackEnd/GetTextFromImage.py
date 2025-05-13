import os
import easyocr
from PIL import Image
import numpy as np
import shutil
import re
from datetime import datetime
from geopy.geocoders import Nominatim
from collections import Counter
import cv2

reader = easyocr.Reader(['ko', 'en'])

SEOUL_DISTRICTS = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구"
]

def get_seoul_district(lat, lon):
    try:
        geolocator = Nominatim(user_agent="seoul-district-finder")
        location = geolocator.reverse((lat, lon), language='ko')
        if location and location.raw:
            address = location.raw.get('address', {})
            if any("서울" in str(v) for v in address.values()):
                for val in address.values():
                    if val in SEOUL_DISTRICTS:
                        return val
        return "비서울"
    except:
        return "비서울"

def extract_date_and_coordinates(text_lines):
    date_patterns = [
        r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})',
        r'(\d{2})[./-](\d{1,2})[./-](\d{1,2})',
        r'(\d{1,2})[\uC6D4][\s]*(\d{1,2})[\uC77C]',
        r'(\d{4})\uB144[\s]*(\d{1,2})\uC6D4[\s]*(\d{1,2})\uC77C',
        r'(\d{4})(\d{2})(\d{2})'
    ]
    coord_patterns = [
        r'\uC704\uB3C4[^\d]*(\d+\.?\d*)[^\d]*\uACBD\uB3C4[^\d]*(\d+\.?\d*)',
        r'lat[^\d]*(\d+\.?\d*)[^\d]*lon[^\d]*(\d+\.?\d*)',
        r'latitude[^\d]*(\d+\.?\d*)[^\d]*longitude[^\d]*(\d+\.?\d*)',
        r'\b(3[7-8]\.\d+)[^\d\n]+(12[6-8]\.\d+)\b'
    ]

    extracted_info = {
        'date': None,
        'coordinates': {'lat': None, 'lon': None},
        'district': None
    }

    full_text = ' '.join(text_lines)

    for pattern in date_patterns:
        matches = re.findall(pattern, full_text)
        if matches:
            match = matches[0]
            try:
                if len(match) == 3:
                    if len(match[0]) == 4:
                        date_str = f"{match[0]}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                    else:
                        year = f"20{match[0]}" if int(match[0]) < 50 else f"19{match[0]}"
                        date_str = f"{year}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                    extracted_info['date'] = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                elif len(match) == 2:
                    year = datetime.now().year
                    date_str = f"{year}-{match[0].zfill(2)}-{match[1].zfill(2)}"
                    extracted_info['date'] = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                break
            except:
                continue

    for pattern in coord_patterns:
        matches = re.findall(pattern, full_text)
        if matches:
            try:
                lat, lon = float(matches[0][0]), float(matches[0][1])
                extracted_info['coordinates']['lat'] = lat
                extracted_info['coordinates']['lon'] = lon
                break
            except:
                continue

    # 날짜 기본값
    if not extracted_info['date']:
        extracted_info['date'] = "날짜 추출 안됨"

    # 지역 판단
    lat, lon = extracted_info['coordinates']['lat'], extracted_info['coordinates']['lon']
    if lat is None or lon is None:
        extracted_info['district'] = "지역 추출 실패"
    else:
        district = get_seoul_district(lat, lon)
        extracted_info['district'] = district if district in SEOUL_DISTRICTS else "비서울"
        
    return extracted_info

def analyze_image_from_path(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_cv = np.array(image)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        result = reader.readtext(thresh, detail=0)
        info = extract_date_and_coordinates(result)

        return {
            'image': os.path.basename(image_path),
            'date': info['date'],
            'district': info['district']
        }

    except Exception as e:
        return {'image': os.path.basename(image_path), 'error': str(e)}
    
def analyze_images_in_folder(folder_path, result_output_folder, frame_output_folder):
    import os
    import shutil
    from collections import Counter
    from .GetTextFromImage import analyze_image_from_path

    results = []

    print(folder_path)
    print(result_output_folder)
    print(frame_output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            result = analyze_image_from_path(image_path)
            results.append(result)
            # ✅ 이미지별 추출 로그 표시
            print(f"📷 {result['image']}: 날짜 = {result.get('date', '없음')}, 지역 = {result.get('district', '없음')}")

    # ✅ 모든 지역 및 날짜 수집
    all_districts = [r['district'] for r in results if 'district' in r and r['district']]
    all_dates = [r['date'] for r in results if 'date' in r and r['date']]

    district_counter = Counter(all_districts)
    date_counter = Counter(all_dates)

    # ✅ 지역 최빈값 결정
    if len(set(all_districts)) == 1 and all_districts[0] in ["비서울", "지역 추출 실패"]:
        most_common_district = all_districts[0]
        district_freq = len(all_districts)
    else:
        filtered_districts = [d for d in all_districts if d not in ["비서울", "지역 추출 실패"]]
        if filtered_districts:
            most_common_district, district_freq = Counter(filtered_districts).most_common(1)[0]
        else:
            most_common_district, district_freq = ("지역 추출 실패", 0)

    # ✅ 날짜 최빈값 결정
    if len(set(all_dates)) == 1 and all_dates[0] == "날짜 추출 안됨":
        most_common_date = "날짜 추출 안됨"
        date_freq = len(all_dates)
    else:
        filtered_dates = [d for d in all_dates if d != "날짜 추출 안됨"]
        if filtered_dates:
            most_common_date, date_freq = Counter(filtered_dates).most_common(1)[0]
        else:
            most_common_date, date_freq = ("날짜 추출 안됨", 0)

    # ✅ 결과 저장 폴더 생성
    os.makedirs(result_output_folder, exist_ok=True)

    # ✅ 요약 텍스트 저장
    summary_path = os.path.join(result_output_folder, "analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("📄 OCR 기반 분석 요약\n")
        if most_common_district not in ["비서울", "지역 추출 실패"] or (len(set(all_districts)) == 1):
            f.write(f"• 최빈 지역: {most_common_district} ({district_freq}회)\n")
        if most_common_date != "날짜 추출 안됨" or (len(set(all_dates)) == 1):
            f.write(f"• 최빈 날짜: {most_common_date} ({date_freq}회)\n")

    print(f"✅ OCR 분석 요약 저장 완료: {summary_path}")

    # ✅ 모든 이미지가 비서울이면 종료
    if all(d == "비서울" for d in all_districts):
        print("⚠️ 모든 이미지가 비서울입니다. 서울 지역 영상만 분석 가능합니다.")
        try:
            shutil.rmtree(frame_output_folder)
            print(f"🧹 프레임 폴더 삭제 완료: {frame_output_folder}")
        except Exception as e:
            print(f"⚠️ 프레임 폴더 삭제 중 오류 발생: {e}")
        return {
            'success': False,
            'district': None,
            'date': None
        }

    if most_common_district == "지역 추출 실패":
        print("⚠️ 지역 정보가 없습니다. 수동으로 입력해 주세요.")
    if most_common_date == "날짜 추출 안됨":
        print("⚠️ 날짜 정보가 없습니다. 수동으로 입력해 주세요.")

    return {
    'success': True,
    'district': most_common_district,
    'date': most_common_date
    }

if __name__ == "__main__":
    folder = input("📂 분석할 이미지 폴더 경로 입력: ").strip()
    if not os.path.isdir(folder):
        print("❌ 유효한 폴더가 아닙니다.")
    else:
        print(f"📑 {folder} 내 이미지 분석 시작...\n")
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder, filename)
                result = analyze_image_from_path(full_path)
                print(result)
