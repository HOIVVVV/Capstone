import os
import json
import shutil

# ✅ 경로 설정
dataset_root = ""  # 데이터셋 최상위 폴더
label_root = os.path.join(dataset_root, "라벨링데이터")  # JSON들이 있는 폴더
image_root = os.path.join(dataset_root, "원천데이터")  # 실제 이미지들이 있는 폴더
output_json_path = os.path.join(dataset_root, "merged_annotations.json")  # 병합된 JSON 저장 경로

# ✅ 최종 JSON 데이터 구조
merged_data = {"images": [], "annotations": [], "categories": []}
image_id_offset = 0
annotation_id_offset = 0
category_set = set()

# ✅ 원천데이터의 이미지 파일 목록 가져오기
image_files = set(os.listdir(image_root))  # 원천데이터 폴더에 존재하는 이미지 파일 목록

# ✅ 라벨링 데이터(JSON) 파일 리스트 가져오기
json_files = [f for f in os.listdir(label_root) if f.endswith(".json")]

# ✅ JSON 병합 시작
for json_file in json_files:
    json_path = os.path.join(label_root, json_file)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 📌 이미지 데이터 처리 (원천데이터에 존재하는 이미지만 추가)
    valid_images = []
    image_id_map = {}  # 기존 image_id → 새로운 image_id 매핑
    for img in data["images"]:
        if img["file_name"] in image_files:  # 원천데이터에 해당 이미지가 존재하는 경우만 추가
            new_image_id = image_id_offset + 1
            image_id_map[img["id"]] = new_image_id  # 기존 ID → 새로운 ID 매핑
            img["id"] = new_image_id
            valid_images.append(img)
            image_id_offset += 1  # ID 증가

    # 📌 어노테이션 데이터 처리 (유효한 이미지 ID만 유지)
    valid_annotations = []
    for ann in data["annotations"]:
        if ann["image_id"] in image_id_map:  # 원천데이터에 존재하는 이미지의 어노테이션만 유지
            ann["id"] = annotation_id_offset + 1
            ann["image_id"] = image_id_map[ann["image_id"]]  # 새로운 image_id 적용
            valid_annotations.append(ann)
            annotation_id_offset += 1  # ID 증가
            category_set.add(ann["category_id"])

    # 📌 병합된 데이터에 추가
    merged_data["images"].extend(valid_images)
    merged_data["annotations"].extend(valid_annotations)

# ✅ 카테고리 데이터 추가 (중복 제거 후 정리)
merged_data["categories"] = [{"id": cat_id, "name": f"category_{cat_id}"} for cat_id in sorted(category_set)]

# ✅ 병합된 JSON 저장
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print("✅ JSON 병합 완료! 원천데이터 이미지 기준으로 라벨링되었습니다.")
