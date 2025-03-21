import os
import json
import shutil

# 기존 데이터 폴더들 (각 폴더마다 annotations.json이 있음)
dataset_root = "라벨링데이터"
folders = ["1-1-1", "1-1-2", "1-2", "1-3", "1-4", "1-5", "2-1" , "2-2"]  # 여기에 모든 폴더 이름 추가

# 통합된 JSON 데이터
merged_data = {"images": [], "annotations": []}
image_id_offset = 0
annotation_id_offset = 0
category_set = set()

# 통합할 이미지 저장 경로
image_output_folder = os.path.join(dataset_root, "image_folder")
os.makedirs(image_output_folder, exist_ok=True)

# 각 폴더의 JSON 파일을 하나로 합치기
for folder in folders:
    json_path = os.path.join(dataset_root, folder, "annotations.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 데이터 처리
    for img in data["images"]:
        new_image_id = img["id"] + image_id_offset  # ID 중복 방지
        img["id"] = new_image_id
        merged_data["images"].append(img)
        
        # 이미지 파일 이동 (하나의 폴더로 모으기)
        old_img_path = os.path.join(dataset_root, folder, img["file_name"])
        new_img_path = os.path.join(image_output_folder, img["file_name"])
        shutil.move(old_img_path, new_img_path)

    # 어노테이션 데이터 처리
    for ann in data["annotations"]:
        ann["id"] += annotation_id_offset  # ID 중복 방지
        ann["image_id"] += image_id_offset  # 이미지 ID도 맞춰줘야 함
        merged_data["annotations"].append(ann)
        category_set.add(ann["category_id"])

    # ID 오프셋 업데이트
    image_id_offset = max(img["id"] for img in merged_data["images"]) + 1
    annotation_id_offset = max(ann["id"] for ann in merged_data["annotations"]) + 1

# 통합된 JSON 저장
output_json_path = os.path.join(dataset_root, "annotations.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print("✅ 모든 JSON이 병합되었고, 이미지가 하나의 폴더로 이동되었습니다.")
