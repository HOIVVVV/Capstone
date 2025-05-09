###GPU 사용 버전###
import os
import torch
import cv2
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import matplotlib.pyplot as plt
from Grad_cam import GradCAM  # Grad-CAM 모듈 임포트
import json

                
def get_class_map_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_codes = sorted({
        ann["attributes"]["Class_code"]
        for ann in data["annotations"]
        if "attributes" in ann and "Class_code" in ann["attributes"]
    })

    class_map = {i: code for i, code in enumerate(class_codes)}  # index → 클래스 코드
    return class_map

# 모델 불러오기 (학습된 가중치 사용)
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

# 마지막 레이어의 출력 크기 수정 (예: 12개 클래스로 fine-tune)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 11)  # 12개의 클래스로 수정

# 학습된 모델 가중치 불러오기
# CUDA 사용 가능 여부에 따라 자동으로 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("resnext_model.pth5", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)

model.eval()  # 평가 모드로 설정

# 클래스 코드 -> 라벨 매핑
try:
    class_map = get_class_map_from_json("학습데이터/merged_annotations.json")
    print("✅ JSON에서 클래스 맵을 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"⚠️ JSON 로딩 실패. 기본 클래스 맵을 사용합니다. 이유: {e}")
    class_map = {
        0: "CL",     # 균열-길이
        1: "CC",  # 균열-원주
        2: "SD",         # 표면손상
        3: "BP",            # 파손
        4: "LP",     # 연결관-돌출
        5: "JF",           # 이음부 손상
        6: "JD",        # 이음부 단차
        7: "DS",         # 토사퇴적
        8: "Etc",                    # 기타결함
        9: "PJ",             # 비손상 - 이음부
        10: "IN",                # 비손상 - 내부
        11: "OUT"                # 비손상 - 외부
    }


# 이미지 전처리 (학습과 동일하게 수정)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_images_in_folder(folder_path, save_base_path):
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)

            video_title = os.path.basename(folder_path).split('_')[-1]
            save_path = os.path.join(save_base_path, video_title)
            os.makedirs(save_path, exist_ok=True)

            frame_number = idx + 1  # 실제 프레임 순서 (1부터 시작)
            predict_image(image_path, save_path, video_title, frame_number)


def predict_image(image_path, save_path, video_title, frame_number):
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size

    preprocess = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(original_image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top3_probs, top3_preds = torch.topk(probabilities, 3)

    top3_indices = top3_preds[0].tolist()
    top3_labels = [class_map.get(idx, f"Unknown({idx})") for idx in top3_indices]
    top3_probs_vals = top3_probs[0].tolist()

    non_damage_labels = ['IN', 'OUT', 'PJ']

    # ✅ 손상 여부 판단
    if top3_probs_vals[0] >= 0.9:
        damage_detected = top3_labels[0] not in non_damage_labels
    else:
        damage_detected = any(label not in non_damage_labels for label in top3_labels)
        
    # ✅ 로그 출력
    print(f"\n📄 [{os.path.basename(image_path)}] 분석 결과:")
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs_vals), 1):
        print(f"  {i}. {label} ({prob*100:.2f}%)")
    print("🔍 손상 감지:", "✅ 예" if damage_detected else "❌ 아니오")

    if not damage_detected:
        os.remove(image_path)
        return []

    # ✅ Grad-CAM 생성
    grad_cam = GradCAM(model, target_layer=model.layer4[2].conv3)
    cam = grad_cam.generate_cam(image_tensor)

    if cam is None or np.max(cam) == 0:
        return top3_labels

    cam_resized = cv2.resize(cam, (original_width, original_height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    original_np = np.array(original_image) / 255.0
    overlay = heatmap + original_np
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(255 * overlay)

    # ✅ 파일명 생성
    frame_tag = f"f{str(frame_number).zfill(3)}"

    if top3_probs_vals[0] >= 0.9:
        label_string = f"{top3_labels[0]}({int(top3_probs_vals[0]*100)})"
    else:
        label_string = ",".join(
            f"{label}({int(prob*100)})"
            for label, prob in zip(top3_labels, top3_probs_vals)
        )

    base_filename = f"{video_title}_{frame_tag}_{label_string}"

    # ✅ 저장
    image_save_path = os.path.join(save_path, f"{base_filename}.jpg")
    gradcam_save_path = os.path.join(save_path, f"{base_filename}_GradCAM.jpg")

    original_image.save(image_save_path)
    cv2.imwrite(gradcam_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return top3_labels


if __name__ == "__main__":
    folder_path = input("예측할 이미지가 있는 폴더 경로를 입력하세요: ")  # 사용자로부터 폴더 경로 입력 받기
    save_path = input("결과 이미지를 저장할 폴더 경로를 입력하세요: ")  # 사용자로부터 저장 경로 입력 받기
    
    if os.path.isdir(folder_path):  # 폴더 경로가 유효한지 확인
        predict_images_in_folder(folder_path, save_path)  # 폴더 내 모든 이미지 예측
    else:
        print("유효하지 않은 폴더 경로입니다.")
        