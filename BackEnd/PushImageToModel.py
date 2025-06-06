import os
import sys
import json
import torch
import cv2
import re
import numpy as np
import torchvision.transforms.v2 as transforms_v2  # PyTorch >=2.0
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from PIL import Image
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from Grad_cam import GradCAM
from BackEnd import progress

# ✅ 클래스 맵 로딩 (index → 라벨명)
def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {int(v): k for k, v in raw_map.items()}

# ✅ 모델 및 클래스 맵 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

#모델 클래스 로드드
class_map = load_class_map("class_map.json")
num_classes = max(class_map.keys()) + 1
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("resnext_model_final.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

def apply_clahe(pil_img):
    img = np.array(pil_img.convert("L"))  # Grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(img)
    return Image.fromarray(cl_img).convert("RGB")

def stretch_histogram(img):
    img_np = np.asarray(img)
    img_min = img_np.min()
    img_max = img_np.max()
    stretched = ((img_np - img_min) / (img_max - img_min + 1e-5) * 255.0).astype(np.uint8)
    return Image.fromarray(stretched)

# ✅ 학습과 동일한 transform 적용 (ToTensor, Normalize 없음)
transform = transforms_v2.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Lambda(stretch_histogram),
        transforms.Resize(256),                            # ✅ 충분히 크게 리사이즈 (선택)
        transforms.CenterCrop(224),                        # ✅ 중앙 자르기
        transforms.ToTensor(),  # PIL → tensor (C, H, W), [0,1]
        # 학습 시 Normalize 안 했으면 아래 생략
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


def predict_images_in_folder(folder_path, save_base_path, video_title):
    image_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)

    for idx, filename in enumerate(image_files):
        percent = 50 + int((idx + 1) / total * 40)  # 50% ~ 90% 사이
        progress["percent"] = percent
        progress["step"] = f"🧠 이미지 분석 중... ({idx + 1}/{total})"
        progress["current_file"] = filename
        print(progress)

        image_path = os.path.join(folder_path, filename)
        save_path = os.path.join(save_base_path, video_title)  # ✅ 정확한 영상 제목 폴더로 저장
        os.makedirs(save_path, exist_ok=True)

        predict_image(image_path, save_path, video_title, idx + 1)

    progress["percent"] = 100
    progress["step"] = "✅ 이미지 분석 완료"


def predict_image(image_path, save_path, video_title, frame_number):
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size

    # ✅ transform 사용
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top3_probs, top3_preds = torch.topk(probs, 3)

    top3_indices = top3_preds[0].tolist()
    top3_labels = [class_map.get(idx, f"Unknown({idx})") for idx in top3_indices]
    top3_probs_vals = top3_probs[0].tolist()

    non_damage_labels = [
        "2-1.관로_내부(Inside,IN)",
        "2-2.관로_외부",
    ]

    if top3_probs_vals[0] >= 0.5:
        damage_detected = top3_labels[0] not in non_damage_labels
    else:
        damage_detected = any(label not in non_damage_labels for label in top3_labels[:2])

    print(f"\n📄 [{os.path.basename(image_path)}] 분석 결과:")
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs_vals), 1):
        print(f"  {i}. {label} ({prob*100:.2f}%)")
    print("🔍 손상 감지:", "✅ 예" if damage_detected else "❌ 아니오")

    if not damage_detected:
        os.remove(image_path)
        return []

    frame_tag = f"f{str(frame_number).zfill(3)}"
    if top3_probs_vals[0] >= 0.9:
        label_string = f"{top3_labels[0]}({int(top3_probs_vals[0]*100)})"
    else:
        label_string = ",".join(f"{l}({int(p*100)})" for l, p in zip(top3_labels, top3_probs_vals))

    base_filename = f"{frame_tag}_{label_string}"
    image_save_path = os.path.join(save_path, f"{base_filename}.jpg")

    original_image.save(image_save_path)
    print(f"🖼️ 이미지 저장: {image_save_path}")

    return top3_labels


if __name__ == "__main__":
    folder_path = input("📁 예측할 이미지 폴더 경로: ").strip()
    save_path = input("💾 결과 저장 폴더 경로: ").strip()
    if os.path.isdir(folder_path):
        predict_images_in_folder(folder_path, save_path)
    else:
        print("❌ 유효하지 않은 폴더 경로입니다.")
