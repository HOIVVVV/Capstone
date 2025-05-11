import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from Grad_cam import GradCAM

# ✅ 클래스 맵 로딩 (index → 라벨명)
def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {int(v): k for k, v in raw_map.items()}  # index -> label

# ✅ 모델 및 클래스 맵 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

class_map = load_class_map("class_map.json")  # 14개 클래스
num_classes = max(class_map.keys()) + 1
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("resnext_model_2.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_images_in_folder(folder_path, save_base_path):
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            video_title = os.path.basename(folder_path).split('_')[-1]
            save_path = os.path.join(save_base_path, video_title)
            os.makedirs(save_path, exist_ok=True)
            predict_image(image_path, save_path, video_title, idx + 1)

def predict_image(image_path, save_path, video_title, frame_number):
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top3_probs, top3_preds = torch.topk(probs, 3)

    top3_indices = top3_preds[0].tolist()
    top3_labels = [class_map.get(idx, f"Unknown({idx})") for idx in top3_indices]
    top3_probs_vals = top3_probs[0].tolist()

    # ✅ 손상 여부 판단
    non_damage_labels = [
        "2-1.이음부(Pipe-Joint,PJ)",
        "2-2.하수관로_내부(Inside,IN)",
        "2-3-1.하수관로_외부_맨홀",
        "2-3-2.하수관로_외부_인버트",
        "2-3-3.하수관로_외부_자동차"
    ]

    if top3_probs_vals[0] >= 0.75:
        damage_detected = top3_labels[0] not in non_damage_labels
    else:
        damage_detected = any(label not in non_damage_labels for label in top3_labels)

    print(f"\n📄 [{os.path.basename(image_path)}] 분석 결과:")
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs_vals), 1):
        print(f"  {i}. {label} ({prob*100:.2f}%)")
    print("🔍 손상 감지:", "✅ 예" if damage_detected else "❌ 아니오")

    if not damage_detected:
        os.remove(image_path)
        return []

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

    frame_tag = f"f{str(frame_number).zfill(3)}"
    if top3_probs_vals[0] >= 0.9:
        label_string = f"{top3_labels[0]}({int(top3_probs_vals[0]*100)})"
    else:
        label_string = ",".join(f"{l}({int(p*100)})" for l, p in zip(top3_labels, top3_probs_vals))

    base_filename = f"{video_title}_{frame_tag}_{label_string}"
    # ✅ 저장
    image_save_path = os.path.join(save_path, f"{base_filename}.jpg")
    gradcam_save_path = os.path.join(save_path, f"{base_filename}_GradCAM.jpg")

    original_image.save(image_save_path)
    cv2.imwrite(gradcam_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"🖼️ 이미지 저장: {image_save_path}")
    print(f"📸 Grad-CAM 저장: {gradcam_save_path}")

    return top3_labels

if __name__ == "__main__":
    folder_path = input("📁 예측할 이미지 폴더 경로: ").strip()
    save_path = input("💾 결과 저장 폴더 경로: ").strip()
    if os.path.isdir(folder_path):
        predict_images_in_folder(folder_path, save_path)
    else:
        print("❌ 유효하지 않은 폴더 경로입니다.")
