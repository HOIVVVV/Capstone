import os
import torch
import cv2
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from Grad_cam import GradCAM
import json

# 클래스 맵 로딩
def get_class_map_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_codes = sorted({
        ann["attributes"]["Class_code"]
        for ann in data["annotations"]
        if "attributes" in ann and "Class_code" in ann["attributes"]
    })
    return {i: code for i, code in enumerate(class_codes)}

# 모델 로딩 (CPU 전용)
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, 11)  # 클래스 수 조정

device = torch.device("cpu")
checkpoint = torch.load("resnext_model.pth5", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# 클래스 맵 로드
try:
    class_map = get_class_map_from_json("학습데이터/merged_annotations.json")
except:
    class_map = {
        0: "CL", 1: "CC", 2: "SD", 3: "BP", 4: "LP", 5: "JF",
        6: "JD", 7: "DS", 8: "Etc", 9: "PJ", 10: "IN", 11: "OUT"
    }

# 전처리
transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 예측 함수
def analyze_image(image_path, save_dir):
    image_name = os.path.basename(image_path)
    original = Image.open(image_path).convert("RGB")
    tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_preds = torch.topk(probs, 3)

    top_labels = [class_map.get(idx.item(), f"Unknown({idx.item()})") for idx in top_preds[0]]
    top_scores = [prob.item() for prob in top_probs[0]]

    # 로그 출력
    print(f"\n📄 {image_name} 분석 결과:")
    for i, (label, score) in enumerate(zip(top_labels, top_scores), 1):
        print(f"  {i}. {label} ({score*100:.2f}%)")

    # Grad-CAM 생성
    grad_cam = GradCAM(model, target_layer=model.layer4[-1])  # 더 안전한 선택
    cam = grad_cam.generate_cam(tensor)
    if cam is None or np.max(cam) == 0 or np.isnan(np.max(cam)):
        print("⚠️ Grad-CAM 생성 실패")
        return

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam_resized = cv2.resize(cam, original.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = np.array(original) / 255.0 + heatmap / 255.0
    overlay = np.uint8(255 * (overlay / np.max(overlay)))

        # 저장할 파일 이름
    filename = os.path.splitext(os.path.basename(image_path))[0] + "_GradCAM.jpg"
    gradcam_save_path = os.path.join(save_dir, filename)

    # overlay 체크
    if np.isnan(overlay).any() or np.max(overlay) == 0:
        print("❌ overlay 값이 이상합니다. 저장 생략.")
        return

    # 저장
    success = cv2.imwrite(gradcam_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if success:
        print(f"✅ Grad-CAM 저장 완료: {gradcam_save_path}")
    else:
        print(f"❌ Grad-CAM 저장 실패: {gradcam_save_path}")

# 폴더 내 모든 이미지 분석
def analyze_folder(image_dir, save_dir):
    valid_exts = (".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    if not files:
        print("❌ 분석할 이미지가 없습니다.")
        return
    for file in sorted(files):
        analyze_image(os.path.join(image_dir, file), save_dir)

if __name__ == "__main__":
    folder_path = input("📁 분석할 이미지 폴더 경로: ").strip()
    save_path = input("💾 Grad-CAM 저장할 폴더 경로: ").strip()
    if os.path.isdir(folder_path):
        analyze_folder(folder_path, save_path)
    else:
        print("❌ 유효하지 않은 폴더 경로입니다.")
