import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from Grad_cam import GradCAM

# ✅ 클래스 맵 로딩 (class_map.json → index → label 변환)
try:
    with open("class_map.json", "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    class_map = {v: k for k, v in raw_map.items()}  # index → label
except Exception as e:
    print(f"❌ class_map.json 로드 실패: {e}")
    class_map = {}

# ✅ 모델 로드 (CPU 기준, 클래스 수 자동 설정)
weights = ResNeXt50_32X4D_Weights.DEFAULT
num_classes = max(class_map.keys()) + 1 if class_map else 11

model = resnext50_32x4d(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("resnext_model_2.pth", map_location="cpu")

try:
    model.load_state_dict(checkpoint, strict=True)
except RuntimeError as e:
    print("⚠️ strict=True 실패! 일부 weight 미적용 가능성:", e)

print("✅ fc 평균 weight:", model.fc.weight.abs().mean().item())

device = torch.device("cpu")
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# ✅ 단일 이미지 분석
def analyze_image(image_path, save_dir):
    image_name = os.path.basename(image_path)
    original = Image.open(image_path).convert("RGB")
    original_resized = original.resize((232, 232), resample=Image.BICUBIC)
    original_cropped = transforms.CenterCrop(224)(original_resized)

    # ✅ 새로운 전처리 적용
    img_np = np.array(original_cropped).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_preds = torch.topk(probs, 3)

    top_labels = [class_map.get(idx.item(), f"Unknown({idx.item()})") for idx in top_preds[0]]
    top_scores = [prob.item() for prob in top_probs[0]]

    print(f"\n📄 {image_name} 분석 결과:")
    for i, (label, score) in enumerate(zip(top_labels, top_scores), 1):
        print(f"  {i}. {label} ({score*100:.2f}%)")

    # ✅ Grad-CAM 생성
    grad_cam = GradCAM(model, target_layer=model.layer4[-1])
    cam = grad_cam.generate_cam(tensor)
    if cam is None or np.max(cam) == 0 or np.isnan(np.max(cam)):
        print("⚠️ Grad-CAM 생성 실패")
        return

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam_resized = cv2.resize(cam, original_cropped.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = np.array(original_cropped) / 255.0 + heatmap / 255.0
    overlay = np.uint8(255 * (overlay / np.max(overlay)))

    # ✅ Grad-CAM 저장
    filename = os.path.splitext(image_name)[0] + "_GradCAM.jpg"
    gradcam_save_path = os.path.join(save_dir, filename)

    if np.isnan(overlay).any() or np.max(overlay) == 0:
        print("❌ overlay 값이 이상합니다. 저장 생략.")
        return

    success = cv2.imwrite(gradcam_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if success:
        print(f"✅ Grad-CAM 저장 완료: {gradcam_save_path}")
    else:
        print(f"❌ Grad-CAM 저장 실패: {gradcam_save_path}")


# ✅ 폴더 내 모든 이미지 분석
def analyze_folder(image_dir, save_dir):
    valid_exts = (".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    if not files:
        print("❌ 분석할 이미지가 없습니다.")
        return
    os.makedirs(save_dir, exist_ok=True)
    for file in sorted(files):
        analyze_image(os.path.join(image_dir, file), save_dir)
        
def preprocess_input_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((232, 232), resample=Image.BICUBIC)
    image = transforms.CenterCrop(224)(image)

    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    # ✅ Normalize 적용 (학습과 동일)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)

    return img_tensor.unsqueeze(0)  # [1, C, H, W]

# ✅ 실행
if __name__ == "__main__":
    folder_path = input("📁 분석할 이미지 폴더 경로: ").strip()
    save_path = input("💾 Grad-CAM 저장할 폴더 경로: ").strip()
    if os.path.isdir(folder_path):
        analyze_folder(folder_path, save_path)
    else:
        print("❌ 유효하지 않은 폴더 경로입니다.")
