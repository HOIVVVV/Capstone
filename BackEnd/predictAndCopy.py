import os
import torch
import shutil
from PIL import Image
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import torchvision.transforms.v2 as transforms_v2
import json
import sys

# ✅ class_map 불러오기
def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {int(v): k for k, v in raw_map.items()}

# ✅ 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

class_map = load_class_map("class_map_final.json")
num_classes = max(class_map.keys()) + 1
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("resnext_model_final.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# ✅ 전처리 정의
transform = transforms_v2.Compose([
    transforms_v2.Resize(256),
    transforms_v2.CenterCrop(224),
    transforms_v2.ToTensor(),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

from concurrent.futures import ThreadPoolExecutor, as_completed

def classify_and_copy_images(input_folder, output_base):
    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total = len(files)

    def process(idx_filename):
        idx, filename = idx_filename
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top1 = torch.argmax(probs, dim=1).item()

        label = class_map.get(top1, f"Unknown({top1})")
        save_dir = os.path.join(output_base, label)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(image_path, os.path.join(save_dir, filename))

        return (idx, filename, label)

    with ThreadPoolExecutor(max_workers=8) as executor:  # 🔁 스레드 개수 조정 가능
        futures = [executor.submit(process, (idx, filename)) for idx, filename in enumerate(files, start=1)]
        for future in as_completed(futures):
            idx, filename, label = future.result()
            percent = (idx / total) * 100
            sys.stdout.write(f"\r📦 분류 진행: [{idx}/{total}] {filename} → {label}  ({percent:.1f}%)")
            sys.stdout.flush()

    print("\n✅ 모든 이미지 분류 및 복사 완료!")

# ✅ 실행 예시
if __name__ == "__main__":
    input_dir = input("분류할 이미지 폴더 경로: ").strip()
    output_dir = input("결과 저장 폴더 경로: ").strip()
    if os.path.isdir(input_dir):
        classify_and_copy_images(input_dir, output_dir)
    else:
        print("❌ 유효하지 않은 입력 경로입니다.")
