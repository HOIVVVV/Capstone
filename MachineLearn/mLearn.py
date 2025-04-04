import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ✅ 1. 데이터셋 클래스 정의 (Class_code로 라벨링)
class CustomDataset(Dataset):
    def __init__(self, json_path, img_root_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.img_root_dir = img_root_dir
        self.transform = transform
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = {ann["image_id"]: ann for ann in data["annotations"]}

        # ✅ 모든 Class_code 수집하여 라벨 매핑
        self.class_codes = sorted({ann["attributes"]["Class_code"] for ann in data["annotations"]})
        self.class_code_to_label = {code: i for i, code in enumerate(self.class_codes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx + 1]
        ann_info = self.annotations.get(img_info["id"], {})

        # ✅ 이미지 경로 사용 (사전 생성된 file_path 사용)
        img_path = img_info["file_path"]
        image = Image.open(img_path).convert("RGB")

        # ✅ 라벨: Class_code 기반
        class_code = ann_info.get("attributes", {}).get("Class_code", "UNKNOWN")
        label = self.class_code_to_label.get(class_code, 0)  # 없으면 0으로 대체

        # 바운딩 박스 (사용하지 않지만 포함)
        bbox = ann_info.get("bbox", [0, 0, 0, 0])

        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(bbox, dtype=torch.float32)

# ✅ 2. 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNeXt 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ 3. 데이터셋 및 데이터로더
dataset = CustomDataset("merged_annotations.json", "원천데이터", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ 4. ResNeXt 모델 불러오기 및 수정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 사용 중인 디바이스: {device}")

model = models.resnext50_32x4d(pretrained=True)
num_classes = len(dataset.class_code_to_label)  # Class_code 개수
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
model = model.to(device)

# ✅ 5. 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 6. 학습 루프
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("🎉 학습 완료!")

# ✅ 7. 모델 저장
torch.save(model.state_dict(), "resnext_model.pth")
print("✅ 모델 저장 완료: resnext_model.pth")
