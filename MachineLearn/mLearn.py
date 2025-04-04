import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ✅ 1. 데이터셋 클래스 정의 (원천데이터 폴더 경로 반영)
class CustomDataset(Dataset):
    def __init__(self, json_path, img_root_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.img_root_dir = img_root_dir
        self.transform = transform
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = {ann["image_id"]: ann for ann in data["annotations"]}

        # 카테고리 ID를 라벨로 매핑
        self.category_map = {ann["category_id"] for ann in data["annotations"]}
        self.category_map = {cat_id: i for i, cat_id in enumerate(sorted(self.category_map))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx + 1]
        ann_info = self.annotations.get(img_info["id"], {})

        # 이미지 경로 구성 (원천데이터 폴더 구조 반영)
        img_path = img_info["file_path"]
        image = Image.open(img_path).convert("RGB")

        # 라벨 가져오기
        category_id = ann_info.get("category_id", 0)
        label = self.category_map.get(category_id, 0)  # 없는 경우 기본값 0

        # 바운딩 박스 (필요하면 사용)
        bbox = ann_info.get("bbox", [0, 0, 0, 0])

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(bbox, dtype=torch.float32)

# ✅ 2. 데이터 변환 (ResNeXt 입력 크기 맞추기)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNeXt 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ 3. 데이터셋 및 데이터로더 생성
dataset = CustomDataset("merged_annotations.json", "원천데이터", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ 4. ResNeXt 모델 불러오기 및 수정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 사용 중인 디바이스: {device}")

model = models.resnext50_32x4d(pretrained=True)  # ResNeXt50-32x4d 사용
num_classes = 10  # 클래스 개수 (확인 필요)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)  # 마지막 레이어 수정
model = model.to(device)

# ✅ 5. 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 6. 학습 루프
num_epochs = 10  # 학습 에포크 수

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, _ in dataloader:  # 바운딩 박스는 사용 안 함
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
