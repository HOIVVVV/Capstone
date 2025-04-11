import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# ✅ 1. Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f) #ㅅㅇㄹㄴㅇㄹ

        self.transform = transform
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = data["annotations"]

        # 클래스 코드 추출 및 라벨 매핑
        self.class_codes = sorted({
            ann["attributes"]["Class_code"]
            for ann in self.annotations
            if "attributes" in ann and "Class_code" in ann["attributes"]
        })
        self.class_map = {code: i for i, code in enumerate(self.class_codes)}

        self.items = []
        for ann in self.annotations:
            image_info = self.images[ann["image_id"]]
            image_path = image_info["file_path"]
            label_code = sorted({
            ann["attributes"]["Class_code"]
            for ann in self.annotations
            if "attributes" in ann and "Class_code" in ann["attributes"]
        })
            label_key = label_code[0] if isinstance(label_code, list) else label_code
            self.items.append((image_path, self.class_map[label_key]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        full_path = img_path
        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ 2. 모델 가중치 로딩 및 transform 정의
weights = ResNeXt50_32X4D_Weights.DEFAULT
transform = weights.transforms()

# ✅ 3. 데이터셋 및 분할
dataset = CustomDataset("merged_annotations.json", transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ 4. 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 사용 중인 디바이스: {device}")

model = resnext50_32x4d(weights=weights)
num_classes = len(dataset.class_map)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ✅ 5. 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 6. 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[{epoch+1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")

# ✅ 7. 평가 (Top-1 Accuracy & F1 Score)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

f1 = f1_score(all_labels, all_preds, average="weighted")
acc = accuracy_score(all_labels, all_preds)

print(f"\n✅ 평가 결과 (Validation Set)")
print(f"Top-1 Accuracy: {acc:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

# ✅ 8. 모델 저장
torch.save(model.state_dict(), "resnext_model.pth")
print("✅ 모델 저장 완료: resnext_model.pth")
