import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    print("CUDA 버전:", torch.version.cuda)
else:
    print("❌ CUDA 사용 불가. CPU만 사용 중입니다.")

# ✅ 1. Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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
            # 해당 annotation의 이미지 정보 가져오기
            image_info = self.images[ann["image_id"]]
            image_path = image_info["file_path"]

            # 해당 annotation에서 'Class_code' 추출
            if "attributes" in ann and "Class_code" in ann["attributes"]:
                label_code = ann["attributes"]["Class_code"]

                # 추출된 'Class_code'로 라벨 매핑
                label_key = self.class_map[label_code]
                self.items.append((image_path, label_key))

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

# 학습용 transform (데이터 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomResizedCrop(224),         # 무작위 자르기 (데이터 증강)
    transforms.RandomHorizontalFlip(),         # 수평 뒤집기
    transforms.RandomRotation(30),             # 회전
    transforms.ToTensor(),                     # PIL → Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 검증용 transform (기본 전처리만)
val_transform = weights.transforms()

# 데이터셋 정의
dataset = CustomDataset("merged_annotations.json", transform=None)  # 전체 데이터셋
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# transform 주입
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# ✅ pin_memory=True 로 GPU 전송 최적화
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# ✅ 4. 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 사용 중인 디바이스: {device}")

# ✅ 성능 최적화
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

model = resnext50_32x4d(weights=weights)
num_classes = len(dataset.class_map)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ✅ 5. 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 6. 학습 루프
# 혼합 정밀도
scaler = GradScaler()

# 학습률 스케줄러
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 데이터 증강
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 혼합 정밀도
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    scheduler.step()  # 학습률 스케줄링
    print(f"[{epoch+1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")
# ✅ 7. 평가 (Top-1 Accuracy, Weighted F1 Score, 클래스별 F1 스코어)
from sklearn.metrics import classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ 전체 F1, Accuracy
f1_weighted = f1_score(all_labels, all_preds, average="weighted")
acc = accuracy_score(all_labels, all_preds)

# ✅ 클래스별 F1 Score
class_f1 = f1_score(all_labels, all_preds, average=None)
report = classification_report(all_labels, all_preds, target_names=list(dataset.class_map.keys()))

print(f"\n✅ 평가 결과 (Validation Set)")
print(f"Top-1 Accuracy: {acc:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")
print(f"\n📊 클래스별 F1 Score:\n{report}")

# ✅ 8. 모델 저장
torch.save(model.state_dict(), "resnext_model.pth")
print("✅ 모델 저장 완료: resnext_model.pth")
