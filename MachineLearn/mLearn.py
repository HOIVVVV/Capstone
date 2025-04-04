import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

# ✅ 1. 데이터셋 클래스 정의 (Class_code로 라벨링)
class CustomDataset(Dataset):
    def __init__(self, json_path, img_root_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.img_root_dir = img_root_dir
        self.transform = transform
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = {ann["image_id"]: ann for ann in data["annotations"]}

        # Class_code를 고유한 라벨로 매핑
        self.class_codes = sorted({ann["attributes"]["Class_code"] for ann in data["annotations"]})
        self.code_to_label = {code: idx for idx, code in enumerate(self.class_codes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx + 1]
        ann_info = self.annotations.get(img_info["id"], {})
        class_code = ann_info.get("attributes", {}).get("Class_code", "UNKNOWN")
        label = self.code_to_label.get(class_code, 0)

        img_path = img_info["file_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        bbox = ann_info.get("bbox", [0, 0, 0, 0])
        return image, label, torch.tensor(bbox, dtype=torch.float32)

# ✅ 2. 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ 3. 데이터 로딩 및 9:1 분할
full_dataset = CustomDataset("merged_annotations.json", "원천데이터", transform=transform)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ 4. 모델 로딩 및 수정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 현재 사용 중인 디바이스: {device}")

num_classes = len(full_dataset.class_codes)
model = models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ✅ 5. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 6. 학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# ✅ 7. 평가 (F1 Score + Top-1 Accuracy)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

top1_acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"\n📊 평가 결과:")
print(f"🎯 Top-1 Accuracy: {top1_acc * 100:.2f}%")
print(f"🎯 F1 Score: {f1:.4f}")

# ✅ 8. 모델 저장
torch.save(model.state_dict(), "resnext_model.pth")
print("✅ 모델 저장 완료: resnext_model.pth")
