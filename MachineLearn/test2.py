import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnext50_32x4d
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ✅ PreprocessedDataset 클래스
class PreprocessedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(os.listdir(folder_path))
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.folder_path, self.files[idx]), weights_only=False)
        img_tensor = torch.from_numpy(data["img"]).permute(2, 0, 1).float() / 255.0
        label = data["label"]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label

# ✅ 전처리 정의 (평가 시)
eval_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ 클래스 맵 로드
with open("class_map.json", "r", encoding="utf-8") as f:
    class_map = json.load(f)
index_to_class = {v: k for k, v in class_map.items()}

# ✅ 데이터 로드
data_dir = "preprocessed_from_folder"
dataset = PreprocessedDataset(data_dir, transform=eval_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ✅ 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_map)
model = resnext50_32x4d(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnext_model_2.pth", map_location=device))
model.to(device)
model.eval()

# ✅ 평가
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ 출력
print("\n🎯 Accuracy:", accuracy_score(all_labels, all_preds))
print("🎯 Weighted F1:", f1_score(all_labels, all_preds, average="weighted"))
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[index_to_class[i] for i in sorted(index_to_class)]))