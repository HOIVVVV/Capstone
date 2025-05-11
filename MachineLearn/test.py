import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnext50_32x4d
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ✅ 평가용 데이터셋 정의
class CustomLabeledDataset(Dataset):
    def __init__(self, root_dir, folder_to_label_map, transform=None):
        self.transform = transform
        self.samples = []

        for folder_name, label_idx in folder_to_label_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder_path): continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder_path, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ✅ 메인 평가 함수
def test_with_classmap(data_root, model_path="resnext_model.pth", class_map_path="class_map.json", batch_size=32):
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)  # e.g., {"cat": 0, "dog": 1, ...}

    # ✅ 여기서 수동 매핑: 평가 폴더명 → 학습 클래스명
    # 이건 자동으로 유사하게 매핑하거나 수동으로 명확히 지정 필요
        folder_to_label_map = {
        "1-1-1.균열-길이(Crack-Longitudinal,CL)": 1,
        "1-1-2.균열-원주(Crack-Circumferential,CC)": 0,
        "1-2.표면손상(Surface-Damage,SD)": 2,
        "1-3.파손(Broken-Pipe,BK)": 3,
        "1-4.연결관-돌출(Lateral-Protruding,LP)": 4,
        "1-5.이음부-손상(Joint-Faulty,JF)": 5,
        "2-1.이음부(Pipe-Joint,PJ)_1": 9,
        "2-2.하수관로_내부(Inside,IN)_1": 10
    }

    index_to_class = {v: k for k, v in class_map.items()}

    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomLabeledDataset(data_root, folder_to_label_map, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_map))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("🎯 Accuracy:", accuracy_score(all_labels, all_preds))
    print("🎯 Weighted F1:", f1_score(all_labels, all_preds, average="weighted"))
    print("📊 Classification Report:")
    used_class_indices = sorted(set(all_labels + all_preds))  # 실제 등장한 클래스 인덱스만 추출
    print(classification_report(
        all_labels,
        all_preds,
        labels=used_class_indices,
        target_names=[index_to_class[i] for i in used_class_indices]
    ))

# ✅ 실행 예시
if __name__ == "__main__":
    test_with_classmap("원천데이터", model_path="resnext_model_2.pth", class_map_path="class_map.json")
