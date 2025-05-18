import os
import json
import torch
import cv2
import numpy as np
import torchvision.transforms.v2 as transforms_v2  # PyTorch >=2.0
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_tiny
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score
from PIL import Image

# ✅ 이미지 폴더에서 직접 불러오는 평가 Dataset
class ImageDatasetWithLabelMap(Dataset):
    def __init__(self, data_root, class_map_path, transform=None):
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        self.label_map = {k: int(v) for k, v in class_map.items()}

        self.samples = []
        for class_name, label in self.label_map.items():
            folder = os.path.join(data_root, class_name)
            if not os.path.isdir(folder): continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def apply_clahe(pil_img):
    img = np.array(pil_img.convert("L"))  # Grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(img)
    return Image.fromarray(cl_img).convert("RGB")

def stretch_histogram(img):
    img_np = np.asarray(img)
    img_min = img_np.min()
    img_max = img_np.max()
    stretched = ((img_np - img_min) / (img_max - img_min + 1e-5) * 255.0).astype(np.uint8)
    return Image.fromarray(stretched)

# ✅ 평가 함수
def test_with_images(data_root, model_path="convnext_model.pth", class_map_path="class_map.json", batch_size=32):
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    index_to_class = {int(v): k for k, v in class_map.items()}

    # ✅ 학습과 동일한 전처리
    transform = transforms_v2.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Lambda(stretch_histogram),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDatasetWithLabelMap(data_root, class_map_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(class_map))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    total = len(dataset)
    all_preds, all_labels = [], []

    print(f"\n🚀 총 {total}개 이미지 평가 시작...\n")

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

            done = min((i + 1) * batch_size, total)
            percent = (done / total) * 100
            print(f"🔁 진행도: [{done}/{total}]  ({percent:.1f}%)", end="\r")

    print("\n✅ 평가 완료!\n")
    print("🎯 Accuracy:", accuracy_score(all_labels, all_preds))
    print("🎯 Weighted F1:", f1_score(all_labels, all_preds, average="weighted"))
    print("📊 Classification Report:")
    used_indices = sorted(set(all_labels + all_preds))
    target_names = [index_to_class.get(i, f"Unknown({i})") for i in used_indices]
    print(classification_report(
        all_labels, all_preds,
        labels=used_indices,
        target_names=target_names
    ))

# ✅ 실행 예시
if __name__ == "__main__":
    test_with_images(
        data_root="원천데이터",  # ← 클래스별 폴더 구조로 되어 있어야 함
        model_path="resnext_model_5.pth",
        class_map_path="class_map.json"
    )
