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
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision.transforms.functional import to_pil_image
from concurrent.futures import ThreadPoolExecutor
from torch.serialization import add_safe_globals
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load fixed class map from JSON
CLASS_MAP_PATH = "class_map.json"
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        FIXED_CLASS_MAP = json.load(f)
else:
    raise FileNotFoundError("class_map.json not found. Please provide a fixed class map.")

# Image verification

def verify_image(path, remove_corrupted=False):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"❌ 손상된 이미지: {path} - {e}")
        if remove_corrupted:
            try:
                os.remove(path)
                print(f"🗑 삭제됨: {path}")
            except Exception as remove_error:
                print(f"삭제 실패: {remove_error}")
        return False


def load_corrupted_images_txt(txt_path):
    if not os.path.exists(txt_path):
        return set()
    with open(txt_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

# Folder label mapping using fixed class_map.json

def generate_folder_label_map(root_dir, corrupted_txt_path=None, remove_corrupted=False):
    corrupted_set = load_corrupted_images_txt(corrupted_txt_path) if corrupted_txt_path else set()
    label_to_index = FIXED_CLASS_MAP
    data = []

    for label_name, label_idx in label_to_index.items():
        folder_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️ 폴더 없음: {folder_path}")
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file)
                if file in corrupted_set:
                    print(f"⚠️ 스킵 (사전 등록 손상): {file}")
                    continue
                if verify_image(image_path, remove_corrupted=remove_corrupted):
                    data.append((image_path, label_idx, label_name))
    return label_to_index, data


class FolderBasedDataset(Dataset):
    def __init__(self, root_dir, transform=None, corrupted_txt_path="corrupted.txt"):
        self.transform = transform
        self.class_map, self.items = generate_folder_label_map(root_dir, corrupted_txt_path, remove_corrupted=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label, _ = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class PreprocessedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(os.listdir(folder_path))
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.folder_path, self.files[idx]), weights_only=False)
        img_array = data["img"]
        if isinstance(img_array, str):
            raise TypeError("img should be a numpy array, not str. Check preprocessing step.")
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        label = data["label"]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label


def get_unique_model_path(base_name="resnext_model", extension=".pth"):
    i = 0
    while True:
        filename = f"{base_name}_{i}{extension}" if i > 0 else f"{base_name}{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1


def main():
    print("학습 시작")
    weights = ResNeXt50_32X4D_Weights.DEFAULT

    base_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_root = "학습데이터/원천데이터"
    corrupted_txt_path = "corrupted.txt"
    dataset = FolderBasedDataset(dataset_root, transform=base_transform, corrupted_txt_path=corrupted_txt_path)
    class_map = dataset.class_map

    preprocessed_dir = "preprocessed_from_folder"
    os.makedirs(preprocessed_dir, exist_ok=True)
    if len(os.listdir(preprocessed_dir)) >= len(dataset):
        print(f"⏭️ 전처리된 이미지가 이미 {len(os.listdir(preprocessed_dir))}개 존재합니다. 전처리 스킵.")
    else:
        for i, (img_path, label, label_name) in enumerate(dataset.items):
            pt_path = os.path.join(preprocessed_dir, f"img_{i}.pt")
            image = Image.open(img_path).convert("RGB").resize((224, 224))  # 🔧 크기 고정
            img_np = np.array(image).astype(np.uint8)  # 🧠 저장 최적화
            torch.save({"img": img_np, "label": label, "class_name": label_name, "path": img_path}, pt_path)

    files = sorted(os.listdir(preprocessed_dir))

    def load_label(filename):
        path = os.path.join(preprocessed_dir, filename)
        return torch.load(path, weights_only=False)["label"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        labels = list(executor.map(load_label, files))

    num_classes = max(labels) + 1

    dataset = PreprocessedDataset(preprocessed_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(init_scale=65536.0)  # 또는 기본값 사용
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()
        print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Weighted F1:", f1_score(all_labels, all_preds, average="weighted"))
    print(classification_report(all_labels, all_preds))

    model_path = get_unique_model_path()
    torch.save(model.state_dict(), model_path)
    print(f"✅ 모델 저장 완료: {model_path}")

    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=4, ensure_ascii=False)
    print("✅ 클래스 맵 저장 완료: class_map.json")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()