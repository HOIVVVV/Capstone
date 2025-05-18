import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.transforms.v2 as transforms_v2  # PyTorch >=2.0
import time

from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision.transforms.functional import to_pil_image
from concurrent.futures import ThreadPoolExecutor
from torch.serialization import add_safe_globals
from collections import Counter
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_cutmix(inputs, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size()[0])
    target_a = targets
    target_b = targets[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

    return inputs, target_a, target_b, lam

def apply_mixup(inputs, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size()[0])
    mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index]
    target_a = targets
    target_b = targets[rand_index]
    return mixed_inputs, target_a, target_b, lam

def mix_criterion(criterion, outputs, target_a, target_b, lam):
    return lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)


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
        path = os.path.join(self.folder_path, self.files[idx])
        data = torch.load(path, weights_only=False)
        img_tensor = data["img"]
        label = data["label"]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label, os.path.basename(path)  # ✅ 파일명도 함께 반환

def get_unique_model_path(base_name="resnext_model", extension=".pth"):
    i = 0
    while True:
        filename = f"{base_name}_{i}{extension}" if i > 0 else f"{base_name}{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

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

def main():
    print("학습 시작")
    weights = ResNeXt50_32X4D_Weights.DEFAULT
    
    preprocess_transform = transforms.Compose([
        transforms.Lambda(apply_clahe),                    # ✅ CLAHE (PIL 전용)
        transforms.Lambda(stretch_histogram),              # ✅ 히스토그램 스트레칭 (PIL 전용)
        transforms.Resize(256),                            # ✅ 충분히 크게 리사이즈 (선택)
        transforms.CenterCrop(224),                        # ✅ 중앙 자르기
        transforms.ToTensor()                              # ✅ Tensor 변환
    ])

    train_transform = transforms_v2.Compose([
    transforms_v2.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms_v2.RandomHorizontalFlip(p=0.5),
    transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms_v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    transforms_v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms_v2.RandomRotation(15),
    transforms_v2.GaussianBlur(kernel_size=3),
    transforms_v2.RandomAdjustSharpness(sharpness_factor=2),
    transforms_v2.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms_v2.Compose([
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ✅ 추가
    ])


    dataset_root = "테스트데이터"#"C:/Users/parkerpark/Downloads/학습 동일 소형 데이터"
    corrupted_txt_path = "corrupted.txt"
    dataset = FolderBasedDataset(dataset_root, corrupted_txt_path=corrupted_txt_path)
    class_map = dataset.class_map

    preprocessed_dir = "preprocessed_from_folder"
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    if len(os.listdir(preprocessed_dir)) >= len(dataset):
        print(f"⏭️ 전처리된 이미지가 이미 {len(os.listdir(preprocessed_dir))}개 존재합니다. 전처리 스킵.")
    else:
        for i, (img_path, label, label_name) in enumerate(dataset.items):
            pt_path = os.path.join(preprocessed_dir, f"img_{i}.pt")
            image = Image.open(img_path).convert("RGB")
            img_tensor = preprocess_transform(image)  # ⬅️ 명시적 transform 적용
            torch.save({"img": img_tensor, "label": label, "class_name": label_name, "path": img_path}, pt_path)


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
    
    # 라벨 기반 Stratified split
    labels = np.array(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # 라벨로부터 클래스 분포 계산
    class_counts = Counter(labels)
    num_classes = max(class_counts.keys()) + 1
    total_samples = sum(class_counts.values())
    
    
    # 📊 클래스별 샘플 수 출력만 (가중치 계산 및 적용 X)
    #print("\n📊 클래스별 샘플 수:")
    #for i in range(num_classes):
    #    print(f"클래스 {i} | 샘플 수: {class_counts[i]}")

    # ❌ 가중치 없이 기본 CrossEntropyLoss 사용
    #criterion = nn.CrossEntropyLoss()

    #클래스별 가중치 계산
    weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    #출력
    print("\n📊 클래스별 샘플 수 및 가중치:")
    for i in range(num_classes):
        print(f"클래스 {i} | 샘플 수: {class_counts[i]} | 가중치: {weights[i]:.4f}")

    #크로스 엔트로피 손실 함수에 가중치 적용
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(init_scale=65536.0)  # 또는 기본값 사용
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 🎯 원하는 방식 설정 (선택)
    use_cutmix = True      # ✅ CutMix 사용
    use_mixup = False      # ✅ MixUp 사용
    mix_alpha = 0.4        # 혼합 강도
    
    for epoch in range(10):
        start_time = time.time()  # ⏱ 시작 시간 기록

        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (images, labels, filenames) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # ✨ CutMix or MixUp 적용
            if use_cutmix:
                images, targets_a, targets_b, lam = apply_cutmix(images, labels, alpha=mix_alpha)
            elif use_mixup:
                images, targets_a, targets_b, lam = apply_mixup(images, labels, alpha=mix_alpha)

            with autocast():
                outputs = model(images)
                if use_cutmix or use_mixup:
                    loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            print(f"🌀 Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | "
                f"PT: {filenames[0]} | Loss: {loss.item():.4f}", end='\r')

        scheduler.step(running_loss)
        epoch_time = time.time() - start_time  # ⏱ 경과 시간 계산
        print(f"\n✅ [Epoch {epoch+1}] 평균 Loss: {running_loss / total_batches:.4f} | "
            f"⏱ 소요 시간: {epoch_time:.2f}초")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
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