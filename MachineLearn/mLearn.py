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
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ 전처리 Dataset - 저장용 transform만 적용
class CustomDataset(Dataset):
    def __init__(self, json_path, transform=None, root_dir="", exclude_corrupted_paths=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.root_dir = root_dir
        self.transform = transform
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = data["annotations"]
        
        # ✅ CustomDataset에서 binary classification용 라벨 생성
        #self.binary_class_map = {"OUT": 0}  # 외부 → 0, 나머지 → 1

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
            if not os.path.isabs(image_path) and self.root_dir:
                image_path = os.path.join(self.root_dir, image_path)

            if exclude_corrupted_paths and image_path in exclude_corrupted_paths:
                continue

            if "attributes" in ann and "Class_code" in ann["attributes"]:
                label_code = ann["attributes"]["Class_code"]
                label_key = self.class_map[label_code]
                self.items.append((image_path, label_key))
                
            #if "attributes" in ann and "Class_code" in ann["attributes"]:
            #    label_code = ann["attributes"]["Class_code"]
            #    binary_label = self.binary_class_map.get(label_code, 1)  # OUT이면 0, 그 외는 1
            #    self.items.append((image_path, binary_label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.items))

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def check_images(image_paths, output_txt="corrupted_images.txt"):
        corrupted = []
        for path in image_paths:
            if not os.path.exists(path):
                corrupted.append(path)
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
            except:
                corrupted.append(path)

        with open(output_txt, "w", encoding="utf-8") as f:
            for path in corrupted:
                f.write(path + "\n")

        return corrupted

# ✅ 학습 시점에서 transform 적용하는 Dataset
class PreprocessedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(os.listdir(folder_path))
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #data = torch.load(os.path.join(self.folder_path, self.files[idx]))  # ← dict 반환
        data = torch.load(os.path.join(self.folder_path, self.files[idx]), weights_only=False)
        img_tensor = data["img"]
        label = data["label"]

        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float() / 255.0
            
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

def main():
    weights = ResNeXt50_32X4D_Weights.DEFAULT

    # transform 저장용 (이미지 크기만 조정)
    base_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # 매 epoch마다 적용될 학습용 transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),          # 다양한 스케일 학습
        transforms.RandomHorizontalFlip(),                             # 좌우 반전
        transforms.RandomRotation(30),                                 # 회전
        transforms.ColorJitter(brightness=0.3, contrast=0.3,           # 밝기, 대비 등 조절
                            saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),                             # 일부 흑백 처리
        transforms.GaussianBlur(kernel_size=3),                        # 블러 처리
        transforms.ToTensor(),                                         # 텐서 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406],               # 정규화
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(torch.cuda.is_available())  # True 나와야 GPU 사용 가능

    corrupted_txt = "corrupted_images.txt"
    if os.path.exists(corrupted_txt):
        with open(corrupted_txt, "r", encoding="utf-8") as f:
            corrupted_paths = [line.strip() for line in f.readlines()]
    else:
        temp_dataset = CustomDataset("학습데이터/merged_annotations.json", transform=None)
        corrupted_paths = CustomDataset.check_images([img_path for img_path, _ in temp_dataset.items], corrupted_txt)

    dataset = CustomDataset("학습데이터/merged_annotations.json", transform=base_transform, exclude_corrupted_paths=corrupted_paths)

    preprocessed_dir = "preprocessed2"
    if not os.path.exists(preprocessed_dir) or len(os.listdir(preprocessed_dir)) == 0:
        print(f"proprocessed된 폴더 없음 생성 시작")
        os.makedirs(preprocessed_dir, exist_ok=True)
        for i, (img, label) in enumerate(dataset):
            # transform 적용된 tensor → PIL 이미지로 되돌리기 (만약 transform이 있었다면)
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img)
            img_np = np.array(img)  # numpy array 형태로 저장
            torch.save({"img": img_np, "label": label}, os.path.join(preprocessed_dir, f"img_{i}.pt"))
        print(f"✅ 전처리 완료: {i + 1}개 저장됨")

    # 파일명 정렬 (순서를 고정시켜야 재현 가능성 ↑)
    files = sorted(os.listdir(preprocessed_dir))

    # 병렬로 라벨만 로드
    def load_label(filename):
        path = os.path.join(preprocessed_dir, filename)
        #return torch.load(path)["label"]
        return torch.load(path, weights_only=False)["label"]

    # ✅ 병렬 처리 (최대 8개 쓰레드 사용)
    with ThreadPoolExecutor(max_workers=8) as executor:
        labels = list(executor.map(load_label, files))

    num_classes = max(labels) + 1
    #num_classes = 2
    dataset = PreprocessedDataset(preprocessed_dir, transform=None)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
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
        
    torch.save(model.state_dict(), "resnext_model.pth5")
    print("✅ 모델 저장 완료: resnext_model.pth5")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
