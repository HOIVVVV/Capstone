import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnext50_32x4d
from sklearn.metrics import classification_report, accuracy_score, f1_score
from PIL import Image
from torchvision import transforms

# ✅ 안전 등록 (PyTorch 2.6 이상에서 필요)
from torch.serialization import add_safe_globals
import numpy as np
add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.float32])

# ✅ 학습과 동일한 구조의 .pt 파일을 불러오는 평가용 Dataset
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
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # ✅ 이미 정규화된 float32
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


# ✅ 평가 함수
def test_with_preprocessed_pt(pt_folder, model_path="resnext_model.pth", class_map_path="class_map.json", batch_size=32):
    # 클래스 맵 로딩
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    index_to_class = {int(v): k for k, v in class_map.items()}

    # Dataset 준비
    dataset = PreprocessedDataset(pt_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 모델 로딩
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext50_32x4d(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_map))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    total = len(dataset)
    all_preds, all_labels = [], []

    print(f"\n🚀 총 {total}개 pt 파일 평가 시작 (batch_size={batch_size})...\n")

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

def convert_eval_images_to_pt_like_training(data_root, class_map_path, save_dir):
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    label_map = {k: int(v) for k, v in class_map.items()}

    transform = transforms.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Lambda(stretch_histogram),
        transforms.Resize(232),  # 학습용 전처리에 맞춤 (비율 유지 없이 resize)
    ])

    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for class_name, label in label_map.items():
        folder = os.path.join(data_root, class_name)
        if not os.path.isdir(folder):
            print(f"❌ 폴더 없음: {folder}")
            continue

        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = transform(image)

                    # ✅ 학습과 동일하게: numpy float32 (0~1 정규화)
                    img_np = np.array(image).astype(np.float32) / 255.0

                    save_path = os.path.join(save_dir, f"eval_{count}.pt")
                    torch.save({
                        "img": img_np,
                        "label": label,
                        "class_name": class_name,
                        "path": img_path
                    }, save_path)
                    count += 1
                except Exception as e:
                    print(f"❌ 변환 실패: {img_path} - {e}")

    print(f"✅ 총 {count}개 평가용 이미지 → 학습 구조 기반 .pt 변환 완료 → {save_dir}")

# ✅ 실행 예시
if __name__ == "__main__":
    convert_eval_images_to_pt_like_training(
        data_root="테스트데이터",
        class_map_path="class_map.json",
        save_dir="평가용_pt폴더"
    )
    test_with_preprocessed_pt("평가용_pt폴더", model_path="resnext_model.pth", class_map_path="class_map.json")
