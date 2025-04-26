import os
import torch
import cv2
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import matplotlib.pyplot as plt
from Grad_cam import GradCAM  # Grad-CAM 모듈 임포트
import json

                
def get_class_map_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_codes = sorted({
        ann["attributes"]["Class_code"]
        for ann in data["annotations"]
        if "attributes" in ann and "Class_code" in ann["attributes"]
    })

    class_map = {i: code for i, code in enumerate(class_codes)}  # index → 클래스 코드
    return class_map

# 모델 불러오기 (학습된 가중치 사용)
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

# 마지막 레이어의 출력 크기 수정 (예: 12개 클래스로 fine-tune)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # 12개의 클래스로 수정

# 학습된 모델 가중치 불러오기
checkpoint = torch.load("resnext_binary_model.pth")
model.load_state_dict(checkpoint, strict=False)  # strict=False로 하여 미ismatch된 가중치는 무시

model.eval()  # 평가 모드로 설정

# 클래스 코드 -> 라벨 매핑
#class_map = get_class_map_from_json("학습데이터/merged_annotations.json")

# ✅ 클래스 매핑 직접 정의 (이진 분류 전용)
class_map = {0: "OUT", 1: "IN"}

# 이미지 전처리 (학습과 동일하게 수정)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# 이미지 전처리 및 Grad-CAM 적용 함수
def predict_image(image_path, save_path):
    # 원본 이미지 열기
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size

    # 전처리
    preprocess = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(original_image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model.to(device)

    # Grad-CAM 객체 생성 (layer4 or layer3 추천)
    grad_cam = GradCAM(model, target_layer=model.layer4[2].conv3)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top3_probs, top3_preds = torch.topk(probabilities, 1)

    # Grad-CAM 생성
    cam = grad_cam.generate_cam(image_tensor)
    
    if cam is None or np.max(cam) == 0:
        print("❌ CAM 생성 실패 또는 모든 값이 0입니다.")
        return []
    else:
        print("✅ CAM 생성 성공. 최대값:", np.max(cam))

    # ✅ CAM을 원본 이미지 크기로 리사이즈
    cam_resized = cv2.resize(cam, (original_width, original_height))

    # 히트맵 생성 및 오버레이
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    original_np = np.array(original_image) / 255.0

    overlay = heatmap + original_np
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(255 * overlay)

    # 저장
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_filepath = os.path.join(save_path, f"{filename}_GradCAM_Overlay.jpg")
    cv2.imwrite(save_filepath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 예측 결과 반환
    top3_results = [
        (class_map[top3_preds[0][i].item()], top3_probs[0][i].item())
        for i in range(1)
    ]
    return top3_results

def predict_images_in_folder(folder_path, save_path):
    # 폴더 내 모든 이미지 파일을 예측 (png, jpg 파일만)
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 저장할 폴더가 없으면 생성
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 확장자 확인
            image_path = os.path.join(folder_path, filename)  # 이미지 파일 경로
            top3_results = predict_image(image_path, save_path)  # 예측 수행
            print(f"{filename}:")
            for i, (label, prob) in enumerate(top3_results, 1):
                print(f"  {i}. {label} ({prob*100:.2f}%)")  # 상위 3개 결과 출력

if __name__ == "__main__":
    folder_path = input("예측할 이미지가 있는 폴더 경로를 입력하세요: ")  # 사용자로부터 폴더 경로 입력 받기
    save_path = input("결과 이미지를 저장할 폴더 경로를 입력하세요: ")  # 사용자로부터 저장 경로 입력 받기
    
    if os.path.isdir(folder_path):  # 폴더 경로가 유효한지 확인
        predict_images_in_folder(folder_path, save_path)  # 폴더 내 모든 이미지 예측
    else:
        print("유효하지 않은 폴더 경로입니다.")
