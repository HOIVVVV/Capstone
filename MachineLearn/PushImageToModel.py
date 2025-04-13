import os
import torch
import cv2
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import matplotlib.pyplot as plt
from Grad_cam import GradCAM  # Grad-CAM 모듈 임포트

# 모델 불러오기 (학습된 가중치 사용)
weights = ResNeXt50_32X4D_Weights.DEFAULT
model = resnext50_32x4d(weights=weights)

# 마지막 레이어의 출력 크기 수정 (예: 8개 클래스로 fine-tune)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 8)  # 8개의 클래스로 수정

# 학습된 모델 가중치 불러오기
checkpoint = torch.load("resnext_model.pth")
model.load_state_dict(checkpoint, strict=False)  # strict=False로 하여 미ismatch된 가중치는 무시

model.eval()  # 평가 모드로 설정

# 클래스 코드 -> 라벨 매핑
class_map = {
    0: 'BK',  # 예시, 실제 코드에 맞게 수정
    1: 'CC',
    2: 'CL',
    3: 'IN',
    4: 'JF',
    5: 'LP',
    6: 'PJ',
    7: 'SD'
}

# 이미지 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet 계열 모델의 평균, 표준편차
    ])
    
    image = Image.open(image_path).convert("RGB")  # 이미지를 RGB 형식으로 열기
    image = transform(image)  # 전처리 적용
    return image.unsqueeze(0)  # 배치 차원 추가

def predict_image(image_path, save_path):
    # 이미지 전처리
    image = preprocess_image(image_path)
    
    # Grad-CAM 객체 생성
    grad_cam = GradCAM(model, target_layer=model.layer4[2].conv3)  # 모델의 특정 레이어 선택
    
    # 이미지를 모델에 입력하고 예측 결과 얻기
    with torch.no_grad():
        outputs = model(image)  # 모델에 이미지 입력
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 확률로 변환
        top3_probs, top3_preds = torch.topk(probabilities, 3)  # 상위 3개 예측

    # 예측된 클래스의 라벨과 확률 반환
    top3_results = []
    for i in range(3):
        label = class_map[top3_preds[0][i].item()]
        prob = top3_probs[0][i].item()
        top3_results.append((label, prob))
    
    # Grad-CAM 결과 생성
    cam = grad_cam.generate_cam(image)  # CAM 생성
    overlay = grad_cam.overlay_cam(image.squeeze().permute(1, 2, 0).cpu().numpy(), cam)  # 오버레이된 이미지 생성

    # 저장할 파일 경로 생성
    image_filename = os.path.basename(image_path)
    save_filename = f"{os.path.splitext(image_filename)[0]}_Grad-CAM{os.path.splitext(image_filename)[1]}"
    save_filepath = os.path.join(save_path, save_filename)

    # Grad-CAM 오버레이 이미지 저장
    overlay_filename = f"{os.path.splitext(image_filename)[0]}_GradCAM_Overlay{os.path.splitext(image_filename)[1]}"
    overlay_filepath = os.path.join(save_path, overlay_filename)
    
    plt.imshow(overlay)
    plt.title(f"Class: {class_map[top3_preds[0][0].item()]} - Top-3 predictions")
    plt.axis('off')  # 축 숨기기
    plt.savefig(overlay_filepath, bbox_inches='tight', pad_inches=0.1)  # 파일로 저장
    plt.close()  # 현재 플롯 닫기

     # 초록 박스 강조 이미지 생성
    #highlighted = grad_cam.highlight_cam_on_image(image, cam)

    # 초록 테두리 강조 이미지 저장 (디버깅: 이미지 확인)
    #highlighted_filename = f"{os.path.splitext(image_filename)[0]}_GradCAM_highlighted{os.path.splitext(image_filename)[1]}"
    #highlighted_filepath = os.path.join(save_path, highlighted_filename)

    # 이미지가 [0, 255] 범위로 변환되도록 보정
    #highlighted = np.clip(highlighted, 0, 1)  # 0과 1 사이로 클리핑
    #highlighted = (highlighted * 255).astype(np.uint8)  # 255 범위로 변환

    # OpenCV는 BGR 형식을 사용하므로 이미지 색상 변환
    #highlighted_bgr = cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR)

    # 하이라이트된 이미지를 화면에 표시
    #plt.imshow(highlighted)
    #plt.title("Highlighted Grad-CAM Image")
    #plt.axis('off')  # 축 숨기기
    #plt.show()

    # 파일 경로 확인 후 저장
    #print(f"Attempting to save highlighted image to: {highlighted_filepath}")  # 저장 경로 출력
    #if cv2.imwrite(highlighted_filepath, highlighted_bgr):
    #    print(f"Highlighted image saved at {highlighted_filepath}")  # 성공적인 저장 시 메시지
    #else:
    #    print(f"Failed to save highlighted image at {highlighted_filepath}")  # 실패 시 메시지

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
