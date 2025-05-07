import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 레이어에서의 활성화 함수 hook 등록
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # 손실을 기준으로 backward 진행
        self.model.zero_grad()
        class_idx = torch.argmax(output)
        output[0, class_idx].backward(retain_graph=True)

        # Grad-CAM 계산
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))  # 각 채널별 중요도 계산
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, weight in enumerate(weights):
            cam += weight * activations[i, :, :]

        # CAM 결과를 ReLU 적용하여 양수 영역만 강조
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))  # 이미지 크기 조정

        # CAM 정규화
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def overlay_cam(self, image, cam):
        # 이미지에 CAM 오버레이
        image = np.array(image) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        overlay = 0.5 * image + 0.5 * heatmap  # CAM 오버레이
        return overlay
    
    def highlight_cam_on_image(self, image, cam):
        # 이미지를 PIL에서 OpenCV 형식으로 변환
        image = np.array(image)  # PIL 이미지를 NumPy 배열로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 사용

        # Grad-CAM 결과를 히트맵으로 변환
        cam = np.maximum(cam, 0)  # CAM 값이 음수일 경우 0으로 클리핑
        cam = cam / np.max(cam)  # CAM 값을 0과 1 사이로 정규화
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # 히트맵 적용
        
        # 히트맵 크기 조정
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 원본 이미지 크기와 일치하도록 조정

        # 이미지와 히트맵을 같은 데이터 타입으로 변환 (np.uint8)
        image = np.uint8(image)
        heatmap = np.uint8(heatmap)

        # 히트맵을 원본 이미지에 오버레이
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)  # 이미지와 히트맵을 결합 (0.7:이미지, 0.3:히트맵)

        # CAM의 가장 강한 부분을 강조하기 위한 네모 박스 그리기
        threshold = 0.5  # 하이라이트 강도 임계값
        cam_thresholded = (cam > threshold).astype(np.uint8)  # 강한 부분만 추출

        # CAM 결과에서 좌표 찾기
        y_indices, x_indices = np.where(cam_thresholded == 1)

        if len(x_indices) > 0 and len(y_indices) > 0:
            # CAM 영역을 기준으로 네모 박스 그리기
            top_left = (min(x_indices), min(y_indices))
            bottom_right = (max(x_indices), max(y_indices))

            # 초록색 네모박스 (BGR 색상)
            cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)

        return overlay
    
    #def overlay_cam(self, image, cam):
        # 이미지를 [0, 1] 범위로 정규화
        image = np.array(image) / 255.0

        # CAM 값 임계값 설정 (CAM이 강조하는 부분을 추출하기 위한 임계값)
        threshold = 0.5  # CAM 값이 이보다 클 때 해당 영역을 강조

        # CAM 값이 임계값 이상인 부분을 찾음
        cam_mask = cam > threshold
        indices = np.argwhere(cam_mask)  # CAM 마스크가 True인 좌표들

        # 이미지에 초록색 네모 그리기 (원본 이미지 위에 그리기)
        image_with_rectangles = image.copy()  # 원본 이미지의 복사본을 생성
        
        for idx in indices:
            y, x = idx  # y, x 좌표
            # 초록색 네모 그리기 (좌표 기준으로 약간의 여백을 주기 위해 작은 크기로 설정)
            cv2.rectangle(image_with_rectangles, (x-10, y-10), (x+10, y+10), (0, 1, 0), 2)  # 초록색 (0, 1, 0)

        # CAM 색상 맵 생성
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0

        # CAM 오버레이 (원본 이미지에 초록색 네모를 그린 후 CAM 히트맵을 덧붙임)
        overlay = 0.5 * image_with_rectangles + 0.5 * heatmap  # CAM 오버레이

        # 0-255 범위로 정규화된 이미지로 변환하여 반환
        return (overlay * 255).astype(np.uint8)
