import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

class YOLOVideoProcessor:
    def __init__(self, model_path, margin_ratio=0.2, confidence_threshold=0.3):
        self.model = YOLO(model_path, task='detect')
        self.margin_ratio = margin_ratio
        self.confidence_threshold = confidence_threshold

    def _quatered_margin_percent(self, img):
        h, w, _ = img.shape    
        hx, hy = w // 2, h // 2
        
        top_left = img[0:hy+int(hy*self.margin_ratio), 0:hx+int(hx*self.margin_ratio)]
        top_right = img[0:hy+int(hy*self.margin_ratio), hx-int(hx*self.margin_ratio):w]
        bottom_left = img[hy-int(hy*self.margin_ratio):h, 0:hx+int(hx*self.margin_ratio)]
        bottom_right = img[hy-int(hy*self.margin_ratio):h, hx-int(hx*self.margin_ratio):w]
        
        return [top_left, top_right, bottom_left, bottom_right]
    
    def _non_max_suppression(self, boxes):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(area)[::-1]
        selected_indices = []
        
        while len(indices) > 0:
            current_index = indices[0]
            selected_indices.append(current_index)
            xx1 = np.maximum(x1[current_index], x1[indices[1:]])
            yy1 = np.maximum(y1[current_index], y1[indices[1:]])
            xx2 = np.minimum(x2[current_index], x2[indices[1:]])
            yy2 = np.minimum(y2[current_index], y2[indices[1:]])
            w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
            overlap_area = w * h
            iou = overlap_area / (area[current_index] + area[indices[1:]] - overlap_area)
            indices = indices[np.where(0 == iou)[0] + 1]
        return selected_indices
    
    def inference(self, frame, timer):
        img_info = {"id": 0}
        height, width = frame.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = frame
        img_info["ratio"] = 0  # TODO

        results = self.model.predict(self._quatered_margin_percent(frame), half=True)

        annotated_frame = frame.copy()
        fh, fw, _ = frame.shape
        hfh, hfw = fh // 2, fw // 2
        w_margin, h_margin = int(hfw * self.margin_ratio), int(hfh * self.margin_ratio)

        NMS_box_list, NMS_conf_list, NMS_cls_name = [], [], []
        output_list = []  # 최종 출력 리스트

        for i, result in enumerate(results):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf[0].cpu().item()  # float로 변환

                if confidence < self.confidence_threshold:
                    continue

                # 좌표 보정
                if i == 1:
                    x1, x2 = x1 + hfw - w_margin, x2 + hfw - w_margin
                elif i == 2:
                    y1, y2 = y1 + hfh - h_margin, y2 + hfh - h_margin
                elif i == 3:
                    x1, x2 = x1 + hfw - w_margin, x2 + hfw - w_margin
                    y1, y2 = y1 + hfh - h_margin, y2 + hfh - h_margin

                cls_id = int(box.cls[0].cpu().numpy())  # 클래스 ID
                NMS_box_list.append([x1, y1, x2, y2])
                NMS_conf_list.append(confidence)
                NMS_cls_name.append(self.model.names[cls_id])

                # 각 결과를 [x1, y1, x2, y2, confidence, 1, class_id] 형태로 저장
                output_tensor = torch.tensor([x1, y1, x2+50, y2+50, confidence, 1, cls_id], dtype=torch.float32)
                output_list.append(output_tensor)

        final_output = []
        if NMS_box_list:
            selected_idx = self._non_max_suppression(np.array(NMS_box_list))
            for idx in selected_idx:
                final_output.append(output_list[idx])  # NMS로 선택된 결과 추가

        # 최종 결과를 N x 7 형태의 하나의 텐서로 변환
        if final_output:
            final_output = torch.stack(final_output)  # 리스트를 [N, 7] 형태의 텐서로 변환
        else:
            final_output = torch.empty((0, 7), dtype=torch.float32)  # 빈 텐서 반환

        return [final_output], img_info


