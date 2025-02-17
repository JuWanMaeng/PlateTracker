import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

class YOLOVideoProcessor:
    def __init__(self, model_path, margin_ratio=0.2, confidence_threshold=0.3, bbox_margin_w = 0.5, bbox_margin_h = 2):
        self.model = YOLO(model_path, task='detect')
        self.margin_ratio = margin_ratio # 4분할 이미지의 margin ration를 설정합니다.
        self.confidence_threshold = confidence_threshold
        self.bbox_margin_w = bbox_margin_w # tracking 과정에서 bbox 넓이(w)의 margin 값을 설정합니다.
        self.bbox_margin_h = bbox_margin_h # tracking 과정에서 bbox 높이(h)의 margin 값을 설정합니다.

    def _quatered_margin_percent(self, img):
        h, w, _ = img.shape    
        hx, hy = w // 2, h // 2
        
        top_left = img[0:hy+int(hy*self.margin_ratio), 0:hx+int(hx*self.margin_ratio)]
        top_right = img[0:hy+int(hy*self.margin_ratio), hx-int(hx*self.margin_ratio):w]
        bottom_left = img[hy-int(hy*self.margin_ratio):h, 0:hx+int(hx*self.margin_ratio)]
        bottom_right = img[hy-int(hy*self.margin_ratio):h, hx-int(hx*self.margin_ratio):w]
        
        return [top_left, top_right, bottom_left, bottom_right]
    
    # tracking 을 위한 bbox margin 적용 함수
    def _bbox_margin(self, x1, y1, x2, y2):

        w = x2 - x1
        h = y2 - y1

        half_w = w // 2
        half_h = h // 2

        center_x = x1 + half_w
        center_y = y1 + half_h

        new_w = w * (1 + 2 * self.bbox_margin_w)
        new_h = h * (1 + 2 * self.bbox_margin_h)

        bm_x1 = max(center_x - (new_w // 2), 0)
        bm_y1 = max(center_y - (new_h // 2), 0)
        bm_x2 = max(center_x + (new_w // 2), 0)
        bm_y2 = max(center_y + (new_h // 2), 0)

        return bm_x1, bm_y1, bm_x2, bm_y2

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
    
    def inference(self, frame, timer, args):
        img_info = {"id": 0}
        height, width = frame.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = frame
        img_info["ratio"] = 0  # TODO
        img_info["bbox_margin_w"] = self.bbox_margin_w
        img_info["bbox_margin_h"] = self.bbox_margin_h

        timer.tic()
        results = self.model.predict(self._quatered_margin_percent(frame), half=True)

        annotated_frame = frame.copy()
        fh, fw, _ = frame.shape
        hfh, hfw = fh // 2, fw // 2
        w_margin, h_margin = int(hfw * self.margin_ratio), int(hfh * self.margin_ratio)
        NMS_box_list, NMS_conf_list, NMS_cls_name = [], [], [] # NMS 에 사용할 box, conf_score, class_name 을 담을 list 
        output_list = []  # 최종 출력 리스트

        for i, result in enumerate(results):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf[0].cpu().item()  # float로 변환

                if confidence < self.confidence_threshold:
                    continue

                # 좌표 보정 (4분할 이미지에 대한 좌표를 원본 이미지에 대한 좌표로 보정)
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

                bm_x1, bm_y1, bm_x2, bm_y2 = self._bbox_margin(x1, y1, x2, y2) # tracking 을 위한 bbox margin 좌표 변환

                # 각 결과를 [x1, y1, x2, y2, confidence, 1, class_id] 형태로 저장 
                #output_tensor = torch.tensor([x1, y1, x2+args.box_margin, y2+args.box_margin, confidence, 1, cls_id], dtype=torch.float32) # 수정전 - bbox margin constant 값으로 줌

                output_tensor = torch.tensor([bm_x1, bm_y1, bm_x2, bm_y2, confidence, 1, cls_id, x1, y1, x2, y2], dtype=torch.float32) # bbox margin 적용 + (원본 x1 y1 x2 y2) 좌표 추가
                output_list.append(output_tensor)

        final_output = []
        original_plot = []
        if NMS_box_list:
            selected_idx = self._non_max_suppression(np.array(NMS_box_list))
            for idx in selected_idx:
                final_output.append(output_list[idx])  # NMS로 선택된 결과 추가
        
        # 최종 결과를 N x 7 형태의 하나의 텐서로 변환 -> N X 11 형태의 텐서 (원본 x1 y1 x2 y2) 좌표 추가
        if final_output:
            final_output = torch.stack(final_output)  # 리스트를 [N, 7] 형태의 텐서로 변환 -> 리스트를 [N, 11] 형태의 텐서로 변환
        else:
            final_output = torch.empty((0, 11), dtype=torch.float32)  # 빈 텐서 반환
 

        return [final_output], img_info


