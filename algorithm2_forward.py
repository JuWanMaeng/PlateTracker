import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

class YOLOVideoProcessor:
    def __init__(self, model_path, margin_ratio=0.2, confidence_threshold=0.3):
        """
        YOLO를 활용한 비디오 객체 탐지 클래스.
        
        Args:
            model_path (str): YOLO 모델 경로.
            margin_ratio (float): 프레임 분할 시 적용할 마진 비율.
            confidence_threshold (float): 탐지 결과를 필터링할 신뢰도 임계값.
        """
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
    
    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()  
            results = self.model.predict(self._quatered_margin_percent(frame), half=True)
            end_time = time.time()
            annotated_frame = frame.copy()
            
            fh, fw, _ = frame.shape
            hfh, hfw = fh // 2, fw // 2
            w_margin, h_margin = int(hfw * self.margin_ratio), int(hfh * self.margin_ratio)
            
            NMS_box_list, NMS_conf_list, NMS_cls_name = [], [], []
            for i, result in enumerate(results):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    if confidence < self.confidence_threshold:
                        continue
                    
                    if i == 1:
                        x1, x2 = x1 + hfw - w_margin, x2 + hfw - w_margin
                    elif i == 2:
                        y1, y2 = y1 + hfh - h_margin, y2 + hfh - h_margin
                    elif i == 3:
                        x1, x2, y1, y2 = x1 + hfw - w_margin, x2 + hfw - w_margin, y1 + hfh - h_margin, y2 + hfh - h_margin
                    
                    NMS_box_list.append([x1, y1, x2, y2])
                    NMS_conf_list.append(confidence)
                    NMS_cls_name.append(self.model.names[int(box.cls[0].cpu().numpy())])
            
            if NMS_box_list:
                selected_idx = self._non_max_suppression(np.array(NMS_box_list))
                for idx in selected_idx:
                    x1, y1, x2, y2 = NMS_box_list[idx]
                    label, conf = NMS_cls_name[idx], NMS_conf_list[idx]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fps_text = f"FPS: {1 / (end_time - start_time):.2f}"
            cv2.putText(annotated_frame, fps_text, (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(annotated_frame)
        
        cap.release()
        out.release()
    
    def process_all_videos(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for video_file in os.listdir(input_dir):
            if video_file.endswith(".mp4"):
                self.process_video(os.path.join(input_dir, video_file), os.path.join(output_dir, video_file))

if __name__ == "__main__":
    model_path = '/workspace/ByteTrack/algorithm2.pt'
    processor = YOLOVideoProcessor(model_path, margin_ratio=0.2, confidence_threshold=0.3)
    input_dir = "/workspace/ByteTrack/input_video"
    output_dir = "/workspace/ByteTrack/result"
    processor.process_all_videos(input_dir, output_dir)
