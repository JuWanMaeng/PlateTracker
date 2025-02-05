#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import os
import re # ocr 과정에서 중복된 공백 요소 제거
import numpy as np
import pytesseract

__all__ = ["vis"]

# object crop 시 사용
#save_dir = "/home/hunature/Desktop/PlateTracker/result/crop_img" 

def convert_origin_dots(x1, y1, w, h, bbox_margin_w, bbox_margin_h):

    orig_w = max(w // (1 + 2 * bbox_margin_w), 0)
    orig_h = max(h // (1 + 2 * bbox_margin_h), 0)

    delta_w = w * (bbox_margin_w / (1 + 2 * bbox_margin_w))
    delta_h = h * (bbox_margin_h / (1 + 2 * bbox_margin_h))

    orig_x1 = max(x1 + delta_w, 0)
    orig_y1 = max(y1 + delta_h, 0)

    return orig_x1, orig_y1, orig_w, orig_h

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, bbox_margin_w, bbox_margin_h, xyxy, tlwhs, obj_ids, args, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 1
    text_thickness = 2
    line_thickness = 3
    ocr_text_scale = 1

    radius = max(5, int(im_w/140.))
    
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        orig_x1, orig_y1, orig_x2, orig_y2 = xyxy[i]
        orig_w = orig_x2 - orig_x1
        orig_h = orig_y2 - orig_y1
        intbox = tuple(map(int, (orig_x1, orig_y1, orig_x1 + orig_w, orig_y1 + orig_h)))

        tlwhbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        # Crop 및 저장
        #crop_img = im[intbox[1]:intbox[3], intbox[0]:intbox[2]]
        #crop_filename = f"{save_dir}/crop_frame{frame_id}_id{obj_ids[i]}.png"
        #cv2.imwrite(crop_filename, crop_img)

        # 면적 기준으로 OCR 적용 -> 다른 아이디어 적용 후 폐기 예정
        if intbox[2] * intbox[3] > 700:

            obj_crop = im[intbox[1]:intbox[3], intbox[0]:intbox[2]] # object cropping

            upscaling_img = cv2.resize(obj_crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4) # image up scaling 

            ocr_result = pytesseract.image_to_string(upscaling_img, lang='kor+eng')

            # 개행 문자를 공백으로, 폼 피드 제거
            preprocess_result = ocr_result.replace('\n', ' ').replace('\x0c', '').strip()

            # 중복된 공백 제거
            final_ocr_result = re.sub(r'\s+', ' ', preprocess_result)

            ocr_text = str(final_ocr_result[:8])

            cv2.putText(im, ocr_text, (intbox[0]+5, intbox[1]), cv2.FONT_HERSHEY_PLAIN, ocr_text_scale, (255, 255, 255),
                        thickness=text_thickness)

        #xyxy_color = (255, 255, 254)
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        #cv2.rectangle(im, tlwhbox[0:2], tlwhbox[2:4], color=xyxy_color, thickness=1)
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    # crop 한 후 fps plot   
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
