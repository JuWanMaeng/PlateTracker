import os
import cv2

# 이미지가 저장된 폴더 경로
path = "/home/hunature/Desktop/PlateTracker/result/crop_img_frame0"

# 지원하는 이미지 확장자
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')



# 폴더 내 모든 파일 순회
for filename in os.listdir(path):
    if filename.lower().endswith(image_extensions):
        file_path = os.path.join(path, filename)

        # 이미지 읽기
        img = cv2.imread(file_path)
        print(img.shape)
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        print(img.shape)
        if img is None:
            print(f"❌ 이미지를 불러올 수 없습니다: {filename}")
            continue

        cv2.imwrite(f"result_{filename}", img)