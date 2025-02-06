import os
import cv2

# 이미지가 저장된 폴더 경로
input_path = "/home/hunature/Desktop/PlateTracker/result/crop_img"

# 지원하는 이미지 확장자
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# 폴더 내 모든 파일 순회
for filename in os.listdir(input_path):
    if filename.lower().endswith(image_extensions) and "id3" in filename:
        file_path = os.path.join(input_path, filename)

        # 이미지 읽기
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ 이미지를 불러올 수 없습니다: {filename}")
            continue

        # 이미지의 높이와 너비 확인
        height, width = img.shape[:2]

        # 결과 출력
        print(f"파일명: {filename}")
        print(f"넓이(Width): {width}, 높이(Height): {height}")
        print("-" * 50)
