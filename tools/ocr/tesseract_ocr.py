import os
import cv2
import pytesseract

# 이미지가 저장된 폴더 경로
path = "/home/hunature/Desktop/PlateTracker/result/crop_img"

# 지원하는 이미지 확장자
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')



# 폴더 내 모든 파일 순회
for filename in os.listdir(path):
    if filename.lower().endswith(image_extensions):
        file_path = os.path.join(path, filename)

        # 이미지 읽기
        img = cv2.imread(file_path)
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        if img is None:
            print(f"❌ 이미지를 불러올 수 없습니다: {filename}")
            continue

        # 흑백 변환 및 이진화
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(
        # gray, 255,
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        # cv2.THRESH_BINARY,
        # 11, 2
        # )

        # OCR 수행
        text = pytesseract.image_to_string(img, lang='kor+eng')
        # OCR 처리 (한국어 + 영어)
        # text = pytesseract.image_to_string(img)

        # 결과 출력
        print(f"파일명: {filename}")
        print(f"OCR 결과:\n{text}")
        print("-" * 50)
