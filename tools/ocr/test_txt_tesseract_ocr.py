import os
import cv2
import pytesseract
from tqdm import tqdm

# 이미지가 저장된 폴더 경로
input_path = "/home/hunature/Desktop/PlateTracker/result/crop_img"
# OCR 결과를 저장할 폴더 경로
output_path = "/home/hunature/Desktop/PlateTracker/result/ocr_texts"

# 출력 폴더가 없으면 생성
os.makedirs(output_path, exist_ok=True)

# 지원하는 이미지 확장자
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# 처리할 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]

# 폴더 내 모든 파일 순회 (tqdm 추가)
for filename in tqdm(image_files, desc="OCR 처리 진행 중", unit="파일"):
    file_path = os.path.join(input_path, filename)

    # 이미지 읽기
    img = cv2.imread(file_path)
    if img is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {filename}")
        continue

    # 이미지 크기 확대
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

    # OCR 수행
    text = pytesseract.image_to_string(img, lang='kor+eng')

    # 결과 저장할 파일 경로 설정
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_file_path = os.path.join(output_path, txt_filename)

    # OCR 결과 파일로 저장
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(f"파일명: {filename}\n")
        f.write(f"OCR 결과:\n{text}\n")

    print(f"✅ OCR 결과 저장 완료: {txt_file_path}")
