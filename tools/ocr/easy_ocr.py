import cv2
import easyocr

path = "/home/hunature/Desktop/PlateTracker/result/crop_img/crop_frame1_id2.png"

reader = easyocr.Reader(['ko'])
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

# ✅ Adaptive Threshold 적용
# thresh = cv2.adaptiveThreshold(
#     image,
#     255,                            # 최대값
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 가우시안 가중치 방식
#     cv2.THRESH_BINARY,              # 이진화 적용
#     11,                             # 블록 크기 (주변 픽셀 기준)
#     2                               # 임계값 상수 (조정 가능)
# )
# 2배 확대 (배율 사용)

# print(thresh.shape)
# cv2.imwrite('thresh.png', thresh)
# OCR 수행
results = reader.readtext(image)
cv2.imwrite('image.png', image)

print(results)


