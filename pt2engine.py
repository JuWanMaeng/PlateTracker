# pt 파일을 engine 파일로 변환하는 코드입니다.

from ultralytics import YOLO

model = YOLO('pretrained/plate_tracker.pt')

model.export(format = 'engine', int8 = True, batch = 4) # int8 적용
