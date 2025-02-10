from ultralytics import YOLO

model = YOLO('pretrained/plate_tracker.pt')

model.export(format = 'engine', int8 = True, batch = 4) #odel.export(format = 'engine',int8 = True)
