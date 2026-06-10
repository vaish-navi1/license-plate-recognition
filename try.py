import os
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  
model.train(data='dataset.yaml', epochs=10, imgsz=416, batch=16, cache=True)
results = model.val()
print(results)
model.predict(source=r"F:\license plate\ultralytics\dataset\test\images\8edca6de-d964-4027-823d-f70d3f872412___12042943_1003224833031653_9073848483429898813_n-jpg_jpeg.rf.1882821d2dd595aee5b484591273cab7.jpg", conf=0.25, save=True)