import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  


source_folder = r"C:\Users\vaish\Downloads\license plate\ultralytics\dataset\test\images"
destination_folder = r"C:\Users\vaish\Downloads\license plate\ultralytics\predictions"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(source_folder, image_file)

    model.predict(source=image_path, conf=0.25, save=True)
    predicted_image_name = os.path.join('runs', 'detect', 'predict', image_file)
    if os.path.exists(predicted_image_name):
        os.rename(predicted_image_name, os.path.join(destination_folder, image_file))

print(f"Predictions saved to {destination_folder}")
