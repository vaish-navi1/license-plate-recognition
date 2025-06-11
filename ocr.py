import easyocr
import cv2
reader = easyocr.Reader(['en'])

image_path = "crop.jpg"
image = cv2.imread(image_path)

result = reader.readtext(image_path)

for detection in result:
    text = detection[1]  
    print(f"Detected Number Plate Text: {text}")