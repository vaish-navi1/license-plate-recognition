import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗"
)

st.title("🚗 License Plate Recognition System")
st.write("Upload a vehicle image to detect and read the license plate")

# Load model only once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Load OCR only once
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

model = load_model()
reader = load_reader()

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("🔍 Detecting license plate..."):

        img_array = np.array(image)
        img_bgr = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2BGR
        )

        results = model(img_bgr, conf=0.25)

        plate_found = False

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cropped_plate = img_bgr[y1:y2, x1:x2]

                if cropped_plate.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(
                    cropped_plate,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    crop_rgb,
                    caption="Detected License Plate",
                    width=300
                )

                # OCR directly on cropped image
                ocr_result = reader.readtext(
                    cropped_plate
                )

                for detection in ocr_result:

                    text = detection[1]
                    prob = detection[2]

                    if prob > 0.30:
                        plate_found = True

                        st.success(
                            f"✅ Detected Number Plate: {text}"
                        )

                        st.info(
                            f"Confidence: {prob:.0%}"
                        )

                # Draw bounding box
                cv2.rectangle(
                    img_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img_bgr,
                    "License Plate",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        if not plate_found:
            st.warning(
                "⚠️ No license plate detected. Try a clearer image."
            )

        final_img = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            final_img,
            caption="Output",
            use_container_width=True
        )
