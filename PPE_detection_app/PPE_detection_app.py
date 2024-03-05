import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

small_model_path = "..models/small_yolo_model/ppe_s.pt"
medium_model_path = "..models/medium_yolo_model/ppe_m.pt"
large_model_path = "..models/large_yolo_model/ppe_l.pt"

# Function to process the frame
def process_frame(frame, model_path):
    model = YOLO(model_path)
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
                  'machinery', 'vehicle']
    # Define colors for each class
    colors = {
        'Hardhat': (136, 242, 7),  # Green 7, 242, 136
        'Mask': (136, 242, 7),  # Green
        'Safety Vest': (136, 242, 7),  # Green
        'NO-Hardhat': (70, 7, 242),  # Red
        'NO-Mask': (70, 7, 242),  # Red
        'NO-Safety Vest': (70, 7, 242),  # Red
        'Person': (255, 238, 82),  # Blue
        'Safety Cone': (255, 255, 0),  # Yellow
        'machinery': (255, 105, 180),  # Pink (Light pink, for example)
        'vehicle': (255, 105, 180)  # Pink
    }
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            cls = int(box.cls[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            if conf > 0.3:
                class_name = classNames[cls]
                color = colors.get(class_name, (255, 255, 255))  # Default to white if class name not in colors dict
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cvzone.putTextRect(frame, f'{class_name} {conf}', (x1, max(35, y1)), scale=0.85, thickness=1, colorB=color, colorT=(0, 0, 0), colorR=color, offset=2)
    return frame

# Streamlit UI
st.title("PPE Detection Live Feed")

# Create a placeholder to update the frame in the app
frame_placeholder = st.empty()

# Open the default webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Loop to capture and display frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with your PPE detection model
    processed_frame = process_frame(frame, model_path)

    # Display the processed frame in the Streamlit app
    frame_placeholder.image(processed_frame, channels="BGR")

    # Break the loop if 'q' is pressed (This won't work in Streamlit, but kept for completeness)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
