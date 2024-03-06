#======================================================================================================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import os

#======================================================================================================================
#logo file path
logo_path = '/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/logos/EagleEye_logo.png'

# Initialize the camera and model paths in session state if not already present
if 'init' not in st.session_state:
    st.session_state['init'] = True
    st.session_state['cap'] = None  # Initial state of the camera is not opened
    st.session_state['model_path'] = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/small_yolo_model/ppe_s.pt"  # Adjust the path accordingly
    st.session_state['class_info'] = {
        0: {'class_name': 'Hardhat', 'color': (136, 242, 7), 'include': True, 'positive_class': True},
        1: {'class_name': 'Mask', 'color': (136, 242, 7), 'include': True, 'positive_class': True},
        2: {'class_name': 'NO-Hardhat', 'color': (70, 7, 242), 'include': True, 'positive_class': False},
        3: {'class_name': 'NO-Mask', 'color': (70, 7, 242), 'include': True, 'positive_class': False},
        4: {'class_name': 'NO-Safety Vest', 'color': (70, 7, 242), 'include': True, 'positive_class': False},
        5: {'class_name': 'Person', 'color': (255, 238, 82), 'include': True}, 'positive_class': True,
        6: {'class_name': 'Safety Cone', 'color': (255, 255, 0), 'include': True, 'positive_class': True},
        7: {'class_name': 'Safety Vest', 'color': (136, 242, 7), 'include': True, 'positive_class': True},
        8: {'class_name': 'machinery', 'color': (255, 105, 180), 'include': True, 'positive_class': True},
        9: {'class_name': 'vehicle', 'color': (255, 105, 180), 'include': True, 'positive_class': True}
    }

#======================================================================================================================
#Defining functions

# Define a function to start the camera
def start_camera():
    if st.session_state['cap'] is None:
        st.session_state['cap'] = cv2.VideoCapture(0)  # Start the camera
        st.session_state['camera_started'] = True

# Define a function to stop the camera
def stop_camera():
    if st.session_state['cap'] is not None:
        st.session_state['cap'].release()  # Release the camera
        st.session_state['cap'] = None
        st.session_state['camera_started'] = False
        st.success("Camera stopped.")

# Function to process the frame
def process_frame(frame):
    model = YOLO(st.session_state['model_path'])
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if st.session_state['class_info'][cls]['include'] == True:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                conf = math.ceil(box.conf[0] * 100) / 100
                if conf > 0.3:
                    class_name = st.session_state['class_info'][cls]['class_name']
                    color = st.session_state['class_info'][cls]['color']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cvzone.putTextRect(frame, f'{class_name} {conf}', (x1, max(35, y1)), scale=0.85, thickness=1, colorB=color, colorT=(0, 0, 0), colorR=color, offset=2)
    return frame

#======================================================================================================================

#making negative classes mirror their respective positive class
st.session_state['class_info'][2]['include'] = st.session_state['class_info'][0]['include']
st.session_state['class_info'][3]['include'] = st.session_state['class_info'][1]['include']
st.session_state['class_info'][4]['include'] = st.session_state['class_info'][7]['include']

#======================================================================================================================
# UI elements

#Logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(logo_path, width=200)  # Adjust width as needed

st.header("PPE Compliance Monitoring System")

#Camera feed
frame_placeholder = st.empty()

#Start/Stop camera buttons
start_button = st.button('Start Camera', on_click=start_camera)
stop_button = st.button('Stop Camera', on_click=stop_camera)

#User input for selecting model
model_options = ['Small', 'Medium', 'Large']
selected_model = st.selectbox('Choose YOLOv8 object detection model:', model_options)
if selected_model == 'Large':
    st.session_state['model_path'] = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/large_yolo_model/ppe_l.pt"
elif selected_model == 'Medium':
    st.session_state['model_path'] = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/medium_yolo_model/ppe_m.pt"
else:
    st.session_state['model_path'] = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/small_yolo_model/ppe_s.pt"

#User input for selecting classes to included
for class_id, info in st.session_state['class_info'].items():
    if info['positive_class']:
        class_name = info['class_name']
        st.session_state['class_info'][class_id]['include'] = st.checkbox(f'{class_name}', value=st.session_state['class_info'][class_id]['include'])

#======================================================================================================================
# Main loop for frame processing

if st.session_state.get('camera_started', False):
    while True:
        ret, frame = st.session_state['cap'].read()
        if not ret:
            st.write("Failed to capture image")
            stop_camera()
            break

        # Process the frame with PPE detection model
        processed_frame = process_frame(frame)

        # Display the processed frame in the Streamlit app
        frame_placeholder.image(processed_frame, channels="BGR")

        # Check if the stop button has been pressed
        if stop_button:
            stop_camera()
            break

#======================================================================================================================
