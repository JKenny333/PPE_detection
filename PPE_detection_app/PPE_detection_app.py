# ======================================================================================================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import os
import time

# =====================================================================================================================
# File paths
logo_path = ('/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/logos'
             '/EagleEye_logo_green_black_text_no_background.png')
model_path_s = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/small_yolo_model/ppe_s.pt"
model_path_m = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/medium_yolo_model/ppe_m.pt"
model_path_l = "/Users/jameskenny/Documents/GitHub/Ironhack/PPE_detection/models/large_yolo_model/ppe_l.pt"

# =====================================================================================================================
# Initializing session states
# Initialize the camera and model paths in session state if not already present
if 'init' not in st.session_state:
    st.session_state['init'] = True

if 'cap' not in st.session_state:
    st.session_state['cap'] = None  # Initial state of the camera is not opened

if 'camera_started' not in st.session_state:
    st.session_state['camera_started'] = None

if 'camera_index' not in st.session_state:
    st.session_state['camera_index'] = 0

if 'model' not in st.session_state:  # Check if the model is already loaded
    st.session_state['model'] = YOLO(model_path_s)  # Load the model and store it in session state

if 'class_info' not in st.session_state:  # Check if the class info is already loaded
    st.session_state['class_info'] = {
        0: {'class_name': 'Hardhat', 'color': (136, 242, 7), 'include': True, 'positive_class': True, 'type': 'ppe'},
        1: {'class_name': 'Mask', 'color': (136, 242, 7), 'include': True, 'positive_class': True, 'type': 'ppe'},
        2: {'class_name': 'NO-Hardhat', 'color': (70, 7, 242), 'include': True, 'positive_class': False,
            'type': 'violation'},
        3: {'class_name': 'NO-Mask', 'color': (70, 7, 242), 'include': True, 'positive_class': False,
            'type': 'violation'},
        4: {'class_name': 'NO-Safety Vest', 'color': (70, 7, 242), 'include': True, 'positive_class': False,
            'type': 'violation'},
        5: {'class_name': 'Person', 'color': (255, 238, 82), 'include': True, 'positive_class': True,
            'type': 'object'},
        6: {'class_name': 'Safety Cone', 'color': (255, 255, 0), 'include': True, 'positive_class': True,
            'type': 'object'},
        7: {'class_name': 'Safety Vest', 'color': (136, 242, 7), 'include': True, 'positive_class': True,
            'type': 'ppe'},
        8: {'class_name': 'machinery', 'color': (255, 105, 180), 'include': True, 'positive_class': True,
            'type': 'object'},
        9: {'class_name': 'vehicle', 'color': (255, 105, 180), 'include': True, 'positive_class': True,
            'type': 'object'}
    }

if 'frame_rate' not in st.session_state:
    st.session_state['frame_rate'] = 0
if 'historical_compliance' not in st.session_state:
    st.session_state['historical_compliance'] = []  # List to store historical compliance percentages
if 'current_compliance' not in st.session_state:
    st.session_state['current_compliance'] = 0  # Variable to store current frame's compliance
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = 0.55  # Variable to store current frame's compliance


# ======================================================================================================================
# Defining functions

# load model
# store model output in cache
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


# Function to start the camera
def start_camera():
    if st.session_state['cap'] is None:
        st.session_state['cap'] = cv2.VideoCapture(st.session_state['camera_index'])  # Start the camera
        st.session_state['cap'].set(3, 1280)  # Set the width to 1280
        st.session_state['cap'].set(4, 720)  # Set the height to 720
        st.session_state['camera_started'] = True


# Function to stop the camera
def stop_camera():
    if st.session_state['cap'] is not None:
        st.session_state['cap'].release()  # Release the camera
        st.session_state['cap'] = None
        st.session_state['camera_started'] = False
        frame_placeholder.empty()  # Clear the placeholder completely


# Function to switch the camera input
def switch_camera():
    st.session_state['camera_index'] = 1 - st.session_state.get('camera_index', 0)
    stop_camera()
    start_camera()


def process_frame(frame, model, confidence_threshold, class_info):
    positive_ppe_count = 0
    violation_count = 0
    results = model(frame, conf=confidence_threshold, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if class_info[cls]['include']:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                conf = math.ceil(box.conf[0] * 100) / 100
                if class_info[cls]['type'] == 'ppe':
                    positive_ppe_count += 1
                elif class_info[cls]['type'] == 'violation':
                    violation_count += 1
                class_name = class_info[cls]['class_name']
                color = class_info[cls]['color']

                # cvzone bounding box
                w, h = x2 - x1, y2 - y1  # set up for cvzone bounding boxes
                # Add a rectangle with styled corners to the image
                cvzone.cornerRect(
                    frame,  # The image to draw on
                    (x1, y1, w, h),  # The position and dimensions of the rectangle (x, y, width, height)
                    l=30,  # Length of the corner edges
                    t=5,  # Thickness of the corner edges
                    rt=0,  # Thickness of the rectangle
                    colorR=(255, 0, 255),  # Color of the rectangle
                    colorC=color  # Color of the corner edges
                )
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cvzone.putTextRect(frame, f'{class_name} {conf}', (x1, max(35, y1)), scale=0.85, thickness=1,
                                   colorB=color, colorT=(0, 0, 0), colorR=color, offset=2)
    # Calculate compliance for the current frame
    total_detections = positive_ppe_count + violation_count
    compliance = (positive_ppe_count / total_detections) * 100 if total_detections > 0 else 100
    st.session_state['current_compliance'] = compliance
    return frame


def update_historical_compliance(current_compliance):
    st.session_state['historical_compliance'].append(current_compliance)
    # Assume 10 frames per second as an example, adjust based on your actual frame rate
    if len(st.session_state['historical_compliance']) > 300:  # 30 seconds * 10 frames per second
        st.session_state['historical_compliance'].pop(0)


# ======================================================================================================================
# Main page UI


# st.header("PPE Compliance Monitoring System")

# Camera feed
frame_placeholder = st.empty()

# ======================================================================================================================
# Sidebar UI

# Logo
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image(logo_path, width=125)  # Adjust width as needed

# title
st.sidebar.markdown(
    "<h1 style='text-align: center; color: #355E66; font-size: 35px; font-weight: bold;'>Control Panel</h1>",
    unsafe_allow_html=True)

# Start/Stop camera buttons
st.sidebar.subheader('Camera Controls')
col4, col5, col6 = st.sidebar.columns([1, 1, 1])
with col4:
    start_button = st.button('Start Camera', on_click=start_camera)
with col5:
    stop_button = st.button('Stop Camera', on_click=stop_camera)
with col6:
    switch_button = st.button('Switch Camera', on_click=switch_camera)

# User input for selecting model
model_options = ['YOLO v8 Small', 'YOLO v8 Medium', 'YOLO v8 Large']
st.sidebar.subheader('Object detection model')
st.sidebar.markdown("- **Small:** fast but less accurate\n"
                    "- **Medium:** balances speed and accuracy\n"
                    "- **Large:** accurate but slow\n"
                    "\n<small>Note: Large model requires high-performance hardware for optimal operation.</small>",
                    unsafe_allow_html=True)
selected_model = st.sidebar.selectbox('Select model:', model_options, index=0)
if selected_model == 'YOLO v8 Large':
    st.session_state['model'] = load_model(model_path_l)
elif selected_model == 'YOLO v8 Medium':
    st.session_state['model'] = load_model(model_path_m)
else:
    st.session_state['model'] = load_model(model_path_s)

# conf_options = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
st.session_state['confidence_threshold'] = st.sidebar.slider('Select detection confidence threshold:', 0.25, 0.9, 0.55)

# User input for selecting classes to included
# Sidebar header for user input
st.sidebar.subheader('Select objects to detect')

# Create two columns with subheadings for PPE and Other
col_ppe, col_other = st.sidebar.columns(2)
with col_ppe:
    st.markdown('**PPE**')
with col_other:
    st.markdown('**Other**')

# Iterate through class_info and organize checkboxes based on 'type'
for class_id, info in st.session_state['class_info'].items():
    if info['positive_class']:  # Only include positive classes for selection
        # Determine the column based on the 'type'
        current_col = col_ppe if info['type'] == 'ppe' else col_other

        # Place checkbox in the appropriate column
        include = current_col.checkbox(f'{info["class_name"]}', value=info['include'])

        # Update the session state based on checkbox input
        st.session_state['class_info'][class_id]['include'] = include

        # Additional logic to mirror the selection for corresponding negative classes
        if class_id == 0:  # Example for Hardhat
            st.session_state['class_info'][2]['include'] = include
        elif class_id == 1:  # Example for Mask
            st.session_state['class_info'][3]['include'] = include
        elif class_id == 7:  # Example for Safety Vest
            st.session_state['class_info'][4]['include'] = include

# st.sidebar.button('Reset to Default', on_click=reset_settings)
# ======================================================================================================================
# Analytics


st.metric(label="System Frame Rate", value=f"{st.session_state['frame_rate']} fps")

st.metric(label="Current PPE Compliance", value=f"{st.session_state['current_compliance']:.2f}%")
# st.metric(label="Historical PPE Compliance", value=f"{st.session_state['historical_compliance']:.2f}%")

st.line_chart(st.session_state['historical_compliance'])

# ======================================================================================================================
# Main loop for frame processing
start_time = time.time()
frame_count = 0
if st.session_state.get('camera_started', False):
    while True:
        ret, frame = st.session_state['cap'].read()
        if not ret:
            st.write("Failed to capture image")
            stop_camera()
            break

        # Process the frame with PPE detection model
        processed_frame = process_frame(frame, st.session_state['model'], st.session_state['confidence_threshold'],
                                        st.session_state['class_info'])

        # Display the processed frame in the Streamlit app
        frame_placeholder.image(processed_frame, channels="BGR")

        frame_count += 1
        if frame_count % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.session_state['frame_rate'] = frame_count / elapsed_time
            # Reset counters
            start_time = time.time()
            frame_count = 0

        update_historical_compliance(st.session_state['current_compliance'])

        # Check if the stop button has been pressed
        if stop_button:
            stop_camera()
            break

# ======================================================================================================================
