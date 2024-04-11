import cv2
import torch
from ultralytics import YOLO
import easyocr
import numpy as np
from collections import Counter 
import re
import streamlit as st
import base64
from gtts import gTTS
from io import BytesIO
import time


def speak(text):
    
    tts = gTTS(text, lang='en')   # Create a text-to-speech object with the given text and language set to English
    
    audio_bytes_io = BytesIO() # Create a BytesIO object to hold the audio data
    
    tts.write_to_fp(audio_bytes_io)  # Write the audio data to the BytesIO object
    
    audio_bytes_io.seek(0)   #Seek to the beginning of the BytesIO stream
    
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')   # Encode the audio data in base64 to embed in HTML
    
    audio_html = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'  # Create the HTML code for an audio player with the encoded audio data
    
    st.components.v1.html(audio_html, height=50) # Use Streamlit's HTML component to display the audio player in the app

def preprocess_for_ocr(im):
    if im is None or im.size == 0:
        return None

    # Set confidence threshold for OCR
    conf = 0.4
    
    # Resize the image to zoom in
    zoom_factor = 4  # Adjust this factor as needed
    resized_width = int(im.shape[1] * zoom_factor)
    resized_height = int(im.shape[0] * zoom_factor)
    if resized_width == 0 or resized_height == 0:
        return None

    zoomed_image = cv2.resize(im, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    # Apply noise reduction techniques for better OCR results
    blurred_image = cv2.GaussianBlur(zoomed_image, (5, 5), 0)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    
    return gray_image


# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Function to perform OCR on a region of interest (ROI) in an image
def getOCR(im, coors):
    if im is None:
        return ""

    # Extract coordinates of the region of interest (ROI)
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    # Crop the ROI from the image
    im = im[y:h, x:w]
    
    # Preprocess image for OCR
    preprocessed_im = preprocess_for_ocr(im)
    if preprocessed_im is None:
        return ""

    # Perform OCR on the preprocessed image
    results = reader.readtext(preprocessed_im)
    
    # Store occurrences of each recognized text
    text_occurrences = Counter()

    # Loop through OCR results
    for result in results:
        bbox, text, score = result
        if len(text) > 2:  # Ensure text is meaningful
            # Use regex to filter only alphabetic characters
            filtered_text = " ".join(re.findall("[A-Za-z]+", text))
            if filtered_text:  # Ensure filtered text is not empty
                # Convert to uppercase and remove spaces for better counting
                cleaned_text = filtered_text.upper()
                text_occurrences[cleaned_text] += 1

    # Return the most occurred text
    if text_occurrences:
        most_common_text = text_occurrences.most_common(1)[0][0]
        return most_common_text
    else:
        # If no common text found, return any single text if available
        if results:
            # Apply regex filter to the first recognized text before returning
            return " ".join(re.findall("[A-Za-z]+", results[0][1]))
        else:
            return "" 


def main_func(cap, model, confidence, vid_type):
    frame_counter_slot = st.empty()
    model_counter_slot=st.empty()
    prev_time = 0
    curr_time = 0								   
								 
				 
				 
    # Initialize global counter for text occurrences across all frames
    text_occurrences_global = Counter()

    # Initialize a counter for processed frames
    processed_frames_count = 0

    # Set the number of frames you want to process
    n_frames_to_process = 20
    st_frame = st.empty()

    # Start timing for total processing
    # total_start_time = time.time()  


    # Loop through video frames
    while cap.isOpened() and processed_frames_count < n_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            # cv2.destroyAllWindows()
            break

        # Start timing for model inference
        start_time = time.time()  

        # Perform detection on the current frame
        results = model(frame, conf=confidence)

        end_time = time.time()

        # Measure model inference time
        model_inference_time = end_time - start_time  

        # Iterate over detected objects
        # Assuming 'results' is obtained from your YOLO detection
        street_plates = results[0]
        result_tensor = results[0].boxes
        if vid_type == "Show-Video":
            res_plotted = results[0].plot()
            st_frame.image(res_plotted,
                               caption='Detected Video',
                               use_column_width=True,
                               channels="BGR")
        else:
            st_frame = st.empty()
        
        for street_plate in street_plates.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = street_plate

            # Extract text from the detected region
            text = getOCR(frame, [x1, y1, x2, y2])
            print("Detected Text:", text)  # Printing the detected text

            # Add the detected text to the global counter
            if text:  # Ensure the text is not empty
                cleaned_text = text.upper()
                text_occurrences_global[cleaned_text] += 1
        
        # Increment the processed frames counter
        processed_frames_count += 1
        curr_time = time.time()
        model_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        frame_counter_slot.write(f'Model Fps: {model_fps:.2f}')
        model_counter_slot.write(f"Model Inference Time: {model_inference_time*1000:.2f}ms")							   
											   
							 
															   
																							
    
    st_frame.empty()

    most_common = ""
    # After processing all frames
    if text_occurrences_global:
        most_common_text, occurrence = text_occurrences_global.most_common(1)[0]
        # print("Most Occurred Text:", most_common_text, "with", occurrence, "occurrences")
        most_common = most_common_text
    else:
        most_common = "No text detected."
					  

    
    return most_common