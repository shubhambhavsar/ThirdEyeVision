from ultralytics import YOLO
import cv2
import numpy as np
# import pyttsx3
import base64
from gtts import gTTS
from io import BytesIO
import streamlit as st

def speak(text):
    
    tts = gTTS(text, lang='en')   # Create a text-to-speech object with the given text and language set to English
    
    audio_bytes_io = BytesIO() # Create a BytesIO object to hold the audio data
    
    tts.write_to_fp(audio_bytes_io)  # Write the audio data to the BytesIO object
    
    audio_bytes_io.seek(0)   #Seek to the beginning of the BytesIO stream
    
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')   # Encode the audio data in base64 to embed in HTML
    
    audio_html = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'  # Create the HTML code for an audio player with the encoded audio data
    
    st.components.v1.html(audio_html, height=50) # Use Streamlit's HTML component to display the audio player in the app


# #initiate tts engine and define properties
# engine=pyttsx3.init()
# engine.setProperty('rate',100)
# engine.setProperty('volume',0.7)

# Load YOLO models
model_traffic = YOLO('yolov8n.pt')
model_class = YOLO(r"bestclasscpr3.pt")

# cap = cv2.VideoCapture(r"C:\Users\user\OneDrive - Loyalist College\AIandDS\Term 2\Step_Presentation\Streamlit_Apps\sameer\walkstop.mp4")


def main_func_ped(cap, confidence, margin):

    frame_skip = 5  # Number of frames to skip between detections. Adjust based on your needs.
    frame_count = 0

    
    pred=''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every nth frame (where n is frame_skip)
        if frame_count % frame_skip == 0:
            results_traffic = model_traffic(frame,classes=[9],show=True,conf=confidence)
            detected_boxes = results_traffic[0].boxes.data
            detected_boxes = detected_boxes.detach().cpu().numpy()

            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Calculate the center and margins for filtering
                width = frame.shape[1]
                center_traffic = (x1 + x2) / 2
                left_margin = width * margin   #0-30
                right_margin = width * 1-margin  #0-30 do 1 minus the value between 0 to 30

                # Process only if the traffic light is within the desired margins
                if left_margin < center_traffic < right_margin:
                    traffic_light_img = frame[y1:y2, x1:x2]
                    results_class = model_class(traffic_light_img, conf=confidence)
                    probs = results_class[0].probs.data.cpu().numpy()

                    max_prob_index = np.argmax(probs)  # Index of highest probability
                    max_class_name = results_class[0].names[max_prob_index]  # Class name of highest probability
                    max_prob = probs[max_prob_index]  # Maximum probability value
                    if pred==max_class_name or max_class_name=='ignore':
                        pass
                    else:
                        data = speak(max_class_name)
                        pred = max_class_name
        frame_count += 1  # Increment frame counter
            
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    cap.release()
    cv2.destroyAllWindows()
