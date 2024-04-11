from ultralytics import YOLO
import cv2
import numpy as np
# import pyttsx3
import base64
from gtts import gTTS
from io import BytesIO
import streamlit as st
import time
import torch
import pandas as pd

def speak(text):
    
    tts = gTTS(text, lang='en')   # Create a text-to-speech object with the given text and language set to English
    
    audio_bytes_io = BytesIO() # Create a BytesIO object to hold the audio data
    
    tts.write_to_fp(audio_bytes_io)  # Write the audio data to the BytesIO object
    
    audio_bytes_io.seek(0)   #Seek to the beginning of the BytesIO stream
    
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')   # Encode the audio data in base64 to embed in HTML
    
    audio_html = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'  # Create the HTML code for an audio player with the encoded audio data
    
    st.components.v1.html(audio_html, height=50) # Use Streamlit's HTML component to display the audio player in the app


# Load YOLO models
model_traffic = YOLO('yolov8n.pt')
model_class = YOLO("bestclasscpr3.pt")
model_obj = YOLO("bestgrayscale.pt")


def main_func_ped(cap, confidence, margin, vid_type):
    frame_counter_slot = st.empty()
    model_counter_slot=st.empty()
    prev_time = 0
    curr_time = 0	
    st_frame = st.empty()

    
    pred=''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_img_3c = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        # Ensure the image dimensions are divisible by 32
        # Calculate new dimensions that are closest to the original dimensions but divisible by 32
        height, width = grayscale_img_3c.shape[:2]
        new_height = (height + 31) // 32 * 32  # Round up to the nearest value divisible by 32
        new_width = (width + 31) // 32 * 32
        # Resize the image
        resized_img = cv2.resize(grayscale_img_3c, (new_width, new_height))
        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).float()
        #model expects pixel values in the range [0, 255]
        input_tensor = input_tensor / 255.0
        # Now, 'input_tensor' is ready for model input and should meet the requirements
        start_time = time.time() 

        results_obj=model_obj(input_tensor,conf=confidence)

        end_time = time.time()

        # Measure model inference time
        model_inference_time = end_time - start_time 
        a=results_obj[0].boxes.data
        a=a.detach().cpu().numpy()
        if vid_type == 'Show-Video':
            res_plotted = results_obj[0].plot()
            st_frame.image(res_plotted,
                            caption='Detected Video',
                            use_column_width=True,
                            channels="BGR")
        else:
            st_frame = st.empty()
        px = pd.DataFrame(a).astype("float")
        if len(px)==0 or len(px)==2:
            # Start timing for model inference
            start_time = time.time()  
            
            results_traffic = model_traffic(frame,classes=[9],conf=confidence)
            
            end_time = time.time()
            
            # Measure model inference time
            model_inference_time = end_time - start_time  
            
            detected_boxes = results_traffic[0].boxes.data
            detected_boxes = detected_boxes.detach().cpu().numpy()
            # if vid_type == 'Show-Video':
            #     res_plotted = results_traffic[0].plot()
            #     st_frame.image(res_plotted,
            #                 caption='Detected Video',
            #                 use_column_width=True,
            #                 channels="BGR")
            # else:
            #     st_frame = st.empty()
    
            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
    
                # Calculate the center and margins for filtering
                width = frame.shape[1]
                center_traffic = (x1 + x2) / 2
                left_margin = width * margin
                right_margin = width * 1-margin
    
                # Process only if the traffic light is within the desired margins
                if left_margin < center_traffic < right_margin:
                    traffic_light_img = frame[y1:y2, x1:x2]
                    results_class = model_class(traffic_light_img)
                    probs = results_class[0].probs.data.cpu().numpy()
    
                    max_prob_index = np.argmax(probs)  # Index of highest probability
                    max_class_name = results_class[0].names[max_prob_index]  # Class name of highest probability
                    max_prob = probs[max_prob_index]  # Maximum probabiliy vatlue
                    if pred==max_class_name or max_class_name=='ignore':
                        pass
                    else:
                        data = speak(max_class_name)
                        pred = max_class_name
            
        else:
            for r in results_obj:
                for b in r.boxes:
                    class_tensor=b.cls
                    class_id=int(class_tensor.item())
                    class_name=r.names[class_id]
                    class_name=str(class_name)
                    if pred==class_name:
                        pass
                    else:
                        data = speak(class_name)
                        pred=class_name
        
        curr_time = time.time()
        model_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        frame_counter_slot.write(f'Model Fps: {model_fps:.2f}')
        model_counter_slot.write(f"Model Inference Time: {model_inference_time*1000:.2f}ms")

    cap.release()
    cv2.destroyAllWindows()
