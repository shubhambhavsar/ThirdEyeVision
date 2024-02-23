import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
import base64
from io import BytesIO

# Initialize YOLO model
model = YOLO("best_Model_Roboflow.pt")

def speak(text):
    # Generate speech from text
    tts = gTTS(text, lang='en')
    audio_bytes_io = BytesIO()
    tts.write_to_fp(audio_bytes_io)
    audio_bytes_io.seek(0)
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
    # Embed audio in HTML for Streamlit
    audio_html = f'<audio autoplay controls><source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

class_name = "dummy text"
speak(class_name)
