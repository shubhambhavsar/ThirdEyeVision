import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition as sr
import tempfile

# Let's assume `audio_segment` is your AudioSegment object
audio_segment = audiorecorder("Click to record", "Click to stop recording")

# Initialize the recognizer
recognizer = sr.Recognizer()

# Step 1: Export the AudioSegment object to a temporary WAV file
with tempfile.NamedTemporaryFile(delete=True) as tmp:
    file_name = f"{tmp.name}.wav"  # Temporary file name
    audio_segment.export(file_name, format="wav")  # Export as WAV

    # Step 2: Use the temporary WAV file with speech_recognition
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)  # Read the entire audio file
        
        # Attempt to recognize the speech in the audio
        try:
            text = recognizer.recognize_google(audio_data)  # Using Google Web Speech API
            st.write(f"Recognized text: {text}")
        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")