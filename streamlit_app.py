import streamlit as st
from ultralytics import YOLO
import cv2
import base64
from gtts import gTTS
from io import BytesIO


# Initialize text-to-speech engine
# engine = pyttsx3.init()

def speak(text):
    tts = gTTS(text, lang='en')
    audio_bytes_io = BytesIO()
    tts.write_to_fp(audio_bytes_io)
    audio_bytes_io.seek(0)
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')
    audio_html = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    st.components.v1.html(audio_html, height=50)

class_name = "dummy text"
speak(class_name)



# def detect_objects(image_path):
#     """
#     Detects objects in an image using YOLOv8 and draws bounding boxes with labels.
#     Speaks out a message when an object is detected.

#     Args:
#         image_path (str): Path to the image file.

#     Returns:
#         img: Image with bounding boxes and labels drawn.
#     """

#     # Load YOLO model
#     model = YOLO("yolov8n.pt")

#     # Read image
#     img = cv2.imread(image_path)

#     # Run object detection
#     results = model(img)

#     # Check if there are any detections
#     if len(results) == 0:
#         print("No objects detected.")
#         return img

#     # Draw bounding boxes and labels
#     for r in results:
#         for b in r.boxes:
#             class_id = int(b.cls.item())
#             class_name = r.names[class_id]
#             x_min, y_min, x_max, y_max = map(int, b.xyxy[0])
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(img, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             # Speak out a message when an object is detected

#             audio_html = speak(class_name)

#             st.components.v1.html(audio_html, height=50)
#             # speak(f"There is a {class_name}")

#     return img

# # Streamlit app
# st.title("Object Detection with Text-to-Speech")

# # File uploader for image
# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Perform object detection when file is uploaded
#     with st.spinner("Performing object detection..."):
#         image_path = "temp_image.jpg"  # Save uploaded image to a temporary file
#         with open(image_path, "wb") as f:
#             f.write(uploaded_file.getvalue())
#         detected_image = detect_objects(image_path)

#     # Display the detected image
#     st.image(detected_image, caption="Detected Objects", use_column_width=True)
