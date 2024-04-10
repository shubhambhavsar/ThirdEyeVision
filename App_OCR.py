import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import cv2
from object_det_app import *
from Pedestrian import *
from speed_modular import *
import base64
from audiorecorder import audiorecorder
import speech_recognition as sr
import tempfile

# welcome_message_played = False


im = Image.open('eye.png')

# Replace the relative path to your weight file
model_path = "best_Model_Roboflow.pt"
# Setting page layout
st.set_page_config(
        page_title="Third Eye Vision",  # Setting page title
        page_icon=im,  # Setting page icon
        # layout="wide",      # Setting layout to wide
        initial_sidebar_state="expanded"  # Expanding sidebar by default
    )

import sqlite3
from hashlib import sha256

# Database setup
DB_FILE = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, password):
    """Add a new user with a hashed password."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashed_password = sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:  # Username already exists
        return False
    finally:
        conn.close()

def authenticate(username, password):
    """Check if a username/password combination is valid."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    user_password = c.fetchone()
    conn.close()
    if user_password:
        hashed_password = sha256(password.encode()).hexdigest()
        return user_password[0] == hashed_password
    return False

init_db()  # Initialize the database

def speak_welc(text):
    
    tts = gTTS(text, lang='en')   # Create a text-to-speech object with the given text and language set to English
    
    audio_bytes_io = BytesIO() # Create a BytesIO object to hold the audio data
    
    tts.write_to_fp(audio_bytes_io)  # Write the audio data to the BytesIO object
    
    audio_bytes_io.seek(0)   #Seek to the beginning of the BytesIO stream
    
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')   # Encode the audio data in base64 to embed in HTML
    
    audio_html = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'  # Create the HTML code for an audio player with the encoded audio data
    
    st.components.v1.html(audio_html, height=0) # Use Streamlit's HTML component to display the audio player in the app


def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        encoded_string = base64.b64encode(image_file.read()).decode()
# Set the background image style
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
)

image_path = r'image_bg.jpg'  # Replace this with the path to your image file


def welcome_page():
    set_background_image(image_path)
    st.title("Third Eye Vision")
    selected = option_menu(
        menu_title=None,
        options=["HOME", "ABOUT", "CONTACT"],
        icons=["house", "briefcase", "person-lines-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "HOME":
        # Add CSS to create a white background for text content #262730
        st.markdown("""
        <style>
            .textbox, .element-container st-emotion-cache-kdanly e1f1d6gn4 {
            background-color: rgba(255, 255, 255, 1); /* Semi-transparent white */
            padding: 10px; /* Some padding around the text */
        }

        /* Override styles for dark theme */
        @media (prefers-color-scheme: dark) {
            .textbox, .element-container st-emotion-cache-kdanly e1f1d6gn4 {
            background-color: #262730; /* Semi-transparent white */
            padding: 10px; /* Some padding around the text */                   
        }
        }            

        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
                                
        <style>
            .custom-font {
                font-size:16px; /* Adjust the size as needed */
                font-family: Arial, Helvetica, sans-serif; 
            }
            .highlight {
                color: #3467eb;  /* Change the color as per your application theme */
            }
            .bold {
                font-weight: bold;
            }
        </style>

        <div class="custom-font">
            <div class="textbox">
            Welcome to <span class="highlight bold">Third Eye!</span> Our application aims to support and help visually impaired individuals in navigating their surroundings better and be alert of dangers through text-to-speech.
            <br><br>
            We provide 3 types of computer vision detections:<br>
            - <span class="bold">"Alert"</span> system lets you know about different types of vehicles and objects approaching you.<br>
            - <span class="bold">"Street Name"</span> system lets you know about the street name.<br>
            - <span class="bold">"Pedestrian"</span> system informs you about what is displayed on the Pedestrian crosswalk signal.<br>
            <br>
            To get started, you can either speak the keyword for the detection you require, or click on the <span class="highlight bold">Get Started</span> button to access more features and tweak parameters!<br><br>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("Get Started"):
            st.session_state['current_page'] = "data_science"
            st.experimental_rerun()

        # Adding file uploader to sidebar for selecting videos
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4","jpg","png", "mov"])
        temporary_location = None
        text_ab = None

        if uploaded_file is not None:
            temporary_location = "testout_simple.mp4"
            try:
                with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                    out.write(uploaded_file.read())  ## Read bytes into file
            except PermissionError:
                st.error("Permission denied to write the temporary file. Please check your permissions.")
            except Exception as e:
                st.error(f"Error saving uploaded file: {e}")
        
        try:
            model = YOLO(model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

        # Creating a session state to store the uploaded video's state
        if 'video_uploaded' not in st.session_state:
            st.session_state.video_uploaded = False

        # Check if a new video has been uploaded
        if uploaded_file is not None and not st.session_state.video_uploaded:
            st.session_state.video_uploaded = True
            st.experimental_rerun()
        
        if st.session_state.video_uploaded:
            speak_welc("Welcome to Third Eye. Please speak pedestrian to know pedestrian signal, speak street name to know the street name, speak alert to initiate the alert system. You may speak now.")

        # Let's assume `audio_segment` is your AudioSegment object
        audio_segment = audiorecorder("Click to record", "Click to stop recording")

        # Initialize the recognizer
        recognizer = sr.Recognizer()
        text_ab = ""
        # Step 1: Export the AudioSegment object to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            file_name = f"{tmp.name}.wav"  # Temporary file name
            audio_segment.export(file_name, format="wav")  # Export as WAV

            # Step 2: Use the temporary WAV file with speech_recognition
            with sr.AudioFile(file_name) as source:
                audio_data = recognizer.record(source)  # Read the entire audio file
                
                # Attempt to recognize the speech in the audio
                try:
                    text_a = recognizer.recognize_google(audio_data)  # Using Google Web Speech API
                    text_ab = text_a
                    st.write(f"Recognized text: {text_a}")
                except sr.UnknownValueError:
                    speak_welc("Sorry, could not understand the audio.")
                except sr.RequestError as e:
                    pass

        # If a video has been uploaded and detected, start object detection
        if st.session_state.video_uploaded:
            vid_cap = cv2.VideoCapture(temporary_location)
            if text_ab.lower() == "street":
                
                most_common= main_func(vid_cap, model, confidence=0.35, vid_type="Hide-Video")

                # Display the most common text
                st.write("Most common text:", most_common)

                audio_html = speak(most_common)
            if text_ab.lower() == "pedestrian":            
                main_func_ped(vid_cap, confidence=0.35, margin=0.10, vid_type="Hide-Video")

            if text_ab.lower() == "alert":     
                main_func_alert(vid_cap,user_conf_value=0.35, margin=0.1, user_class_id=[1, 2, 3, 5, 7], user_fps_value=1, vid_type="Hide-Video")
        text_ab = ""

    page_bg_img = f"""
    <style>
    [data-testid="stFileUploader"] {{
    background-color: rgb(255, 255, 255);
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    padding: 10px; /* Some padding around the text */
    margin: 10px 0; /* Some space above and below the text box */
    }}

    /* Override styles for dark theme */
    @media (prefers-color-scheme: dark) {{
        [data-testid="stFileUploader"] {{
        background-color: #262730;
        background-size: 180%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
        padding: 10px; /* Some padding around the text */
        margin: 10px 0; /* Some space above and below the text box */
    }}}}

    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Header Section
    css = """
    <style>
        h1{
            color: white;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)




    if selected == "ABOUT":
        css = """
        <style>
            h2,p,li {
            color: white;
            }
        </style>
        """
        st.markdown(css,unsafe_allow_html=True)
        st.header("🚀 Elevate Independence")
        st.write("""
            "Third Eye" leverages groundbreaking computer vision technology to empower blind and low-vision individuals. Navigate your surroundings with newfound confidence as "Third Eye" interprets the world in real-time.
            """)

        st.header("🌟 Features")
        st.write("""
            - **Object Recognition:** Instantly identify obstacles, signage, and objects.
            - **Text-to-Speech:** Hear descriptions of your surroundings, making navigation intuitive.
            - **Navigation Support:** Get real-time assistance to move around safely and efficiently.
            """)

        st.header("💡 How It Works")
        st.write("""
        By analyzing live video feeds, "Third Eye" detects and vocalizes the presence of obstacles and signage, converting visual information into audible guidance. This real-time support system is designed to promote greater inclusivity and accessibility, enhancing the daily lives of visually impaired individuals.
        """)

        # Interactive demo or more information about the technology could go here

        st.header("🌐 Join Our Community")
        st.write("""
        Become a part of the "Third Eye" community and contribute to a world where technology bridges the gap towards a more inclusive society. Share your experiences, suggest improvements, and help us make "Third Eye" better for everyone.
        """)

        st.markdown('</div>', unsafe_allow_html=True)



    if selected == "CONTACT":
        css2 = """
        <style>
            [data-testid="StyledLinkIconContainer"]{
            color: white;
            }
            [data-testid="stWidgetLabel"]{
            color: white;
            }
        .st-emotion-cache-eqffof.e1nzilvr5 p {
            color: white;
        }
        </style>
        """
        st.markdown(css2, unsafe_allow_html=True)


        # Add a Contact Us section
        st.header("📬 Contact Us")
        st.write(
            "We'd love to hear from you! Whether you have a question, feedback, or just want to say hello, please don't hesitate to reach out.")

        with st.form("contact_form"):
            name = st.text_input("Name", placeholder="Your Name")
            email = st.text_input("Email", placeholder="Your Email Address")
            message = st.text_area("Message", placeholder="Your Message Here")

            submit_button = st.form_submit_button("Send Message")

            if submit_button:
                # Here you would include the logic to handle the form data, such as sending an email
                # This is a placeholder to simulate form submission
                st.success(f"Thank you, {name}, for reaching out! We'll get back to you soon.")


# Data Scientist page
def data_science_page():
    set_background_image(image_path)
    page_bg_img = f"""
    <style>
    [data-testid="stTab"]{{
    background-color: rgb(255, 255, 255);
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    padding: 10px; /* Some padding around the text */
    margin: 10px 0; /* Some space above and below the text box */
    }}

    /* Override styles for dark theme */
    @media (prefers-color-scheme: dark) {{
        [data-testid="stTab"]{{
        background-color: #262730;
        background-size: 180%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
        padding: 10px; /* Some padding around the text */
        margin: 10px 0; /* Some space above and below the text box */
        }}}}


    [data-testid="textInputRootElement"]{{
    background-color: rgb(255, 255, 255);
    border-color: black
    }}

    /* Override styles for dark theme */
    @media (prefers-color-scheme: dark) {{
        [data-testid="textInputRootElement"]{{
        background-color: #262730;
        border-color: white
        }}}}

    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Header Section
    css = """
    <style>
        h1 {
            color: white;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


    col1, col2 = st.columns([0.9, 0.18])  # Adjust the ratio as needed
    with col2:
        if st.button("Home Page"):
            st.session_state['current_page'] = "welcome"
            st.experimental_rerun()

    st.title("Data Scientist Page!")
    selected = option_menu(
        menu_title=None,
        options=["SIGN UP / LOGIN", "ABOUT", "CONTACT"],
        icons=["", "briefcase", "person-lines-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    if selected == "SIGN UP / LOGIN":
        # if st.button("Get Started"):
        #     st.session_state['current_page'] = "signup_login"
        #     st.experimental_rerun()
        # st.title("Sign Up / Login")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        
        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if authenticate(username, password):
                    st.session_state['current_page'] = "app"
                    st.session_state['username'] = username
                    # Force a rerun of the app to immediately reflect the updated state
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
                    
        with signup_tab:
            new_username = st.text_input("Enter Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            if st.button("Sign Up"):
                if add_user(new_username, new_password):
                    st.success("Account created successfully. Please login.")
                else:
                    st.error("This username is already taken.")
    
    if selected == "ABOUT":

        css2 = """
        <style>
            [data-testid="StyledLinkIconContainer"],li{
            color: white;
            }
        .st-emotion-cache-eqffof.e1nzilvr5 p {
            color: white;
        }
        </style>
        """
        st.markdown(css2, unsafe_allow_html=True)
    
        with st.container():
            st.markdown(css, unsafe_allow_html=True)
            st.header("🚀 Elevate Independence")
            st.write("""
                "Third Eye" leverages groundbreaking computer vision technology to empower blind and low-vision individuals. Navigate your surroundings with newfound confidence as "Third Eye" interprets the world in real-time.
                """)

            st.header("🌟 Features")
            st.markdown("""
            - **Object Recognition:** Instantly identify obstacles, signage, and objects.
            - **Text-to-Speech:** Hear descriptions of your surroundings, making navigation intuitive.
            - **Navigation Support:** Get real-time assistance to move around safely and efficiently.
            """)

            st.header("💡 How It Works")
            st.write("""
            By analyzing live video feeds, "Third Eye" detects and vocalizes the presence of obstacles and signage, converting visual information into audible guidance. This real-time support system is designed to promote greater inclusivity and accessibility, enhancing the daily lives of visually impaired individuals.
            """)

            # Interactive demo or more information about the technology could go here

            st.header("🌐 Join Our Community")
            st.write("""
            Become a part of the "Third Eye" community and contribute to a world where technology bridges the gap towards a more inclusive society. Share your experiences, suggest improvements, and help us make "Third Eye" better for everyone.
            """)

    if selected == "CONTACT":
        css2 = """
        <style>
            [data-testid="StyledLinkIconContainer"]{
            color: white;
            }
            [data-testid="stWidgetLabel"]{
            color: white;
            }
        .st-emotion-cache-eqffof.e1nzilvr5 p {
            color: white;
        }
        </style>
        """
        st.markdown(css2, unsafe_allow_html=True)

        # Add a Contact Us section
        st.header("📬 Contact Us")
        st.write(
            "We'd love to hear from you! Whether you have a question, feedback, or just want to say hello, please don't hesitate to reach out.")

        with st.form("contact_form"):
            name = st.text_input("Name", placeholder="Your Name")
            email = st.text_input("Email", placeholder="Your Email Address")
            message = st.text_area("Message", placeholder="Your Message Here")

            submit_button = st.form_submit_button("Send Message")

            if submit_button:
                # Here you would include the logic to handle the form data, such as sending an email
                # This is a placeholder to simulate form submission
                st.success(f"Thank you, {name}, for reaching out! We'll get back to you soon.")


def app_page():
    col1, col2 = st.columns([0.9, 0.18])  # Adjust the ratio as needed
    with col2:  # Use the second column to place the logout button
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['current_page'] = "welcome"
            st.experimental_rerun()
    with col1:
        st.markdown(f"""
    <span style='font-size: 24px'>Hello, {st.session_state.get('username', 'Guest')}!</span>
    """, unsafe_allow_html=True)

    # Creating sidebar
    with st.sidebar:
        st.header("Image/Video Config")  # Adding header to sidebar
        # Adding file uploader to sidebar for selecting videos
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4","jpg","png","mov"])
        temporary_location = None

        if uploaded_file is not None:
            temporary_location = "testout_simple.mp4"
            try:
                with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                    out.write(uploaded_file.read())  ## Read bytes into file
            except PermissionError:
                st.error("Permission denied to write the temporary file. Please check your permissions.")
            except Exception as e:
                st.error(f"Error saving uploaded file: {e}")

        det_type = st.radio(
            "Select Detection Type",
            ["Street Name", "Pedestrian", "Alert"])
        
        if det_type == "Alert":
            class_type = st.multiselect(
                "Select Class Type",
                ['bicycle', 'car', 'motorcycle', 'bus', 'truck'])

            FPS = st.radio(
                "Select Duration of Seconds",
                [1,2,3])       


        if det_type == "Alert":

            # Model Options
            margin = float(st.slider(
                "Select Frame Margin", 0, 10, 10)) / 100
        if det_type == "Pedestrian":

            # Model Options
            margin = float(st.slider(
                "Select Frame Margin", 0, 25, 25)) / 100
            
        vid_type = st.radio(
            "Select If you want to see video",
            ["Show-Video", "Hide-Video"])
            
        # Model Options
        confidence = float(st.slider(
            "Select Model Confidence", 25, 60, 35)) / 100

    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    # Creating a session state to store the uploaded video's state
    if 'video_uploaded' not in st.session_state:
        st.session_state.video_uploaded = False

    # Check if a new video has been uploaded
    if uploaded_file is not None and not st.session_state.video_uploaded:
        st.session_state.video_uploaded = True
        st.experimental_rerun()

    # Header Section
    st.title("Third Eye Vision")

    selected = option_menu(
        menu_title=None,
        options=["HOME", "ABOUT", "CONTACT"],
        icons=["house", "briefcase", "person-lines-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "HOME":
        st.header("Upload the video file and click on the start detection")


        # If a video has been uploaded and detected, start object detection
        if st.session_state.video_uploaded:
            vid_cap = cv2.VideoCapture(temporary_location)
            if det_type == "Street Name" and st.sidebar.button('Start Detection'):
                
                most_common = main_func(vid_cap, model, confidence, vid_type=vid_type)

                # Display the most common text
                st.write("Most common text:", most_common)

                audio_html = speak(most_common)

            if det_type == "Pedestrian" and st.sidebar.button('Start Detection'):

                main_func_ped(vid_cap, confidence, margin, vid_type=vid_type)
            
            if det_type == "Alert" and st.sidebar.button('Start Detection'):
                class_items = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
                assigned_numbers = [1, 2, 3, 5, 7]

                # Creating a dictionary to map items to their assigned numbers
                item_to_number = dict(zip(class_items, assigned_numbers))
                class_no = [item_to_number.get(item)for item in class_type if item in item_to_number]

                main_func_alert(vid_cap,user_conf_value=confidence, margin=margin, user_class_id=class_no, user_fps_value=FPS, vid_type=vid_type)


    if selected == "ABOUT":

        st.header("🚀 Elevate Independence")
        st.write("""
            "Third Eye" leverages groundbreaking computer vision technology to empower blind and low-vision individuals. Navigate your surroundings with newfound confidence as "Third Eye" interprets the world in real-time.
            """)

        st.header("🌟 Features")
        st.markdown("""
        - **Object Recognition:** Instantly identify obstacles, signage, and objects.
        - **Text-to-Speech:** Hear descriptions of your surroundings, making navigation intuitive.
        - **Navigation Support:** Get real-time assistance to move around safely and efficiently.
        """)

        st.header("💡 How It Works")
        st.write("""
        By analyzing live video feeds, "Third Eye" detects and vocalizes the presence of obstacles and signage, converting visual information into audible guidance. This real-time support system is designed to promote greater inclusivity and accessibility, enhancing the daily lives of visually impaired individuals.
        """)

        # Interactive demo or more information about the technology could go here

        st.header("🌐 Join Our Community")
        st.write("""
        Become a part of the "Third Eye" community and contribute to a world where technology bridges the gap towards a more inclusive society. Share your experiences, suggest improvements, and help us make "Third Eye" better for everyone.
        """)

    if selected == "CONTACT":
        # Add a Contact Us section
        st.header("📬 Contact Us")
        st.write(
            "We'd love to hear from you! Whether you have a question, feedback, or just want to say hello, please don't hesitate to reach out.")

        with st.form("contact_form"):
            name = st.text_input("Name", placeholder="Your Name")
            email = st.text_input("Email", placeholder="Your Email Address")
            message = st.text_area("Message", placeholder="Your Message Here")

            submit_button = st.form_submit_button("Send Message")

            if submit_button:
                # Here you would include the logic to handle the form data, such as sending an email
                # This is a placeholder to simulate form submission
                st.success(f"Thank you, {name}, for reaching out! We'll get back to you soon.")

# Page routing
if "current_page" not in st.session_state:
    st.session_state['current_page'] = "welcome"
if st.session_state['current_page'] == "welcome":
    welcome_page()
    # audio_welc()
if st.session_state['current_page'] == "data_science":
    data_science_page()
elif st.session_state['current_page'] == "app":
    app_page()





# image  references
# <a href="https://www.flaticon.com/free-icons/password" title="password icons">Password icons created by Pixel perfect - Flaticon</a>