# ThirdEyeVision 👁️

An AI-powered assistive technology application that empowers visually impaired individuals to navigate their surroundings with confidence using real-time computer vision and text-to-speech technology.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://thirdeyevision.streamlit.app/)

## 🌟 Overview

ThirdEyeVision leverages cutting-edge computer vision and deep learning to provide real-time environmental awareness for blind and low-vision individuals. By analyzing live video feeds, the application detects obstacles, reads street signs, and interprets pedestrian signals, converting visual information into immediate audible guidance.

## 🎯 Mission

To promote greater inclusivity and accessibility by transforming visual information into audio feedback, enabling visually impaired users to navigate streets safely, avoid obstacles, and make informed decisions about their surroundings.

## ✨ Key Features

### 🚨 Alert System
- **Real-time Object Detection**: Identifies approaching vehicles (cars, bikes, trucks) and obstacles
- **Proximity Warnings**: Announces detected objects with distance estimation
- **Multi-object Tracking**: Monitors multiple objects simultaneously
- **Audio Alerts**: Immediate voice notifications for potential hazards

### 🛣️ Street Name Recognition
- **Sign Detection**: Automatically identifies street name signs
- **Text Extraction**: Reads text from signs regardless of font, size, or lighting conditions
- **Voice Announcement**: Clearly announces street names to aid navigation
- **Multi-language Support**: Recognizes text in various languages

### 🚦 Pedestrian Signal Detection
- **Traffic Light Recognition**: Identifies pedestrian crosswalk signals
- **Signal Status**: Distinguishes between walk and don't walk signals
- **Safe Crossing Guidance**: Provides real-time crossing advisories
- **Color & Symbol Detection**: Analyzes both color and symbol information

## 🛠️ Technology Stack

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **YOLO (You Only Look Once)** | Real-time object detection for vehicles, obstacles, and traffic signals | Latest |
| **OpenCV** | Video processing, frame capture, and image manipulation | Latest |
| **EasyOCR** | Optical character recognition for street name extraction | Latest |
| **gTTS (Google Text-to-Speech)** | Convert detected information into natural speech | Latest |
| **Streamlit** | Web application framework and user interface | Latest |
| **Python** | Primary programming language | 3.x |

### Additional Libraries
- NumPy - Numerical computations and array operations
- PIL/Pillow - Image processing
- Speech Recognition - Voice command input (optional)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Live Video Feed                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenCV (Frame Capture & Processing)             │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    YOLO     │  │   EasyOCR   │  │    YOLO     │
│   Vehicle   │  │   Street    │  │  Pedestrian │
│  Detection  │  │    Name     │  │   Signal    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │    gTTS Engine         │
           │  (Text-to-Speech)      │
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │    Audio Feedback      │
           │   to User              │
           └────────────────────────┘
```

## 📊 Dataset & Data Annotation

### Custom Dataset Collection
- **Local Street Names**: Extensive collection of street sign images from various locations
- **Pedestrian Signals**: Comprehensive dataset of traffic light signals in different conditions
- **Environmental Variations**: Images captured across various:
  - Lighting conditions (day, night, dusk, dawn)
  - Weather conditions (sunny, cloudy, rainy)
  - Angles and perspectives
  - Distance variations

### Data Annotation Process
- Manual labeling of street name regions
- Bounding box annotations for vehicle and object detection
- Signal state classification (walk/don't walk)
- Quality control and validation

## 🚀 Getting Started

### Prerequisites

- Git (for cloning the repository)
- Python 3.x (3.7 or higher recommended)
- Webcam or video input device
- Speakers or headphones for audio output

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shubhambhavsar/ThirdEyeVision.git
   cd ThirdEyeVision
   ```

2. **Install Dependencies**
   
   For Windows:
   ```bash
   pip install -r requirements.txt
   ```
   
   For macOS/Linux:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python -m streamlit run Python_Scripts\App.py
   ```
   
   For macOS/Linux:
   ```bash
   python3 -m streamlit run Python_Scripts/App.py
   ```

4. **Access the Application**
   - The Streamlit server will start automatically
   - Open your web browser and navigate to: `http://localhost:8501`
   - Or use the URL displayed in your terminal

### Live Demo

Access the deployed application: [https://thirdeyevision.streamlit.app/](https://thirdeyevision.streamlit.app/)

**Note**: The application's formatting is optimized for Streamlit Cloud. Font visibility and background themes may differ when running locally.

## 💻 Usage

### Starting the Application

1. **Launch**: Run the application using the command above
2. **Allow Permissions**: Grant camera and microphone access when prompted
3. **Select Detection Mode**: Choose from Alert, Street Name, or Pedestrian systems

### Interaction Methods

#### Voice Commands
Speak the following keywords to activate detection modes:
- "Alert" - Activate vehicle/object detection
- "Street Name" - Enable street sign recognition
- "Pedestrian" or "Crosswalk" - Start traffic signal detection
- "Stop" - Pause detection

#### Manual Controls
- Click **Get Started** button for full feature access
- Adjust detection parameters:
  - Confidence threshold
  - Detection frequency
  - Audio volume
  - Processing speed

### Best Practices

- **Lighting**: Ensure adequate lighting for optimal detection
- **Camera Position**: Hold device steady and point toward the area of interest
- **Distance**: Keep objects within 2-30 feet for best results
- **Audio**: Use headphones in noisy environments for clearer feedback

## 🎓 How It Works

### Detection Pipeline

1. **Video Capture**: OpenCV continuously captures frames from the video source
2. **Frame Processing**: Each frame is preprocessed (resizing, normalization)
3. **Detection**: 
   - YOLO model identifies objects and their locations
   - EasyOCR extracts text from detected sign regions
   - Custom classifier determines pedestrian signal status
4. **Information Synthesis**: Detected information is formatted for speech
5. **Audio Generation**: gTTS converts text to natural-sounding speech
6. **User Feedback**: Audio is played through speakers/headphones

### YOLO Object Detection

YOLO (You Only Look Once) uses a single neural network to:
- Divide images into a grid
- Predict bounding boxes and class probabilities for each grid cell
- Achieve real-time detection speeds (30+ FPS)
- Detect multiple objects simultaneously

### EasyOCR Text Recognition

EasyOCR employs deep learning for text detection and recognition:
- Text detection using CRAFT (Character Region Awareness)
- Text recognition using CRNN (Convolutional Recurrent Neural Network)
- Supports 80+ languages
- Handles rotated and curved text

## 🌍 Use Cases

- **Daily Navigation**: Help users identify their location and navigate streets
- **Safe Street Crossing**: Provide real-time traffic signal information
- **Obstacle Avoidance**: Alert users to approaching vehicles and obstacles
- **Independence**: Enable autonomous movement in urban environments
- **Confidence Building**: Increase user confidence through environmental awareness

## 🤝 Contributing

We welcome contributions from the community!

## 🐛 Known Issues & Limitations

- Performance may vary based on hardware capabilities
- OCR accuracy depends on sign clarity and lighting conditions
- Detection range limited by camera quality and resolution
- Audio latency may occur on slower systems
- Requires stable internet connection for gTTS (cloud-based)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐ on GitHub!

---

**Made with ❤️ for accessibility and inclusion**

*Empowering visually impaired individuals through technology*
