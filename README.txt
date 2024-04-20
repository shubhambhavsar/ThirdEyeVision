# ThirdEye Application

Welcome to the ThirdEye application repository. Follow the instructions below to set up and run the application on your local machine.

Note: Please note that the application's formatting is optimized for the Streamlit cloud, so the font visibility or the background theme may differ when run locally.

Application URL: https://thirdeyevision.streamlit.app/

## Prerequisites

- Git (for cloning the repository)
- Python 3.x

## Installation Steps

Step 1. 

Method 1: 
- You can download the zip file of the project and extract it to your desired location.
- Must download three audio dependencies ffmpeg.exe, ffplay.exe, and ffprob.exe from the following URL:
https://azureloyalistcollege-my.sharepoint.com/personal/shubhamjitendraku_loyalistcollege_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshubhamjitendraku%5Floyalistcollege%5Fcom%2FDocuments%2FThird%5FEye%5FAudio%5FDependencies&ga=1


OR

Method 2: **Clone the Repository**
- You can clone the repository using the following command:
- git clone https://github.com/shubhambhavsar/Streamlit_app.git


Step 2. 

Install Required Libraries Depending on your operating system, use one of the following methods:

For Windows:
- pip install -r requirements.txt

For Linux:Update your package list and install the necessary packages:
- sudo apt-get update
- sudo apt-get install -y $(cat packages.txt)

Step 3. To run the Application, navigate to the project directory in your command prompt or terminal, and execute the following command:
- python -m streamlit run Python_Scripts\App.py

This will start the Streamlit server and the application should be accessible via your web browser at the address indicated in the terminal (typically localhost:8501).


Enjoy using ThirdEye!