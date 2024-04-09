from ultralytics import YOLO
import pandas as pd
import cv2
# import pyttsx3
import base64
from gtts import gTTS
from io import BytesIO
import streamlit as st
import time


def tts(name):
    text = f'Alert, {name} approaching!'

    tts = gTTS(text, lang='en')   # Create a text-to-speech object with the given text and language set to English
    
    audio_bytes_io = BytesIO() # Create a BytesIO object to hold the audio data
    
    tts.write_to_fp(audio_bytes_io)  # Write the audio data to the BytesIO object
    
    audio_bytes_io.seek(0)   #Seek to the beginning of the BytesIO stream
    
    audio_base64 = base64.b64encode(audio_bytes_io.read()).decode('utf-8')   # Encode the audio data in base64 to embed in HTML
    
    audio_html = f'<audio autoplay controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'  # Create the HTML code for an audio player with the encoded audio data
    
    st.components.v1.html(audio_html, height=50) # Use Streamlit's HTML component to display the audio player in the app

model = YOLO('yolov8n.pt')
prop_val = 0.0025
perc = 15

#classids
#1=bicycle
#2=car
#3=motorcycle
#5=bus
#7=truck

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



# # Initialize the text-to-speech engine
# def init_engine():
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 130)
#     engine.setProperty('volume', 0.5)
#     return engine




# #tts function
# def tts(name,engine):
#     engine.say(f'Alert, {name} approaching!')
#     engine.runAndWait()

def calculate_distance(width_in_frame, known_width, focal_length):
    distance = (focal_length * known_width) / width_in_frame
    return distance

#preprocess frame
def preprocess(frame):
    height,width=frame.shape[:2]
    if height<width:

        crop_width=int(width*0.5)
        crop_height=int(height*1)
        start_x=(width-crop_width)//2
        start_y=(height-crop_height)//2
        c_frame=frame[start_y:start_y+crop_height,start_x:start_x+crop_width]
    else:
        c_frame=frame
    return c_frame

#penalty function for missed detect
def penalty(list_id,car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance):
    if len(list_id)==0:
        cycle=max(0,cycle-5)
        area_cycle=area_cycle[5:]
        cycle_distance[5:]
        car=max(0,car-5)
        area_car=area_car[5:]
        car_distance[5:]
        bike=max(0,bike-5)
        area_bike=area_bike[5:]
        bike_distance[5:]
        bus=max(0,bus-5)
        area_bus=area_bus[5:]
        bus_distance[5:]  
        truck=max(0,truck-5)
        area_truck=area_truck[5:]
        truck_distance[5:]
    else:
        if 1.0 not in list_id:
            cycle=max(0,cycle-5)
            area_cycle=area_cycle[5:]
            cycle_distance[5:]
        if 2.0 not in list_id:
            car=max(0,car-5)
            area_car=area_car[5:]
            car_distance[5:]
        if 3.0 not in list_id:
            bike=max(0,bike-5)
            area_bike=area_bike[5:]
            bike_distance[5:]
        if 5.0 not in list_id:
            bus=max(0,bus-5)
            area_bus=area_bus[5:]
            bus_distance[5:]             
        if 7.0 not in list_id:
            truck=max(0,truck-5)
            area_truck=area_truck[5:]
            truck_distance[5:]
    return car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance
#checkpoint for tts
def checkpoint(fps2,car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance):
    if car>fps2:
        first10_Car=area_car[:10]
        avg_first_car=sum(first10_Car)/len(first10_Car)
        last10_car=area_car[-10:]
        avg_last_car=sum(last10_car)/len(last10_car)
        x_car=avg_first_car
        y_car=avg_last_car
        percentage_increase_car = ((y_car - x_car) / x_car) * 100
        first10distance_car=car_distance[:10]
        avg_first_car_distance=sum(first10distance_car)/len(first10distance_car)
        last10distance_car=car_distance[-10:]
        avg_last_car_distance=sum(last10distance_car)/len(last10distance_car)
        if percentage_increase_car>perc and avg_first_car_distance>avg_last_car_distance:
            
            tts('car')
            car=0
            area_car.clear()
            
            car_distance.clear()
        else:
            car=0
            area_car.clear()
            car_distance.clear()
            
    if bus>fps2:
        first10_bus=area_bus[:10]
        avg_first_bus=sum(first10_bus)/len(first10_bus)
        last10_bus=area_bus[-10:]
        avg_last_bus=sum(last10_bus)/len(last10_bus)
        x_bus=avg_first_bus
        y_bus=avg_last_bus
        percentage_increase_bus = ((y_bus - x_bus) / x_bus) * 100
        first10distance_bus=bus_distance[:10]
        avg_first_bus_distance=sum(first10distance_bus)/len(first10distance_bus)
        last10distance_bus=bus_distance[-10:]
        avg_last_bus_distance=sum(last10distance_bus)/len(last10distance_bus)
        if percentage_increase_bus>perc and avg_first_bus_distance>avg_last_bus_distance:
            tts('bus')
            bus=0
            area_bus.clear()
            bus_distance.clear()
        else:
            bus=0
            area_bus.clear()
            bus_distance.clear()
            
    if truck>fps2:
        first10_truck=area_truck[:10]
        avg_first_truck=sum(first10_truck)/len(first10_truck)
        last10_truck=area_truck[-10:]
        avg_last_truck=sum(last10_truck)/len(last10_truck)
        x_truck=avg_first_truck
        y_truck=avg_last_truck
        percentage_increase_truck = ((y_truck - x_truck) / x_truck) * 100
        first10distance_truck=truck_distance[:10]
        avg_first_truck_distance=sum(first10distance_truck)/len(first10distance_truck)
        last10distance_truck=truck_distance[-10:]
        avg_last_truck_distance=sum(last10distance_truck)/len(last10distance_truck)
        if percentage_increase_truck>perc and avg_first_truck_distance>avg_last_truck_distance:
            tts('truck')
            truck=0
            area_truck.clear()
            truck_distance.clear()
        else:
            truck=0
            area_truck.clear()
            truck_distance.clear()
    if cycle>fps2:
        first10_cycle=area_cycle[:10]
        avg_first_cycle=sum(first10_cycle)/len(first10_cycle)
        last10_cycle=area_cycle[-10:]
        avg_last_cycle=sum(last10_cycle)/len(last10_cycle)
        x_cycle=avg_first_cycle
        y_cycle=avg_last_cycle
        percentage_increase_cycle = ((y_cycle - x_cycle) / x_cycle) * 100
        first10distance_cycle=cycle_distance[:10]
        avg_first_cycle_distance=sum(first10distance_cycle)/len(first10distance_cycle)
        last10distance_cycle=cycle_distance[-10:]
        avg_last_cycle_distance=sum(last10distance_cycle)/len(last10distance_cycle)
        print(cycle_distance)
        if percentage_increase_cycle>perc and avg_first_cycle_distance>avg_last_cycle_distance:
            tts('cycle')
            cycle=0
            area_cycle.clear()
            cycle_distance.clear()
        else:
            cycle=0
            area_cycle.clear()
            cycle_distance.clear()
    if bike>fps2:
        first10_bike=area_bike[:10]
        avg_first_bike=sum(first10_bike)/len(first10_bike)
        last10_bike=area_bike[-10:]
        avg_last_bike=sum(last10_bike)/len(last10_bike)
        x_bike=avg_first_bike
        y_bike=avg_last_bike
        percentage_increase_bike = ((y_bike - x_bike) / x_bike) * 100
        first10distance_bike=bike_distance[:10]
        avg_first_bike_distance=sum(first10distance_bike)/len(first10distance_bike)
        last10distance_bike=bike_distance[-10:]
        avg_last_bike_distance=sum(last10distance_bike)/len(last10distance_bike)
        if percentage_increase_bike>perc and avg_first_bike_distance>avg_last_bike_distance:
            tts('bike')
            bike=0
            area_bike.clear()
            bike_distance.clear()
        else:
            bike=0
            area_bike.clear()
            bike_distance.clear()
    return car, bus, truck, cycle, bike, area_car, area_bus, area_truck, area_bike, area_cycle, car_distance,bus_distance,truck_distance,cycle_distance,bike_distance

# Process each frame
def process_frame(cropped_frame,user_conf_value,user_class_id,car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance,margin):
    results = model(cropped_frame,classes=user_class_id,conf=user_conf_value)
    a=results[0].boxes.data
    # Plotting and displaying the frame with detections (add this at the end of your function)
    res_plotted = results[0].plot()
    a=a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list_id=[]
    for value in px.iloc[:, -1]: 
        list_id.append(value)
        car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance=penalty(list_id,car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance)
    for index, row in px.iterrows():
        d = int(row[5])
        c = class_list[d]
        height_crop, width_crop = cropped_frame.shape[:2]
        area_frame=height_crop*width_crop
        left_margin=width_crop*margin
        right_margin=width_crop*(1-margin)
        if 'car' in c:
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            area=(x2-x1)*(y2-y1)
            prop=area/area_frame
            if prop>prop_val:
                centre_car=x1+x2/2
                if left_margin<centre_car<right_margin:
                    car=car+1
                    area_car.append(area)
                    width=x2-x1
                    dist=calculate_distance(width,known_width,focal_length)
                    car_distance.append(dist)
            else:
                car=max(0,car-5)
                area_car=area_car[5:]
                car_distance[5:]

                    
        elif 'bicycle' in c:
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            area=(x2-x1)*(y2-y1)
            prop=area/area_frame
            if prop>prop_val:
                cycle=cycle+1
                area_cycle.append(area)
                width=x2-x1
                dist=calculate_distance(width,known_width,focal_length)
                cycle_distance.append(dist)
            else:
                cycle=max(0,cycle-5)
                area_cycle=area_cycle[5:]
                cycle_distance[5:]              
                
                

        elif 'truck' in c:
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            area=(x2-x1)*(y2-y1)
            prop=area/area_frame
            if prop>prop_val:
                centre_truck=x1+x2/2
                if left_margin<centre_truck<right_margin:
                    truck=truck+1
                    area_truck.append(truck)
                    width=x2-x1
                    dist=calculate_distance(width,known_width,focal_length)
                    truck_distance.append(dist)
                else:
                    truck=max(0,truck-5)
                    area_truck=area_truck[5:]
                    truck_distance[5:]
                           
                    
        elif 'bus' in c:
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            area=(x2-x1)*(y2-y1)         
            prop=area/area_frame
            if prop>prop_val:
                centre_bus=x1+x2/2
                if left_margin<centre_bus<right_margin:
                    bus=bus+1
                    area_bus.append(area)
                    width=x2-x1
                    dist=calculate_distance(width,known_width,focal_length)
                    bus_distance.append(dist)
                else:
                    bus=max(0,bus-5)
                    area_bus=area_bus[5:]
                    bus_distance[5:]
            
                    
        elif 'motorcycle' in c:
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            area=(x2-x1)*(y2-y1)       
            prop=area/area_frame
            if prop>prop_val:
                bike=bike+1
                area_bike.append(area)
                width=x2-x1
                dist=calculate_distance(width,known_width,focal_length)
                bike_distance.append(dist)
            else:
                bike=max(0,bike-5)
                area_bike=area_bike[5:]
                bike_distance[5:]               
                
        else:
            pass
    return car, bus, truck, cycle, bike, area_car, area_bus, area_truck, area_bike, area_cycle, res_plotted,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance
# Main function to run the program
def main_func_alert(cap, user_conf_value, margin, user_class_id, user_fps_value, vid_type):
    # engine = init_engine()
    # cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps2 = fps * user_fps_value


    # Start timing for total processing
    total_start_time = time.time()  
    
    # initialize counters
    global car
    global bus
    global truck
    global cycle
    global bike
    global area_cycle
    global area_car
    global area_bus
    global area_bike
    global area_truck
    global car_distance
    global bus_distance
    global truck_distance
    global bike_distance
    global cycle_distance
    global known_width
    global focal_length

    car=0
    bus=0
    truck=0
    cycle=0
    bike=0
    area_cycle=[]
    area_car=[]
    area_bus=[]
    area_bike=[]
    area_truck=[]
    car_distance=[]
    bus_distance=[]
    truck_distance=[]
    bike_distance=[]
    cycle_distance=[]
    known_width= 180.0
    focal_length = 600 
    
    st_frame = st.empty()
    prev_time = 0
    curr_time = 0
    frame_counter_slot = st.empty()
    model_counter_slot=st.empty()
    margin=float(margin)
    while True:
        car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance=checkpoint(fps2,car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance)
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing for model inference
         
        start_time=time.time()
        cropped_frame=preprocess(frame)
        car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,res_plotted,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance =process_frame(cropped_frame,user_conf_value,user_class_id,car,bus,truck,cycle,bike,area_car,area_bus,area_truck,area_bike,area_cycle,car_distance,bus_distance,truck_distance,cycle_distance,bike_distance,margin)
        end_time=time.time()
        

        # Measure model inference time
        model_inference_time = end_time - start_time  
        if vid_type == 'Show-Video':
            st_frame.image(res_plotted,
                   caption='Detected Video',
                   use_column_width=True,
                   channels="BGR")        
            
        else:
            st_frame = st.empty()
        # Display the performance metrics
         # Measure total processing time
        curr_time = time.time()
        model_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        model_counter_slot.write(f"Model Inference Time: {model_inference_time*1000:.2f}ms")
        frame_counter_slot.write(f'Model Fps: {model_fps:.2f}')


    cap.release()
    cv2.destroyAllWindows()


   
