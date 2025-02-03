
# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
import numpy as np
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from io import BytesIO
from keras.models import load_model
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from ultralytics import YOLO
import cv2
import time
from flask import jsonify
from datetime import datetime
from datetime import date
from openpyxl import load_workbook
from collections import defaultdict
import os
import pandas as pd
import pygetwindow as gw
from deepface import DeepFace
import shutil
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from PIL import Image as PILImage
from datetime import datetime
#from io import Bytes
#from sqlDB import mydb
import sqlite3
import threading
import subprocess
import config
import base64
i=0
correct_face_count=0
incorrect_face_count=0

import boto3
import uuid 
import schedule





# AWS S3 and CloudFront settings
s3_bucket_name = 'ipcamera'
cloudfront_domain = 'https://d1hxsynrm0sbko.cloudfront.net/'
s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID_1,    # your account id
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY_1,    # your secret access key
    region_name=config.AWS_REGION_1
)
 
# S3 connection settings
s3_client1 = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key= config.AWS_SECRET_ACCESS_KEY,
   # your secret access key
    region_name=config.AWS_REGION_1
)
s3_bucket_name1 = 'xaitestimages'
s3_bucket_name2= 'yachna'
  # replace with your actual S3 bucket name
 
#conn1= sqlite3.connect("Members.db", check_same_thread=False)
#conn2= sqlite3.connect("EntryLog.db", check_same_thread=False)

# Create a cursor
#curM1= conn1.cursor()
#curE2= conn2.cursor()
# Paths to store video files
video_output_path = 'static/videos'

def upload_to_s3(local_path, s3_path):
    try:
        s3_client1.upload_file(local_path, s3_bucket_name1, s3_path)
        print(f"Uploaded {local_path} to s3://{s3_bucket_name1}/{s3_path}")
        print("dir: ", local_path)
        #os.remove(local_path)    
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")








# Keep track of processed videos
processed_videos = set() 


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return loss

def datewise_filter(FILE_PATH, start_date, end_date):
    df=pd.read_excel(FILE_PATH)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
    df = df.dropna(subset=['Date']) 
    start_date = pd.to_datetime(start_date) 
    end_date = pd.to_datetime(end_date)
    
 
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    unique_values = filtered_df['Name'].unique()
    #filtered_df.to_excel(output_path, index=False) 
    print(unique_values)
    print(len(unique_values))
    return unique_values, len(unique_values) 

def saveunknown(frame,path):
    print("fframe ", frame)
    imgs= os.listdir(path)
    #last_img=imgs[len(imgs)-1]
    print("save unknown images ", imgs)

    uid = uuid.uuid4()
    uid = str(uid)

    filename= f'{uid}'

    out3 = DeepFace.find(frame, path, enforce_detection=False)
    if isinstance(out3, list) and len(out3) > 0:
        out3_df = pd.DataFrame(out3[0])
    else:
        out3_df = pd.DataFrame(out3)

    print("out3_df", out3_df)
    if out3_df.empty:
        print("No matching images found.")
        cv2.imwrite(os.path.join(path, filename + '.jpg'), frame)
        print("saving unknown....")
        img_path= path+"/"+ filename + ".jpg"
        print("path=", img_path)
        #save_image_and_append_data(img_path, "Unknown_DataBase.xlsx", filename)
        writeUDB(filename,img_path)
        
        #append_to_excel("./backend/test1//Unknown_DataBase.xlsx", filename, frame)

        #source_file_path = filename
        #destination_folder_path = "./Unknown"
        #append_jpg_to_folder(source_file_path, destination_folder_path)
        #i=i+1
    else:
        # Display the DataFrame
        print(out3_df)
   


def extract_face(filename, required_size=(160, 160)):
 # load image from file
   #image = Image.open(filename)
   # convert to RGB, if needed
   image = filename.convert('RGB')
   # convert to array
   pixels = asarray(image)
   # create the detector, using default weights
   detector = MTCNN()
   # detect faces in the image
   results = detector.detect_faces(pixels)
   # extract the bounding box from the first face
   x1, y1, width, height = results[0]['box']
   # bug fix
   x1, y1 = abs(x1), abs(y1)
   x2, y2 = x1 + width, y1 + height
   # extract the face
   face = pixels[y1:y2, x1:x2]
   # resize pixels to the model size
   image = Image.fromarray(face)
   image = image.resize(required_size)
   face_array = asarray(image)
   return face_array

# get the face embedding for one face
def get_embedding(model, face_pixels):
    #face_pixels = cv2.resize(face_pixels, (160, 160))
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def writeDB(name, file_path):
    now = datetime.now()
    print("new path===", file_path)
    #date_str = now.strftime('%Y-%m-%d')
    #time_str = now.strftime('%H:%M:%S')
    datetime_str = now.strftime('%Y-%m-%d %H:%M:%S')
    datetime_str1 = datetime_str.replace(' ', '_')
    
    try:
            # Connect to the SQLite database with a timeout
        conn = sqlite3.connect('members.db', timeout=30)  # Wait for up to 30 seconds
        conn.execute('PRAGMA journal_mode=WAL;')  # Enable Write-Ahead Logging for better concurrency
        cur = conn.cursor()

        cur.execute("SELECT uid FROM Members WHERE Name = ?", (name,))
        uid = cur.fetchone()

        # Commit the transaction
        conn.commit()
        print("Committed successfully")
        conn.close()

        file_name = f"{uid[0]}_{datetime_str1}.jpg"
        s3_path = f"{'knownimages'}/{file_name}"
        print("s3 ", s3_path)

        upload_to_s3(file_path,s3_path)
        
        public_url = f"https://{s3_bucket_name1}.s3.{s3_client.meta.region_name}.amazonaws.com/{s3_path}"
        print("KNOwn s3 URL ", public_url)

        try:
            conn1=sqlite3.connect("EntryLog.db",timeout=60)
            conn1.execute('PRAGMA journal_mode=WAL;')  # Enable Write-Ahead Logging for better concurrency
            cur1 = conn1.cursor()
            cur1.execute('INSERT INTO KNOWNDB (id, Timestamp_entry, Image_Tag) VALUES(?, ?, ?)', 
                     (uid[0], datetime_str, public_url))
            conn1.commit() 
            #conn1.close()
        except sqlite3.OperationalError as e:
            #print(f"Database is locked: {e}")
            return jsonify({'error': ' Entry Database is locked, try again later.'}), 500

    except sqlite3.OperationalError as e:
            print(f"Database is locked: {e}")
            return jsonify({'error': 'Database is locked, try again later.'}), 500
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return jsonify({'error': 'SQLite error occurred'}), 500
    finally:
            # Ensure that the connection is closed
            if conn and conn1:
                conn.close()
                conn1.close()


    
    #with open(file_path,"rb") as f:
    #    binImage1=f.read()
        
    #print("bin............=", binImage1)
    #mycursor=mydb.cursor() 
    #sql="INSERT INTO LIVEVIDEODATA(MEM_NAME, DATE_TIME, KNOWN_IMAGE ) VALUES(%s, %s, %s)"
    #val=(name, datetime_str, binImage1)
    #mycursor.execute(sql, val)
    #mydb.commit()
    #--uid_result, membership = cur1.execute("SELECT uid, Membership FROM Members WHERE Name = ?", (name,)).fetchone()
        
    #--print("Unique ID:", uid_result)
    #--print("Mem id", membership)

    #--file_name = f"{uid_result}.jpg"
    #--image=cv2.imread(file_path)
    
    #--cv2.imwrite(file_name, file_path)


    # Using cur3 for INSERT operation
    
    #print("......committed successfully")


def writeUDB(name,roi):
    now = datetime.now()
    #date_str = now.strftime('%Y-%m-%d')
    #time_str = now.strftime('%H:%M:%S')
    datetime_str = now.strftime('%Y-%m-%d %H:%M:%S')
    datetime_str1 = datetime_str.replace(' ', '_')

    print("name= ", name)
    print("roi= ", roi)
    with open(roi,"rb") as f:
        binImage1=f.read()
        print("bin1=", binImage1)
    print("UDBBBB")
    #binImage=image_to_binary(roi)
    #type("type of", type(binImage))
    # mycursor=mydb.cursor()

    #sql="INSERT INTO UNKNOWN_DB(UNKNOWN_NAME, DATE_TIME, UNKNOWN_IMAGE) VALUES(%s, %s, %s)"
    #val=(name, datetime_str,binImage1)
    #mycursor.execute(sql, val)
    #mydb.commit()
    print("next.....")
    try:
        conn2=sqlite3.connect("EntryLog.db",timeout=60)
        conn2.execute('PRAGMA journal_mode=WAL;')  # Enable Write-Ahead Logging for better concurrency
        cur2 = conn2.cursor() 
        filename=f'{name}_{datetime_str}'
        s3_path = f"{'unknownimages'}/{name}_{datetime_str1}.jpg"
        upload_to_s3(roi,s3_path)
        public_url = f"https://{s3_bucket_name1}.s3.{s3_client.meta.region_name}.amazonaws.com/{s3_path}"
        print("unknwon s3 : ", public_url)
        #os.remove(roi)
        cur2.execute('INSERT INTO UNKNOWNDB (id, NAME, Timestamp_entry, Image_Tag) VALUES(?, ?, ?,?)', (name, filename ,datetime_str, public_url))
        conn2.commit()
        cur2.close()
        conn2.close()
        print("committed successfully")
        
    except sqlite3.OperationalError as e:
            print(f"Database is locked: {e}")
            return jsonify({'error': 'Database is locked, try again later.'}), 500
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return jsonify({'error': 'SQLite error occurred'}), 500
    finally:
            # Ensure that the connection is closed
            if conn2:
                conn2.close()




def crop_image(image_path, crop_size):
    # Open an image file
    with PILImage.open(image_path) as img:
        # Crop the image to the given size
        cropped_img = img.crop((0, 0, crop_size[0], crop_size[1]))
        return cropped_img
 
 

def identify_face(embedding, dict):
    
    min_distance = 100
    try:
        for (dict_embedding,name) in dict.items():
            # Compute euclidean distance between current embedding and the embeddings from the 'embeddings' folder
            distance = np.linalg.norm(embedding - dict_embedding)
            print("Distance", distance)
            if distance < min_distance:
                min_distance = distance
                identity = name
          # Compute confidence
        confidence = round(max(0, 1 - min_distance), 2)
        
        if min_distance <0.3:
            # remove 'embeddings/' from identity
            #identity = identity[:11]
            result = "It's " + str(identity) + ", the distance is " + str(min_distance)

            return str(identity), str(confidence)

        else:
            result = "Not in the database, the distance is " + str(min_distance)

            return "unknown",str(confidence)

    except Exception as e:
        print(str(e))

        return str(e), str(confidence)




def test(cam):

    time.sleep(1)
    #data1 = load('faces-dataset2.npz')
    yolo_model = YOLO('yolov8n-face (3) (2).pt') 
    data = load('Embeddings/faces-EmbeddingSatPooMoKejNew4.npz')
    trainX, trainy = data['arr_0'], data['arr_1']

    a = np.array(trainX)
    b = np.array(trainy)
    dict = {}
    for A, B in zip(a, b):
        dict[tuple(A)] = B

    
    model = load_model('./models/facenet_model1.h5', custom_objects={'triplet_loss': triplet_loss})
    print('Loaded Model')
    global correct_face_count


    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    #print("training x = " , trainX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)


    trainy = out_encoder.transform(trainy)
    
    

    today = date.today()
    today= today.strftime("%Y%m%d")
    print("Today's date:", today)
   
    daily_data = {today: set()}

    # Define the number of frames to skip
    skip_frames = 0  # Process every 10th frame
    #schedule_deletion()
    # Frame counter
    frame_count = 0
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Start the timer to limit the video recording to 60 seconds (1 minute)
 





    while True:
        
        ret,frame= cam.read()
        
        print("started camera")

        frame_count+=1

        if frame_count % (skip_frames +1) ==0 and frame is not None: 
            
            
            rgb_frame = frame[:, :, ::-1]
               
            #print("rgb ", rgb_frame)
            # Detect faces using YOLOv8
            yolo_results = yolo_model(frame, conf=0.7)

            for result in yolo_results:
                boxes = result.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    label = box.cls
                    confidence = box.conf

                    # If the detected object is a person (label 0), process further
                    if label == 0:
                        print("person detected")
                        face_image = rgb_frame[y1:y2, x1:x2]
                        image = Image.fromarray(face_image)
                        image = image.resize((160,160))
                        face_array = asarray(image)                
                        #cv2.imshow("res", img_resized)
                        emb=get_embedding(model, face_array)
                        #print("emb", emb)
                        samples = expand_dims(emb, axis=0)
                    
                        res, conf= identify_face(emb, dict)
                        print("conf", conf)
                        if res!="unknown":
                            correct_face_count += 1
                            if res not in daily_data[today]:
                                daily_data[today].add(res)
                                directory = './images2/' + today
                                os.makedirs(directory, exist_ok=True)
                              
                                if frame is not None:
                                    roi = frame[y1-50:y2+100, x1-50:x2+100]
                                #roi = frame[y1-50:y2+100,x1-50:x2+100]
                                #cv2.imshow("roi", roi)
                                # Save the image to './saved_photos/YYYYMMDD/res.jpg'
                                    print("rrrroi ",roi)
                                    cv2.imwrite(os.path.join(directory, res + '.jpg'), roi)
                                    file_Path1= directory +"/"+ res + ".jpg" 
                                
                                    print("path= ", file_Path1)
                               
                                    writeDB(res, file_Path1)
                              
                        if res=="unknown":
                            path="./images1"+ "/" + today
                           # shutil.copy("./images1/008_d1f87068.jpg", path)
                            if os.path.exists(path):
                                if frame is not None:
                                    roi=frame[y1-50:y2+100, x1-50:x2+100]
                                    print("fframe", frame)
                                    print("rroi ", roi)
                                    saveunknown(roi, path)
                                
                            else:
                                os.makedirs(path)
                                shutil.copy("./images1/008_d1f87068.jpg", path)
                        
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        #cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # Get the size of the text to position it at the bottom center
                        text_size = cv2.getTextSize(current_time, font, 0.8, 2)[0]
                        text_width = text_size[0]
                    
                        
                        # Position the text at the bottom center of the frame
                        x = (frame.shape[1] - text_width) // 2  # Center horizontally
                        y = frame.shape[0] -15  # Position near the bottom

                        # Put the current time text on the frame
                        cv2.putText(frame, current_time, (x+150, y), font, 0.8, (255, 255, ), 2)



                        cv2.putText(frame, res, (x1 + 46, y2 - 156), font, 0.7, (255, 255, 255), 1)
                        cv2.putText(frame, conf, (x1 - 6, y2 -166), font, 0.7, (0, 0, 255), 1)
                        cv2.imshow("result", frame)
                        windows = gw.getWindowsWithTitle("Camera Feed")
                        if windows:
                            window = windows[0]
                            if window.isMinimized:
                                window.restore()
                            window.activate()

                        # Hit 'q' on the keyboard to quit!
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                      
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        
                            # Convert the frame to bytes and yield it in a multipart format
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                    
                    
                    else:
                        print("else block")
                        font = cv2.FONT_HERSHEY_DUPLEX
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # Get the size of the text to position it at the bottom center
                        text_size = cv2.getTextSize(current_time, font, 0.8, 2)[0]
                        text_width = text_size[0]
                        

                        # Position the text at the bottom center of the frame
                        x = (frame.shape[1] - text_width) // 2  # Center horizontally
                        y = frame.shape[0] + 40  # Position near the bottom

                        # Put the current time text on the frame
                        cv2.putText(frame, current_time, (x, y), font, 0.8, (255, 255, 255), 2)

                        ret, jpeg = cv2.imencode('.jpg', frame)
                        
                            # Convert the frame to bytes and yield it in a multipart format
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    
    cam.release()   
    cv2.destroyAllWindows()           
