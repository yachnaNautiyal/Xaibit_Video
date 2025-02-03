from flask import Flask, request, jsonify, Response, render_template,send_from_directory
from flask_cors import CORS
from flask_compress import Compress
import json
import os
import base64
import pickle
import time
import threading
import subprocess
import traceback
from tensorflow.keras import backend as K 
#import mysql.connector
import sqlite3
#import sqliteDB
import pandas as pd  # Added to handle dates from the requests
from openpyxl import load_workbook
from livePredict_flask import test
import logging
from logging.handlers import RotatingFileHandler
import traceback
import cv2
from datetime import datetime, timedelta
import uuid
import os
import boto3
import requests
from botocore.exceptions import NoCredentialsError
import config

from PIL import Image
from io import BytesIO


app = Flask(__name__, template_folder='templates/dist', static_folder='templates/dist')    
Compress(app)
CORS(app)

DATA_FILE = 'NewRegistration.xlsx'
DATA_FORM_FILE = 'DataFormAnalysis.json'
IMAGE_DIR = 'images3'

# Initialize registration data file
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as file:
        json.dump([], file)

# Initialize data form file
if not os.path.exists(DATA_FORM_FILE):
    with open(DATA_FORM_FILE, 'w') as file:
        json.dump([], file)

# Ensure image directory exists
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


#conn=sqlite3.connect("Members.db", check_same_thread=False)
#conn1= sqlite3.connect("EntryLog.db", check_same_thread=False)
# Create a cursor
#cur = conn.cursor()
#cur1=conn1.cursor()

# AWS S3 bucket configuration
S3_BUCKET_NAME1 = 'ipcamera'

S3_CLOUDFRONT_URL1 = 'https://d34xanpfs3oa8l.cloudfront.net/'  # Replace with your CloudFront URL
 
# S3 connection settings
s3_client1 = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,  # your account id
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,  # your secret access key
    region_name=config.AWS_REGION
)
 

video_output_path = 'static/videos'

def upload_to_s3(local_path, s3_path):
    try:
        s3_client1.upload_file(local_path, S3_BUCKET_NAME1, s3_path)
        print(f"Uploaded {local_path} to s3://{S3_BUCKET_NAME1}/{s3_path}")
        print("dir: ", local_path)
        #os.remove(local_path)    
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")


def capture_video(duration, output_path):
    """Capture video for a specified duration and save it to the given output path."""
    try:
        timestamp = time.strftime('%Y%m%d-%H%M%S')  # Format as yymmddhhmmss
        video_filename = f'cam1_loc1_{timestamp}.mp4'  # Save as cam1-yymmdd:hhmmss
        video_output_full_path = os.path.join(output_path, video_filename)
        compressed_video_filename = f'compressed_{video_filename}'
        compressed_video_full_path = os.path.join(output_path, compressed_video_filename)
 
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
 
        # Capture video
        ffmpeg_command = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', 'rtsp://YachnaCCTV:Dakshit@123@192.168.1.34:554/stream1',  # Replace with your RTSP stream URL
            '-t', str(duration),
            '-r', '24',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-an',
            '-pix_fmt', 'yuv420p',
            '-s', '1280x720',
            '-f', 'mp4',
            video_output_full_path
        ]
        print(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
        process = subprocess.run(ffmpeg_command, stderr=subprocess.PIPE, universal_newlines=True)
        if process.returncode != 0:
            print(f"Error capturing video: {process.stderr}")
            return None
 
        # Compress the video
        compress_command = [
            'ffmpeg',
            '-i', video_output_full_path,
            '-vcodec', 'libx264',
            '-crf', '28',
            compressed_video_full_path
        ]
        print(f"Running FFmpeg compression command: {' '.join(compress_command)}")
        compress_process = subprocess.run(compress_command, stderr=subprocess.PIPE, universal_newlines=True)
        if compress_process.returncode == 0:
            print(f"Compression successful. Removing original file: {video_output_full_path}")
            os.remove(video_output_full_path)  # Remove the original uncompressed video
            return compressed_video_full_path
        else:
            print(f"Compression failed: {compress_process.stderr}")
            return video_output_full_path
 
    except Exception as e:
        print(f"Error capturing video: {e}")
        return None
 

def upload_to_s3video(file_path, bucket_name, s3_path):
    """Upload file to S3 and delete local file after successful upload."""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_path)
        print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_path}")
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")
    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {e}")

def capture_and_upload_long_video():
    """Continuously capture videos, upload them to S3, and delete local file."""
    while True:
        print("Capturing video.")
        video_path = capture_video(duration=900, output_path=video_output_path)  # 15 minutes = 900 seconds
        if video_path:
            print(f"Uploading video {video_path} to S3.")
            s3_path = os.path.basename(video_path)
            upload_to_s3video(video_path, S3_BUCKET_NAME1, s3_path)
        time.sleep(.1)  # Wait before starting the next capture




def log_db_status():
    try:
        conn = sqlite3.connect('members.db')
        cursor = conn.cursor()
        cursor.execute('PRAGMA database_list;')  # Check open databases
        result = cursor.fetchall()
        logging.info(f"Database status: {result}")
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error checking DB status: {e}")


# Function to set up multiple log files with rotation
def setup_logger(name, log_file, level=logging.ERROR, max_bytes=1024*1024, backup_count=5):
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


 
# Function to retrieve and filter S3 videos
def get_s3_videos(start_datetime=None, end_datetime=None):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME1)
        videos = []
        for obj in response.get('Contents', []):
            video_key = obj['Key']
            print(f"Processing file: {video_key}")  # Debug: log the file being processed
 
            # Ensure the video matches the expected naming pattern
            if video_key.startswith("cam1_loc1_") and video_key.endswith(".mp4"):
                # Extract the timestamp part of the filename
                video_timestamp_str = video_key[len("cam1_loc1_"):-len(".mp4")]
                print(f"Extracted timestamp: {video_timestamp_str}")  # Debug: log the extracted timestamp
 
                try:
                    # Convert the extracted timestamp into a datetime object
                    video_datetime = datetime.strptime(video_timestamp_str, '%Y%m%d-%H%M%S')
                    print(f"Parsed datetime: {video_datetime}")  # Debug: log the parsed datetime
 
                    # Apply date filtering
                    if start_datetime and end_datetime:
                        if start_datetime <= video_datetime <= end_datetime:
                            videos.append(f"{S3_CLOUDFRONT_URL1}{video_key}")
                            print(f"Video added: {video_key}")  # Debug: log added video
                    else:
                        videos.append(f"{S3_CLOUDFRONT_URL1}{video_key}")
                        print(f"Video added without filtering: {video_key}")  # Debug: log added video
 
                except ValueError:
                    print(f"Skipping file due to invalid timestamp format: {video_key}")  # Debug: log invalid format
                    continue  # Skip files with invalid timestamp format
 
        return videos
    except NoCredentialsError:
        return {"error": "AWS credentials not found. Ensure your credentials are configured correctly."}
    except Exception as e:
        return {"error": str(e)}

# Set up multiple loggers for different files
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")

log_filename = f'error_{now_str}.log'

# Initialize the error logger with the dynamically generated filename
error_logger = setup_logger('error_logger', log_filename)


#S3 Bucket
# S3 connection settings
s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,  # your account id
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,  # your secret access key
    region_name=config.AWS_REGION
)
s3_bucket_name = 'xaitestimages'  # replace with your actual S3 bucket name
 
# Directories and corresponding S3 subfolders
directories = {
    'images1': 'unknownimages',
    'images2': 'knownimages',
    'images3': 'members'
}
 
 # Function to upload files to S3
def upload_to_s3(local_path, s3_path):
    try:
        s3_client.upload_file(local_path, s3_bucket_name, s3_path)
        #print(f"Uploaded {local_path} to s3://{s3_bucket_name}/{s3_path}")
        #print("dir: ", local_path)
        os.remove(local_path)    
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # or "JPEG" depending on the image type
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_public_image_links(bucket_name):
    response = s3_client.list_objects_v2(Bucket=bucket_name)
 
    # Check if the bucket contains any objects
    if 'Contents' not in response:
        #print("No images found in the bucket.")
        return []
 
    # Construct public URLs
    image_links = []
    for item in response['Contents']:
        file_key = item['Key']

        # Public URL format for S3 objects
        public_url = f"https://{bucket_name}.s3.{s3_client.meta.region_name}.amazonaws.com/{file_key}"
 
        image_links.append(public_url)
 
    return image_links


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory(f'{app.static_folder}/assets', path)
 
@app.route('/<path:path>')
def catch_all(path):
    # Check if the requested file exists in the dist folder
    try:
        return send_from_directory(app.static_folder, path)
    except:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/main-program', methods=['POST'])
def main_program():
        global cam
        #print("mainnnnn")
        cam=cv2.VideoCapture(0)
        test(cam)
          
        return jsonify({'message': 'Live Recording on'})

# Route for MJPEG video stream
@app.route('/video_feed')
def video_feed():
    #cam=cv2.VideoCapture(0)
    return Response(test(cv2.VideoCapture("rtsp://admin:VCVOIC@45.249.168.237:554/H.264/AVC")), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to get S3 videos list
@app.route('/videos', methods=['GET'])
def list_videos():
    # Get the start and end dates and times from the request
    start_date = request.args.get('start_date')
    start_time = request.args.get('start_time', '00:00')  # Default to '00:00' if not provided
    end_date = request.args.get('end_date')
    end_time = request.args.get('end_time', '23:59')  # Default to '23:59' if not provided

    # Correct the logging of parameters
    #print("Start Date:", start_date)
    #print("Start Time:", start_time)
    #print("End Date:", end_date)
    #print("End Time:", end_time)

    try:
        # Check if both start_date and end_date are provided
        if start_date and end_date:
            # Combine the date and time to form datetime objects
            start_datetime = datetime.strptime(f"{start_date} {start_time}", '%Y-%m-%d %H:%M')
            end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M')
        else:
            # If start_date and end_date are not provided, set datetime to None
            start_datetime, end_datetime = None, None
        
        # Fetch S3 videos based on the date-time range
        s3_videos = get_s3_videos(start_datetime, end_datetime)
        print("running..............", s3_videos)
        return jsonify({"s3_videos": s3_videos})

    except Exception as e:
        # In case of an error, return a 500 response with the error message
        return jsonify({"error": str(e)}), 500




@app.route('/new-registration', methods=['POST'])
def new_registration():
    try:
        log_db_status() 
        data = request.json
        #print('Received data:', data)

        id = data.get('id')
        name = data.get('name')
        phone = data.get("phone")
        age = data.get("age")
        email = data.get("email")
        bloodGroup = data.get("bloodGroup")
        height = data.get("height")
        weight = data.get("weight")
        address = data.get("address")
        emergencyContactName = data.get("emergencyContactName")
        emergencyContactNumber = data.get("emergencyContactNumber")
        uid = uuid.uuid4()
        uid = str(uid)
        print("working  name")
        registration_data = [[uid, id, name, phone, email, age, bloodGroup, height, weight, address, emergencyContactName, emergencyContactNumber]]
        
        image_data= data.get('image')
        #print("i1", type(image_data)) 
        image_base64 = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_base64)

        now = datetime.now()
        start_date = str(now.strftime("%Y-%m-%d"))
        end_date = str(now + timedelta(days=365))

        try:
            # Connect to the SQLite database with a timeout
            conn = sqlite3.connect('members.db', timeout=30)  # Wait for up to 30 seconds
            conn.execute('PRAGMA journal_mode=WAL;')  # Enable Write-Ahead Logging for better concurrency
            cur = conn.cursor()

            file_name = f"{uid}_{id}.jpg"
            image = Image.open(BytesIO(image_bytes))
            file_path = os.path.join(IMAGE_DIR, file_name)
            image.save(file_path)
            #print("path:  ", file_path)
            #print("D ", directories)


            local_path=file_path
            s3_path = f"{'members'}/{file_name}"
            upload_to_s3(local_path, s3_path)
           # print("upload successful")
            public_url = f"https://{s3_bucket_name}.s3.{s3_client.meta.region_name}.amazonaws.com/{s3_path}"
            #print("image URL: ", public_url)


            # Perform the insert operation
            cur.execute('INSERT INTO Members (uid, Name , Mobile_Number, Membership, Email, Age, Blood_Group, Height, Weight, Address, Emergency_Contact_Name, Emergency_Contact_Number,Profile_Picture, Membership_start_date, Membership_end_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (uid, name, phone, id, email, age, bloodGroup, height, weight, address, emergencyContactName, emergencyContactNumber,  str(public_url),start_date, end_date))

            # Commit the transaction
            conn.commit()
            #print("Committed successfully")
            conn.close()
        except sqlite3.OperationalError as e:
            #print(f"Database is locked: {e}")
            return jsonify({'error': 'Database is locked, try again later.'}), 500
        except sqlite3.Error as e:
            #print(f"SQLite error: {e}")
            return jsonify({'error': 'SQLite error occurred'}), 500
        finally:
            # Ensure that the connection is closed
            if conn:
                conn.close()

        return jsonify({'message': 'Registration successful', 'data': registration_data}), 200

    except Exception as e:
        #print("Registration failed.")
        error_logger = logging.getLogger(__name__)
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({'error': 'Internal Server Error'}), 500
    

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        data = request.json
        user_id = data.get('id')
        user_name = data.get('name')
        image_data= data.get('image')
        #print("i1", type(image_data)) 
        image_base64 = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_base64)
        
        try:
            # Connect to the SQLite database with a timeout
            conn = sqlite3.connect('members.db', timeout=30)  # Wait for up to 30 seconds
            conn.execute('PRAGMA journal_mode=WAL;')  # Enable Write-Ahead Logging for better concurrency
            cur = conn.cursor()

            # Perform the select operation
            uid_result = cur.execute("SELECT uid FROM Members WHERE Membership = ?", (user_id,)).fetchone()
            # Commit the transaction
            conn.commit()

            #print("selected successfully")

            if uid_result:
                uid = uid_result[0]  # Extract the uid from the tuple
            #print("Unique ID:", uid)
            image = Image.open(BytesIO(image_bytes))
            if not user_id or not user_name:
                return jsonify({'error': 'ID and Name are required.'}), 400


            file_name = f"{uid}_{user_id}.jpg"
            #print("file:  ", file_name)
            # Full path to save the image
            file_path = os.path.join(IMAGE_DIR, file_name)
            image.save(file_path)
            #print("path:  ", file_path)
            #print("D ", directories)


            local_path=file_path
            s3_path = f"{'members'}/{file_name}"
            upload_to_s3(local_path, s3_path)
            #print("upload successful")

            #image_links = get_public_image_links(s3_bucket_name)

            public_url = f"https://{s3_bucket_name}.s3.{s3_client.meta.region_name}.amazonaws.com/{s3_path}"
            #print("image URL: ", public_url)
            os.remove(local_path)
            #print("UUUUUid ", uid)
            cur.execute('UPDATE Members SET  Profile_Picture = ? WHERE uid= ?', (str(public_url), uid))
            conn.commit()
            conn.close()
            #print("Image set in DB")        

        except sqlite3.OperationalError as e:
            #print(f"Database is locked: {e}")
            return jsonify({'error': 'Database is locked, try again later.'}), 500
        except sqlite3.Error as e:
            #print(f"SQLite error: {e}")
            return jsonify({'error': 'SQLite error occurred'}), 500
        finally:
            # Ensure that the connection is closed
            if conn:
                conn.close()

      
    
        '''
        if cv2.imwrite(file_path, image):
            print(f"Image saved successfully as {file_path}")
        else:
         print("Failed to save the imageeeeeee")
        
        
        for idx, image_data in enumerate(images):
            image_base64 = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_base64)
            image_path = os.path.join(user_dir, f'image_{idx + 1}.jpeg')
            simages.append(image_bytes)
            serialized_data = pickle.dumps(simages)
            with open(image_path, 'wb') as image_file:
                image_file.write(image_bytes)
            saved_images.append(image_path)
        '''




        #mycursor = mydb.cursor()
        #sql = "UPDATE Members SET Profile_Picture = %s WHERE Membership = %s AND Name = %s"
        #val = (serialized_data, user_id, user_name)
        #mycursor.execute(sql, val)
        #mydb.commit()
        #cur.execute("UPDATE Members SET Profile_Picture= ? Where Membership= ? AND Name= ?", (images, user_id, user_name))

        return jsonify({'message': 'Images uploaded successfully'}), 200
    except Exception as e:
        #print("upload image ")
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/unknown-data-analysis', methods=['GET'])
def unknown_data_analysis():

    try:
            start_date_str = request.args.get('startDate')
            end_date_str = request.args.get('endDate')
            
            if not start_date_str or not end_date_str:
                return jsonify({'error': 'startDate and endDate are required'}), 400

            conn=sqlite3.connect("EntryLog.db")
            cursor= conn.cursor()
            query = 'SELECT DATE(Timestamp_Entry) AS date, COUNT(DISTINCT id) AS unique_count FROM UNKNOWNDB WHERE DATE(Timestamp_Entry) BETWEEN ? AND ? GROUP BY DATE(Timestamp_Entry)' 
            cursor.execute(query, (start_date_str, end_date_str))
            results = cursor.fetchall()
            #print("unknown data results:  ", results)
            date_unique_count_dict = {str(row[0]): row[1] for row in results}

            return jsonify(date_unique_count_dict), 200
    except Exception as e:
            #print("unknown data..")
            error_logger.exception("Error occurred", exc_info=True)
            traceback_str = traceback.format_exc()
            return jsonify({"error": str(e)}), 500

@app.route('/capture-rtsp', methods=['GET'])
def capture_rtsp_image():
    rtsp_url = "rtsp://admin:VCVOIC@45.249.168.237:554/H.264/AVC"
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        return jsonify({"error": "Could not open RTSP stream"}), 400

    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Could not capture image from RTSP stream"}), 400

    # Convert the frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = base64.b64encode(buffer).decode('utf-8')

    # Release the RTSP stream
    cap.release()

    # Return the image as a base64 string
    return jsonify({"image": img_bytes})
         


@app.route('/data-analysis', methods=['GET'])
def data_analysis():
    try:
        start_date_str = request.args.get('startDate')
        end_date_str = request.args.get('endDate')
        
        if not start_date_str or not end_date_str:
            return jsonify({'error': 'startDate and endDate are required'}), 400
        #print("data1")
        conn=sqlite3.connect("EntryLog.db")
        cursor= conn.cursor()
        query = 'SELECT DATE(Timestamp_Entry) AS date, COUNT(DISTINCT id) AS unique_count FROM KNOWNDB WHERE DATE(Timestamp_Entry) BETWEEN ? AND ? GROUP BY DATE(Timestamp_Entry)' 
        cursor.execute(query, (start_date_str, end_date_str))
        results = cursor.fetchall()
        #print("known data results:  ", results)
        date_unique_count_dict = {str(row[0]): row[1] for row in results}

        return jsonify(date_unique_count_dict), 200
    except Exception as e:
        #print("known data..")
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e)}), 500




@app.route('/unknown-images-by-date', methods=['GET'])
def unknown_images_by_date():
    global mydb
    try:
        date_str = request.args.get('date')
        
        if not date_str:
            raise ValueError("Missing 'date' parameter")

        date = pd.to_datetime(date_str, format='%Y-%m-%d').date()
        #print("unknown Date: ", date)
        try:
            conn=sqlite3.connect("EntryLog.db")
            cursor= conn.cursor()
            #print("unkkk")
        
            if not conn:
                return jsonify({"error": "Database connection failed"}), 500

            sql = 'SELECT DISTINCT Image_Tag FROM UNKNOWNDB WHERE DATE(Timestamp_entry) = ?'
            cursor.execute(sql, (date,))
            results = cursor.fetchall()
            #print("unknown image results :  ", results)
            first_values = [item[0] for item in results]
            #print(first_values) 
            image_data = []
        
            for image_url in first_values:
                # Fetch the image
                response = requests.get(image_url)
                #print("response: ", response)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    #print("working1")
                    # Example of converting the image to base64
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    #print("working1")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                   #print("working3")
                    image_data.append(img_str)
                else: 
                    #print(f"Error fetching image from {image_url}: {response.status_code}")
                    continue  # Skip this image and move to the next one
            #print("final: ", image_data)
            conn.close()
            # Return the images in JSON format
            return jsonify(image_data), 200
        
        except sqlite3.OperationalError as e:
            #print(f"Database is locked: {e}")
            return jsonify({'error': 'Database is locked, try again later.'}), 500
        
        except sqlite3.Error as e:
            #print(f"SQLite error: {e}")
            return jsonify({'error': 'SQLite error occurred'}), 500
        
        finally:
            # Ensure that the connection is closed
            if conn:
                conn.close()

    except Exception as e:
        #print("unknown images..", e)
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e)}), 500, 500

@app.route('/known-images-by-date', methods=['GET'])
def known_images_by_date():
    try:
        date_str = request.args.get('date')
        
        if not date_str:
            raise ValueError("Missing 'date' parameter")

        date = pd.to_datetime(date_str, format='%Y-%m-%d').date()

        
        try:
                conn=sqlite3.connect("EntryLog.db")
                cursor= conn.cursor()
                #print("kkk")
            
                if not conn:
                    return jsonify({"error": "Database connection failed"}), 500

                sql = 'SELECT DISTINCT id FROM KNOWNDB WHERE DATE(Timestamp_entry) = ?'
                cursor.execute(sql, (date,))
                results1 = cursor.fetchall()
                #conn.close()
                first_values = [item[0] for item in results1]
                #print("first values:  ", first_values)
                conn1=sqlite3.connect("Members.db")
                cursor1= conn1.cursor()
                resultsss=[]
            
                sql = 'SELECT Profile_Picture FROM Members WHERE uid IN ({})'.format(','.join(['?']*len(first_values)))
                cursor1.execute(sql, first_values)
                res = cursor1.fetchall()
                resultsss.append(res)


                #print("known image results :  ", resultsss)
                first_values1  = [item[0] for sublist in resultsss for item in sublist]
                #print("lists ", first_values1) 
                image_data = []
            
                for image_url in first_values1:
                    # Fetch the image
                    response = requests.get(image_url)
                    #print("response: ", response)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        #print("working1")
                        # Example of converting the image to base64
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        #print("working1")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        #print("working3")
                        image_data.append(img_str)
                    else: 
                        print(f"Error fetching image from {image_url}: {response.status_code}")
                        continue  # Skip this image and move to the next one
                #print("final: ", image_data)
                conn.close()
                # Return the images in JSON format
                return jsonify(image_data), 200
            
        except sqlite3.OperationalError as e:
                #print(f"Database is locked: {e}")
                return jsonify({'error': 'Database is locked, try again later.'}), 500
            
        except sqlite3.Error as e:
                #print(f"SQLite error: {e}")
                return jsonify({'error': 'SQLite error occurred'}), 500
            
        finally:
                # Ensure that the connection is closed
                if conn:
                    conn.close()

    except Exception as e:
            #print("known images..", e)
            error_logger.exception("Error occurred", exc_info=True)
            traceback_str = traceback.format_exc()
            return jsonify({"error": str(e)}), 500, 500
    
@app.route('/fetchDetails', methods=['GET'])
def getDetails():
    try:
        name = request.args.get('name')
        print("1...", name)

        if not name:
            return jsonify({"error": "Name parameter is required"}), 400

        try:
            # Connect to the database
            conn = sqlite3.connect("Members.db")
            cursor = conn.cursor()
            sql = 'SELECT Mobile_Number, Membership, Age,Email, Blood_Group, Height, Weight, Address, Emergency_Contact_Name, Emergency_Contact_Number, Membership_start_date, Membership_end_date,Profile_Picture FROM Members WHERE Name = ?'
            cursor.execute(sql, (name,))
            results1 = cursor.fetchall()

            if not results1:
                return jsonify({"error": "No details found for the provided name"}), 404

            # Extract the first result (assuming the name is unique)
            result = results1[0]
            print("results.... ", result)

            # Convert profile picture to base64 (if applicable)
            profile_picture_base64 = None
            if result[9]:  # If Profile_Picture exists
                try:
                    with open(result[12], "rb") as img_file:  # Assuming it's a file path
                        profile_picture_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                except Exception as e:
                    print(f"Error encoding profile picture: {e}")
            
            # Prepare the structured response
            result_dict = {
                "Mobile_Number": result[0],
                "Membership": result[1],
                "Age": result[2],
                "Email": result[3],
                "Blood_Group": result[4],
                "Height": result[5],
                "Weight": result[6],
                "Address": result[7],
                "Emergency_Contact_Name": result[8],
                "Emergency_Contact_Number": result[9],
                "Membership_start_date":result[10],
                "Membership_end_date": result[11], 
                "Profile_Picture": profile_picture_base64  # Include base64 image data if available
            }
            print("result dic ", result_dict)
            return jsonify(result_dict), 200

        except sqlite3.OperationalError as e:
            print(f"SQLite OperationalError: {e}")
            return jsonify({'error': 'Database is locked, try again later.'}), 500
        
        except sqlite3.Error as e:
            print(f"SQLite Error: {e}")
            return jsonify({'error': 'SQLite error occurred'}), 500
        
        finally:
            # Ensure that the connection is closed
            if conn:
                conn.close()

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback_str = traceback.format_exc()
        print("Full Traceback:", traceback_str)  # Log the full traceback for debugging
        return jsonify({"error": str(e)}), 500

   
@app.route('/names-by-date', methods=['GET'])
def names_by_date():
    global mydb
    try:
        date_str = request.args.get('date')
        #print("yesssss")
        if not date_str:
            raise ValueError("Missing 'date' parameter")

        date = pd.to_datetime(date_str, format='%Y-%m-%d').date()

        try:
                conn=sqlite3.connect("EntryLog.db")
                cursor= conn.cursor()
                #print("kkk")
            
                if not conn:
                    return jsonify({"error": "Database connection failed"}), 500

                sql = 'SELECT DISTINCT id FROM KNOWNDB WHERE DATE(Timestamp_entry) = ?'
                cursor.execute(sql, (date,))
                results1 = cursor.fetchall()
                #conn.close()
                first_values = [item[0] for item in results1]
                print("first values:  ", first_values)
                conn1=sqlite3.connect("Members.db")
                cursor1= conn1.cursor()
                resultsss=[]
            
                sql = 'SELECT Name FROM Members WHERE uid IN ({})'.format(','.join(['?']*len(first_values)))
                #print("query", sql)
                cursor1.execute(sql, first_values)
                res = cursor1.fetchall()
                resultsss.append(res)

                #print("unknown image results :  ", results)
                names = [item[0] for sublist in resultsss for item in sublist]
                #print("names:: ", names)
                conn.close()
                # Return the images in JSON format
                return jsonify(names), 200
                
        except sqlite3.OperationalError as e:
                #print(f"Database is locked: {e}")
                return jsonify({'error': 'Database is locked, try again later.'}), 500
            
        except sqlite3.Error as e:
                #print(f"SQLite error: {e}")
                return jsonify({'error': 'SQLite error occurred'}), 500
            
        finally:
                # Ensure that the connection is closed
                if conn:
                    conn.close()

    except Exception as e:
        #print("names data..")
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/timestamp-by-date", methods=['GET'])
def timestamp_date():
    try:
        date_str = request.args.get('date')
        print("yesssss", date_str)
        if not date_str:
            raise ValueError("Missing 'date' parameter")

        date = pd.to_datetime(date_str, format='%Y-%m-%d').date()

        try:
                conn=sqlite3.connect("EntryLog.db")
                cursor= conn.cursor()
                
                sql = 'SELECT max(Timestamp_Entry) FROM KNOWNDB WHERE DATE(Timestamp_Entry) = ? GROUP BY id'
                print("query", sql)
                cursor.execute(sql, (date,))
                print("query executed")
                res = cursor.fetchall()
                
                print("DATES results :  ", res)
                
                conn.close()
                # Return the images in JSON format
                return jsonify(res), 200
        except sqlite3.OperationalError as e:
                #print(f"Database is locked: {e}")
                return jsonify({'error': 'Database is locked, try again later.'}), 500
            
        except sqlite3.Error as e:
                #print(f"SQLite error: {e}")
                return jsonify({'error': 'SQLite error occurred'}), 500
            
        finally:
                # Ensure that the connection is closed
                if conn:
                    conn.close()

    except Exception as e:
        #print("names data..")
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/untimestamp-by-date", methods=['GET'])
def untimestamp_date():
    try:
        date_str = request.args.get('date')
        print("yesssss", date_str)
        if not date_str:
            raise ValueError("Missing 'date' parameter")

        date = pd.to_datetime(date_str, format='%Y-%m-%d').date()

        try:
                conn=sqlite3.connect("EntryLog.db")
                cursor= conn.cursor()
                
                sql = 'SELECT max(Timestamp_Entry) FROM UNKNOWNDB WHERE DATE(Timestamp_Entry) = ? GROUP BY id'
                print("query", sql)
                cursor.execute(sql, (date,))
                print("query executed")
                res = cursor.fetchall()
                
                print("DATES results :  ", res)
                
                conn.close()
                # Return the images in JSON format
                return jsonify(res), 200
        except sqlite3.OperationalError as e:
                #print(f"Database is locked: {e}")
                return jsonify({'error': 'Database is locked, try again later.'}), 500
            
        except sqlite3.Error as e:
                #print(f"SQLite error: {e}")
                return jsonify({'error': 'SQLite error occurred'}), 500
            
        finally:
                # Ensure that the connection is closed
                if conn:
                    conn.close()

    except Exception as e:
        #print("names data..")
        error_logger.exception("Error occurred", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    
    #long_video_thread = threading.Thread(target=capture_and_upload_long_video)
    #long_video_thread.start()
    app.run(debug=True, host="0.0.0.0", port=5000)
    #long_video_thread.join()
    
