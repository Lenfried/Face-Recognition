import face_recognition
import cv2
import numpy as np
try:
    from picamera2 import Picamera2
    USE_PICAM = True
except ImportError:
    USE_PICAM = False

import time
import os
import pickle
from datetime import datetime
from imutils import paths



from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


load_dotenv()

# Server config
# Step 2 set up your server configuration
SERVER = "smtp.gmail.com" # Using gmail for the demo but you can look at any provider and check if they have support
PORT = 587 # Gmails unique port
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS") # Export your variables you set
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
if USE_PICAM:
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
    picam2.start()
else:
    picam2 = cv2.VideoCapture(0)
    picam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    picam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



# Initialize our variables
cv_scaler = 4 # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# List of names that will trigger the GPIO pin
authorized_names = ["alberto"]  # Replace with names you wish to authorise THIS IS CASE-SENSITIVE


def process_frame(frame, cooldown):
    global face_locations, face_encodings, face_names
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    # Step one filp to inrtuder detected when unknown or unauthrized is discoverd
    # No documentation needed jsut changing up a few vaible placements and names
    intruder_detected = False
    
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Check if the detected face is in our authorized list
            if name not in authorized_names:
                intruder_detected = True

        elif name == "Unknown":
                intruder_detected = True
        face_names.append(name)
            
            
    
    # Step 3 Set up your condition to work now
    # If the flip was switched on then we have to take the photo of the intruder
    if intruder_detected and cooldown <= 0:
        
        # Create the folder if is does exist (Link to os.makedirs documentation)
        folder = "intruder_photos"
        os.makedirs(folder, exist_ok=True)


        # Create a timestamp we did this in the image capture script (Could provide some documentation here)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Give the file a name and concatate with the timestamp
        filename = f"intruder_{timestamp}.jpg"

        # Join the file path so open cv can write to it(link to path.join documentation)
        filepath = os.path.join(folder, filename)

        # Write to the file used in the image capture file Could also link docs if confused
        cv2.imwrite(filepath, frame)
        
        # Reset your cooldown period
        cooldown = 30

    else:
        # Increment our cooldown each frame
        cooldown -=1

        # Make sure we are returning are cooldown so we dont reset it to 0 every frame but to the previous decrement
    return frame, cooldown

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
        # Add an indicator if the person is authorized
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Intialize the cooldown
cooldown = 0

while True:
    # Capture a frame from camera
    if USE_PICAM:
        frame = picam2.capture_array()
    else:
        ret, frame = picam2.read()
        if not ret:
            continue
    
    # Process the frame with the function
    processed_frame, cooldown = process_frame(frame, cooldown)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
if USE_PICAM:
    picam2.stop()
else:
    picam2.release()

# When the camera shuts off we now have to try to send the intruder report to our emails
try:
    # Set up a msg that can hold mulitple types Link to the MIME Documentation
    msg = MIMEMultipart()
    # Set up the sender and receiver along with the subject of the message. Use standard m
    msg["To"] = EMAIL_ADDRESS
    msg["From"] = EMAIL_ADDRESS
    msg["Subject"] = f"Intruder Breakdown {datetime.now()}"

    # Put all of the images in the intruder_phots insde of a list for use to sift through 
    imagePath = list(paths.list_images("intruder_photos"))  

    # Edge case in case there are no images
    if not imagePath:
        print("no intruders this run") 
    else:
    # Keep track of the image paths we found in a list so we have the record hotory incase something goes wrong with the server
    # Encode image to a MIMEImage and remove it
        images_found = []
        for image in imagePath:
            # Read the image with cv2 then encode it could link to the docs here
            raw_img = cv2.imread(image)
            success, encoded_img = cv2.imencode(".jpg", raw_img)
            if success:
                # If converstion was ok we then convert to byters and then create a MIME Image object
                # Attach to what our message has currently
                msg.attach(MIMEImage(encoded_img.tobytes()))

        # Have our server set up i might either leave this hardcoded or again links to the docs with just order defined
        # You want to start and open upi the server, then login, then send the email
        with smtplib.SMTP(SERVER, PORT) as server:
            server.starttls()
            server.login(user=EMAIL_ADDRESS, password=EMAIL_PASSWORD)
            server.sendmail(from_addr=EMAIL_ADDRESS, to_addrs=EMAIL_ADDRESS , msg= msg.as_string())

        # Free up space after you have sent the email
        for image in images_found:
            os.remove(image)

except Exception as e:
    print(f"email failed: {e}")