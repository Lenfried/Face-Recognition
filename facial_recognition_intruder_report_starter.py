"""
This file picks up from facial_recognition.py (which you should already have
working) and extends it with three new things:

  1. An "intruder detected" check with any face we don't trust flipping a flag
  2. A photo of the intruder saved to the disk
  3. An email report sent when you close the camera, with all the
     captured photos attached

The work is split into steps. Each step has a comment block explaining
WHAT to do and links to the docs you'll need. The goal is to get comfortable opening docs 
and finding the function you want.

SETUP


1. Sync the project dependencies. We're using uv as the package manager
   and the new libraries (python-dotenv, imutils) are already in
   pyproject.toml one command pulls them in:

       uv sync

   uv docs if you've never used it:
       https://docs.astral.sh/uv/

2. Create a file called `.env` in this folder. Put this inside it:

       EMAIL_ADDRESS=your.email@gmail.com
       EMAIL_PASSWORD=your_app_password_here

3. If you're using Gmail, your normal account password WILL not work
   for SMTP. You need to generate an "App Password" which means you need 2FA 
   you can just use the cisealert account if you dont feel like linking yours:

   Other providers (Outlook, Yahoo, Proton, etc.) have their own SMTP
   settings the steps are the same, just swap the server/port in
   STEP 2.

4. Add `.env` to your `.gitignore` so you don't accidentally push your
   password to GitHub. Seriously, don't skip this one.
"""

import face_recognition
import cv2
import numpy as np
import time
import pickle

try:
    from picamera2 import Picamera2
    USE_PICAM = True
except ImportError:
    USE_PICAM = False


# STEP 1 - Imports

# You'll need a few new modules for this extension. Here's what each one
# is for, with the docs:
#
#   - python-dotenv : loads your .env file into environment variables
#       https://pypi.org/project/python-dotenv/
#
#   - os            : reading env vars, making folders, deleting files
#       https://docs.python.org/3/library/os.html
#
#   - datetime      : timestamps for unique photo filenames
#       https://docs.python.org/3/library/datetime.html
#
#   - imutils.paths : easy way to list every image in a folder
#       https://github.com/PyImageSearch/imutils
#
#   - smtplib       : the actual "send the email" part
#       https://docs.python.org/3/library/smtplib.html
#
#   - email.mime.multipart    : builds the email body + attachments
#       https://docs.python.org/3/library/email.mime.html#module-email.mime.multipart

#   - email.mime.image        : builds the image attachments
#       https://docs.python.org/3/library/email.mime.html#module-email.mime.image
#
#   import the modules above. For the email/mime ones look at
#       MIMEMultipart (the container) and MIMEImage (each photo).
# 




# STEP 2 - Load .env and configure the email server


# Two things to do here:
#
# (a) Tell python-dotenv to actually read the .env file. It's one
#     function call. After it runs, os.getenv("EMAIL_ADDRESS") will
#     return whatever you put in .env.
#         https://pypi.org/project/python-dotenv/
#
# (b) Set up the SMTP server config. Four variables:
#
#         SERVER         - "smtp.gmail.com"
#         PORT           - 587 for Gmail + TLS
#         EMAIL_ADDRESS  - pull from .env with os.getenv(...)
#         EMAIL_PASSWORD - pull from .env with os.getenv(...)
#
#
# TODO: call load_dotenv() and set the four variables above.



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




authorized_names = ["your-name"] # Replace with names you wish to authorise THIS IS CASE-SENSITIVE
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

    # STEP 3a - Intruder flag
    # Add a single boolean for THIS frame that starts False and flips to
    # True the moment we find any face we don't trust. We'll set it
    # below inside the loop.
    #
    # TODO: create `intruder_detected = False`
    # --------------------------------------------------------------------------


    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name not in authorized_names:
                pass

            # STEP 4b - Known but unauthorized?
            # We matched a real person. If their name is NOT in
            # authorized_names, that's still an intruder - flip the flag (Set your variable to true).
            #
            # TODO: remove pass and check `name` against authorized_names; flip the flag
            # ------------------------------------------------------------------

        else:
            # STEP 4c - Complete stranger
            
            # No match at all. `name` is still "Unknown". This is the
            # most obvious intruder case - flip the flag.
            #
            # TODO: flip the flag (you can delete the `pass` once you do)

            pass

        face_names.append(name)


    
    # STEP 5 - Save a photo (with a cooldown so we don't spam the disk)
    
    # If `intruder_detected` is True AND the cooldown has counted down to
    # 0 (or below), save the current frame as a photo. Then reset the
    # cooldown so we don't immediately save another one.
    
    # If the condition isn't met, decrement the cooldown by 1 instead.
    # That way the timer ticks down even on frames where nothing
    # happens.
    
    # The cooldown counter has to LIVE BETWEEN frames. That's
    # why this function takes it as a parameter and returns it back. If
    # you tried to set `cooldown = 0` inside this function it would
    # reset every frame and never actually work. (You'll initialize it
    # outside the loop in STEP 7.)
    #
    # Structure:
    #
    #   1. Make sure the "intruder_photos" folder exists
    #         os.makedirs (use exist_ok=True so it doesn't crash on a
    #         re-run):
    #         https://docs.python.org/3/library/os.html#os.makedirs
    #
    #   2. Build a timestamp string for a unique filename (line 46 of the image capture file we did it already)
    #         datetime.now().strftime(...):
    #         https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    #
    #   3. Join folder + filename into a full path (line 48 of the image capture file)
    #         os.path.join:
    #         https://docs.python.org/3/library/os.path.html#os.path.join
    #
    #   4. Write the frame to that path
    #         cv2.imwrite:
    #         https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    #
    #   5. Reset cooldown to something like 30
    #
    # TODO: implement the if/else block below



    # IMPORTANT - return the cooldown too. If you only return the frame,
    # the caller has no way to know how many ticks are left and you're
    # back to the "resets every frame" bug.
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


# ==============================================================================
# STEP 7 - Initialize the cooldown
# ==============================================================================
# This has to be ABOVE the while loop. If you put it inside, it resets
# to 0 on every iteration and the cooldown does nothing (ask me how I
# know).
#
# TODO: create `cooldown = 0` here
# ==============================================================================



while True:
    # Capture a frame from camera
    if USE_PICAM:
        frame = picam2.capture_array()
    else:
        ret, frame = picam2.read()
        if not ret:
            continue

    # STEP 8 - Thread the cooldown through process_frame

    # process_frame now takes a cooldown and returns one. Update the
    # call so we pass our current cooldown in and unpack the returned
    # value back into the same variable.
    #
    # Before:  processed_frame = process_frame(frame)
    # After:   processed_frame, cooldown = process_frame(frame, cooldown)
    
    processed_frame = process_frame(frame)  # TODO: update this line

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


# STEP 9 - Send the intruder report
# Everything below runs ONCE, after you press 'q'. Doing it here instead
# of inside the camera loop means we don't have to deal with threading
# the (slow) network calls don't block the video feed because the video
# feed is already done.
#
# The recipe:
#
#   1. Build a MIMEMultipart message and set To/From/Subject headers
#      (the headers are just dict-style assignments like
#      `msg["Subject"] = "..."` - the valid keys are the standard email
#      headers from RFC 5322).
#         MIMEMultipart:
#           https://docs.python.org/3/library/email.mime.html#email.mime.multipart.MIMEMultipart
#         RFC 5322 (header reference):
#           https://datatracker.ietf.org/doc/html/rfc5322#section-3.6
#
#   2. List every image in the intruder_photos folder
#         imutils.paths.list_images:
#           https://github.com/PyImageSearch/imutils
#
#   3. If the list is empty, print something like "no intruders this
#      run" and stop. No need to send a blank email.
#
#   4. For each image: read it (cv2.imread), encode it as JPEG bytes
#      (cv2.imencode), wrap those bytes in a MIMEImage, and attach it
#      to msg. Keep a list of which images you successfully attached -
#      you'll need it in step 6.
#         cv2.imread:
#           https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
#         cv2.imencode:
#           https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63
#         MIMEImage:
#           https://docs.python.org/3/library/email.mime.html#email.mime.image.MIMEImage
#
#   5. Connect to SERVER on PORT with smtplib.SMTP (use a `with`
#      statement so it auto-closes). Then in order:
#         server.starttls()    # upgrade to encrypted
#         server.login(...)    # use your EMAIL_ADDRESS / EMAIL_PASSWORD
#         server.sendmail(...) # use msg.as_string() for the message arg
#
#         smtplib.SMTP:
#           https://docs.python.org/3/library/smtplib.html#smtplib.SMTP
#
#   6. ONLY after sendmail() succeeds, delete the files you attached
#      (os.remove). If you delete them before sending and the network
#      call fails, you lose your evidence forever. Doing it after means
#      a failed send leaves the photos on disk for the next run.
#         os.remove:
#           https://docs.python.org/3/library/os.html#os.remove
#
# Wrap the whole thing in try/except so a Gmail hiccup doesn't crash
# silently. Print the exception so you can debug it.
# ==============================================================================

# TODO: implement step 9 here
