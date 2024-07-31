# %%

# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['hello', 'thanks'])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Loop through each action, sequence, and frame to record data
for action, sequence, frame in product(actions, range(sequences), range(frames)):
    # If it is the first frame of a sequence, wait for the spacebar key press to start recording
    if frame == 0: 
        while True:
            if keyboard.is_pressed(' '):
                break
            ret, image = cap.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Making predictions using holistic model
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = holistic_model.process(image)
            image.flags.writeable = True

            # Converting back the RGB image to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Drawing Right hand Land Marks
            mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS
            )

            # Drawing Left hand Land Marks
            mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS
            )
            
            # print(image)
            
            cv2.putText(image, f'Recording data for "{action}". Sequence number {sequence}.',
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'Pause.', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Press "Space" when you are ready.', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # print("Hehe")
            cv2.imshow('Camera', image)
            
            cv2.waitKey(1)
            
            
            # Check if the 'Camera' window was closed and break the loop
            if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                break
    else:
        # For subsequent frames, directly read the image from the camera
        ret, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True

        # Converting back the RGB image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Draw the hand landmarks on the image
        mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS
        )

        # Drawing Left hand Land Marks
        mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS
        )

        # Display text on the image indicating the action and sequence number being recorded
        cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                    (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('Camera', image)
        cv2.waitKey(1)

    # Check if the 'Camera' window was closed and break the loop
    if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    # Extract the landmarks from both hands and save them in arrays
    keypoints = keypoint_extraction(results)
    frame_path = os.path.join(PATH, action, str(sequence), str(frame))
    np.save(frame_path, keypoints)

# Release the camera and close any remaining windows
cap.release()
cv2.destroyAllWindows()
