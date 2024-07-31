# %%

# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model')

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()


# Run the loop while the camera is open
while cap.isOpened():
    # Read a frame from the camera
    _, image = cap.read()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
    keypoints.append(keypoint_extraction(results))

    # Check if 10 frames have been accumulated
    if len(keypoints) == 10:
        # Convert keypoints list to a numpy array
        keypoints = np.array(keypoints)
        # Make a prediction on the keypoints using the loaded model
        prediction = model.predict(keypoints[np.newaxis, :, :])
        # Clear the keypoints list for the next set of frames
        keypoints = []

        # Check if the maximum prediction value is above 0.9
        if np.amax(prediction) > 0.9:
            # Check if the predicted sign is different from the previously predicted sign
            if last_prediction != actions[np.argmax(prediction)]:
                # Append the predicted sign to the sentence list
                sentence.append(actions[np.argmax(prediction)])
                # Record a new prediction to use it on the next cycle
                last_prediction = actions[np.argmax(prediction)]

    # Limit the sentence length to 7 elements to make sure it fits on the screen
    if len(sentence) > 7:
        sentence = sentence[-7:]

    # Reset if the "Spacebar" is pressed
    if keyboard.is_pressed(' '):
        sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

    # Check if the list is not empty
    if sentence:
        # Capitalize the first word of the sentence
        sentence[0] = sentence[0].capitalize()

    # Check if the sentence has at least two elements
    if len(sentence) >= 2:
        # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
        if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
            # Check if the second last element of sentence belongs to the alphabet or is a new word
            if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                # Combine last two elements
                sentence[-1] = sentence[-2] + sentence[-1]
                sentence.pop(len(sentence) - 2)
                sentence[-1] = sentence[-1].capitalize()

    # Perform grammar check if "Enter" is pressed
    if keyboard.is_pressed('enter'):
        # Record the words in the sentence list into a single string
        text = ' '.join(sentence)
        # Apply grammar correction tool and extract the corrected result
        grammar_result = tool.correct(text)

    if grammar_result:
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2

        # Draw the sentence on the image
        cv2.putText(image, grammar_result, (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2

        # Draw the sentence on the image
        cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the image on the display
    cv2.imshow('Camera', image)

    cv2.waitKey(1)

    # Check if the 'Camera' window was closed and break the loop
    if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Shut off the server
tool.close()
