import streamlit as st
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow import keras
import language_tool_python
import pymongo
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI_1')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION_1')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB"""
    try:
        client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

def ingest_data():
    mongo_client = get_mongo_client(MONGODB_URI)
    db = mongo_client[DB_NAME]
    collection = db[DB_COLLECTION]
    return collection

def get_embedding(text: str) -> list[float]:
    if not text.strip():
        return []
    response = genai.embed_content(model="models/text-embedding-004", content=text)
    embedding = response['embedding']
    return embedding

def vector_search(user_query):
    collection = ingest_data()
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "client_embedding",
            "numCandidates": 150,
            "limit": 2
        }
    }
    unset_stage = {
        "$unset": "client_embedding"
    }
    project_stage = {
        "$project": {
            "_id": 0,
            "therapist": 1,
            "score": {
                "$meta": "vectorSearchScore"
            }
        }
    }
    pipeline = [vector_search_stage, unset_stage, project_stage]
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, conversation_history):
    get_knowledge = vector_search(query)
    search_result = ""
    for result in get_knowledge:
        search_result += f"Answer: {result.get('therapist', 'N/A')}\n"
    combined_information = (
        f"{conversation_history}\n"
        "Based on the above conversation, it is crucial to provide an answer that is relevant and contextualized. "
        "Please focus on answering the following query in relation to the conversation:\n"
        f"Query: {query}\n"
        "Here are some results that may help in forming the answer:\n"
        f"{search_result}."
    )
    return combined_information

def get_response(rag_user_query):
    response = model.generate_content(rag_user_query)
    return response.text

# Initialize Streamlit UI
st.set_page_config(page_title="Chat with me", page_icon="🤖")
sidebar_selection = st.sidebar.selectbox("Choose a page:", ["AI-Therapist"])

if sidebar_selection == "AI-Therapist":
    st.title("Chat with AI-Therapist")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"type": "AI", "content": "Hello, I am an AI-Therapist. How can I help you?"}
        ]
    
    # Initialize state for camera
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
        st.session_state.cap = None

    if 'last_text' not in st.session_state:
        st.session_state.last_text = ""

    # Move the buttons to the sidebar
    start_cam = st.sidebar.button("Start Camera", key="start_camera")
    end_cam = st.sidebar.button("Close Camera", key="end_camera")
    use_text = st.sidebar.button("Use Text", key="use_text")

    if start_cam:
        if st.session_state.camera_active:
            st.session_state.cap.release()
            cv2.destroyAllWindows()
            st.session_state.last_text = ""
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.camera_active = True

    if end_cam:
        if st.session_state.camera_active:
            st.session_state.camera_active = False
            st.session_state.cap.release()
            cv2.destroyAllWindows()
            # st.session_state.last_text = ""

    if use_text:
        if st.session_state.camera_active:
            st.session_state.camera_active = False
            st.session_state.cap.release()
            cv2.destroyAllWindows()
            # st.session_state.last_text = ""

    # Handle camera and sign language recognition
    if st.session_state.camera_active:
        PATH = os.path.join('data')
        sentence, keypoints, last_prediction, grammar_result = [], [], [], []

        actions = np.array(os.listdir(PATH))
        model = keras.models.load_model('my_model.h5')
        tool = language_tool_python.LanguageToolPublicAPI('en-UK')
        mp_holistic = mp.solutions.holistic
        holistic_model = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils

        cap = st.session_state.cap
        if not cap.isOpened():
            st.write("Cannot access camera.")
        else:
            video_placeholder = st.sidebar.empty()
            while st.session_state.camera_active:
                ret, image = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic_model.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                keypoints.append(keypoint_extraction(results))

                if len(keypoints) == 10:
                    keypoints = np.array(keypoints)
                    prediction = model.predict(keypoints[np.newaxis, :, :])
                    keypoints = []
                    if np.amax(prediction) > 0.9:
                        if last_prediction != actions[np.argmax(prediction)]:
                            sentence.append(actions[np.argmax(prediction)])
                            last_prediction = actions[np.argmax(prediction)]

                if len(sentence) > 7:
                    sentence = sentence[-7:]

                if keyboard.is_pressed(' '):
                    sentence, keypoints, last_prediction, grammar_result = [], [], [], []

                if sentence:
                    sentence[0] = sentence[0].capitalize()
                    if len(sentence) >= 2:
                        if sentence[-1] in string.ascii_letters:
                            if sentence[-2] in string.ascii_letters or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                                sentence[-1] = sentence[-2] + sentence[-1]
                                sentence.pop(len(sentence) - 2)
                                sentence[-1] = sentence[-1].capitalize()

                if keyboard.is_pressed('enter'):
                    text = ' '.join(sentence)
                    grammar_result = tool.correct(text)

                display_text = grammar_result if grammar_result else ' '.join(sentence)
                st.session_state.last_text = display_text
                textsize = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2
                cv2.putText(image, display_text, (text_X_coord, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                video_placeholder.image(image, channels="BGR")
    else: 
        user_query = None

        st.write("Camera Output:", st.session_state.last_text)
        if st.session_state.last_text != "":
            user_query = st.session_state.last_text
            st.session_state.last_text = ""

        if user_query:
            pass
        else:
            user_query = st.chat_input("Type your message here...")

        if user_query:
            conversation_history = ""
            num_recent_messages = 3
            recent_messages_count = 0
            for message in reversed(st.session_state.chat_history):
                if recent_messages_count >= num_recent_messages:
                    break
                if message["type"] == "Human":
                    conversation_history = f"User: {message['content']}\n" + conversation_history
                    recent_messages_count += 1
                elif message["type"] == "AI":
                    conversation_history = f"Bot: {message['content']}\n" + conversation_history

            st.session_state.chat_history.append({"type": "Human", "content": user_query })
            
            rag_user_query = get_search_result(user_query, conversation_history)
            response = get_response(rag_user_query)
            st.session_state.chat_history.append({"type": "AI", "content": response})

        for message in st.session_state.chat_history:
            if message["type"] == "AI":
                with st.chat_message("AI"):
                    st.write(message["content"])
            elif message["type"] == "Human":
                with st.chat_message("Human"):
                    st.write(message["content"])
        
        end_cam = False

        