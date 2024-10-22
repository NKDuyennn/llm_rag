import streamlit as st
import pymongo
import google.generativeai as genai
import os
# from sentence_transformers import SentenceTransformer
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
# import PyPDF2

# Load environment variables from .env file
load_dotenv()

# Access the key
MONGODB_URI = os.getenv('MONGODB_URI_1')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION_1')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# embedding_model = SentenceTransformer(EMBEDDING_MODEL)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB"""
    try:
      client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python")
      print("Connection to MongoDB successful")
      return client
    except pymongo.errors.ConnectionFailure as e:
      print(f"Connection failed: {e}")
      return None

def ingest_data():
    mongo_uri = MONGODB_URI
    if not mongo_uri:
      print("MONGO_URI not set in environment variables")

    mongo_client = get_mongo_client(mongo_uri)
    # ingest data into MongoDB
    db = mongo_client[DB_NAME]
    collection = db[DB_COLLECTION]

    return collection

def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text")
        return []

    response = genai.embed_content(model="models/text-embedding-004", content=text)
    embedding = response['embedding']
        
    return embedding

def vector_search(user_query):
    """
    Perform a vector search in the MongoDB collection based on the user way

    Args:
    user_query (str): The user's query string.
    collection(MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """
    collection = ingest_data()

    #Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "client_embedding",
            "numCandidates": 150,   # Number of candidate matches to consider
            "limit": 2              # Return top 2 matches
        }
    }

    unset_stage = {
          "$unset": "client_embedding"    # Exclude the 'client_embedding' field from the results
    }

    project_stage = {
          "$project": {
              "_id": 0,     # Exclude the _id field
              "therapist": 1,   # Include the therapist field
              "score": {
                  "$meta": "vectorSearchScore"  # Include the search score
              }
          }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    # Excecute the search
    results = collection.aggregate(pipeline)

    return list(results)

def get_search_result(query, conversation_history):

    get_knowledge = vector_search(query)

    search_result = ""
    for  result in get_knowledge:
        # print('---result', result)
        search_result += f"Answer: {result.get('therapist', 'N/A')}"
        search_result += "\n"

    combined_information = (
        f"{conversation_history}\n"
        "Based on the above conversation, it is crucial to provide an answer that is relevant and contextualized. "
        "Please focus on answering the following query in relation to the conversation:\n"
        f"Query: {query}\n"
        "Here are some results that may help in forming the answer:\n"
        f"{search_result}."
    )

    print("---------------COMBINED_INFOMATION-------------------------")
    print(combined_information)
    print("---------------COMBINED_INFOMATION-------------------------")
    return combined_information

def get_response(rag_user_query):
    response = model.generate_content(rag_user_query)
    print("---------------RESPONSE-------------------------")
    print(response.text)
    print("---------------RESPONSE-------------------------")
    return(response.text)

# Page config
st.set_page_config(page_title="Chat with me", page_icon="🤖")

# Sidebar buttons
sidebar_selection = st.sidebar.selectbox("Choose a page:", ["AI-Therapist", "Vision Mamba"])

if sidebar_selection == "AI-Therapist":

    st.title("Chat with AI-Therapist")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"type": "AI", "content": "Hello, I am an AI-Therapist. How can I help you?"}
        ]

    user_query = st.chat_input("Type your message here...")

    if user_query:
        # Initialize conversation history string
        conversation_history = ""

        # Limit the number of recent messages to concatenate
        num_recent_messages = 3
        recent_messages_count = 0

        # Iterate through chat history in reverse to get the most recent messages
        for message in reversed(st.session_state.chat_history):
            if recent_messages_count >= num_recent_messages:
                break
            
            if message["type"] == "Human":
                conversation_history = f"User: {message['content']}\n" + conversation_history
                recent_messages_count += 1
            elif message["type"] == "AI":
                conversation_history = f"Bot: {message['content']}\n" + conversation_history
        
        print("---------------CONVERSATION_HISTORY-------------------------")
        print(conversation_history)
        print("---------------CONVERSATION_HISTORY-------------------------")

        st.session_state.chat_history.append({"type": "Human", "content": user_query})
        
        # Get the RAG-based search result
        rag_user_query = get_search_result(user_query, conversation_history)

        # Get the response from the model
        response = get_response(rag_user_query)
        st.session_state.chat_history.append({"type": "AI", "content": response})

    # Display conversation
    for message in st.session_state.chat_history:
        if message["type"] == "AI":
            with st.chat_message("AI"):
                st.write(message["content"])
        elif message["type"] == "Human":
            with st.chat_message("Human"):
                st.write(message["content"])
                
elif sidebar_selection == "Vision Mamba":
    st.markdown('<a href="https://nkduyen-mamba.streamlit.app/" target="_blank">CLICK HERE TO Q&A ABOUT MAMBA</a>', unsafe_allow_html=True)
