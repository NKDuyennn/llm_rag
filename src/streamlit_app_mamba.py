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
MONGODB_URI = os.getenv('MONGODB_URI_2')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION_2')
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
            "path": "embedding",
            "numCandidates": 150,   # Number of candidate matches to consider
            "limit": 4              # Return top 2 matches
        }
    }

    unset_stage = {
          "$unset": "embedding"    # Exclude the 'client_embedding' field from the results
    }

    project_stage = {
          "$project": {
              "_id": 0,     # Exclude the _id field
              "content": 1,   # Include the therapist field
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
        search_result += f"Answer: {result.get('content', 'N/A')}"
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
sidebar_selection = st.sidebar.selectbox("Choose a page:", ["Vision Mamba", "AI-Therapist"])

if sidebar_selection == "Vision Mamba":

    st.title("Q&A about Vision Mamba")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a AI. How can I help you?"),
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
            
            if isinstance(message, HumanMessage):
                conversation_history = f"User: {message.content}\n" + conversation_history
                recent_messages_count += 1
            elif isinstance(message, AIMessage):
                conversation_history = f"Bot: {message.content}\n" + conversation_history
            
        
        print("---------------CONVERSATION_HISTORY-------------------------")
        print(conversation_history)
        print("---------------CONVERSATION_HISTORY-------------------------")
        # conversation_history += "Given the above conversation, "

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        # Get the RAG-based search result
        rag_user_query = get_search_result(user_query, conversation_history)

        # Get the response from the model
        response = get_response(rag_user_query)
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

elif sidebar_selection == "AI-Therapist":
    st.markdown('<a href="https://nkduyen-therapist.streamlit.app/">Click here to access the Chat with AI-Therapist</a>', unsafe_allow_html=True)

