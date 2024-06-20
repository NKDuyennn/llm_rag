import streamlit as st
import pymongo
import google.generativeai as genai
import os
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Access the key
MONGODB_URI = os.getenv('MONGODB_URI')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
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
            "limit": 4             # Return top 2 matches
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
        search_result += f"Infor: {result.get('content', 'N/A')}"
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

def pdf_to_mongodb(path):

    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    document_chunks = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in document_chunks]
    dataset_df = pd.DataFrame({'content': texts})

    tqdm.pandas()
    dataset_df["embedding"] = dataset_df["content"].progress_apply(get_embedding)
    
    collection = ingest_data()

    collection.delete_many({})
    # insert dataset to mongoDB
    documents = dataset_df.to_dict("records")
    collection.insert_many(documents)

    print("Data ingestion into MongoDB completed")

# Page config
st.set_page_config(page_title="Chat with ME", page_icon="🤖")

# Sidebar for file upload
st.sidebar.title("Upload PDF Document ")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    
    save_folder = "data/pdf"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"File saved at {save_path}")
    st.sidebar.text(f"Waitting ...")
    pdf_to_mongodb(save_path)
    st.sidebar.success(f"Update database sucessful")

    try:
        os.remove(save_path)
        st.sidebar.success(f"File {uploaded_file.name} deleted from {save_path}")
    except Exception as e:
        st.sidebar.error(f"Error deleting file: {e}")
# Sidebar buttons
sidebar_selection = st.sidebar.selectbox("Choose a page:", ["Custom Data", "AI-Therapist", "Vision Mamba"])

if sidebar_selection == "Custom Data":
    st.title("Chat with Your Custom Data")
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"type": "AI", "content": "Hello, I am an AI. How can I help you?"}
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
                
elif sidebar_selection == "AI-Therapist":
    st.title("Chat with AI-Therapist")
    st.markdown('<a href="https://nkduyen-therapist.streamlit.app/" target="_blank">CLICK HERE TO CHAT WITH AI-THERAPIST</a>', unsafe_allow_html=True)
elif sidebar_selection == "Vision Mamba":
    st.title("Q&A about Vision Mamba")
    st.markdown('<a href="https://nkduyen-mamba.streamlit.app/" target="_blank">CLICK HERE TO Q&A ABOUT MAMBA</a>', unsafe_allow_html=True)

