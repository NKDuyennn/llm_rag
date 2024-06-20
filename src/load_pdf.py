from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import pymongo
import os
import time

# Access the key
MONGODB_URI = os.getenv('MONGODB_URI_2')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION_2')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Đường dẫn tới tệp PDF
pdf_path = 'D:\\Work\\Jobfair\\RAG_gemini_mongoDB\\data\\pdf\\2404.18861v1.pdf'

# Tạo loader để tải tài liệu PDF
loader = PyPDFLoader(pdf_path)

# Tải tài liệu PDF thành danh sách các đối tượng Document
documents = loader.load()

# Kiểm tra cấu trúc của documents
print("Documents structure:", documents)

# Khởi tạo RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

# Tách các tài liệu thành các đoạn nhỏ hơn
document_chunks = text_splitter.split_documents(documents)

# Trích xuất phần text từ mỗi đối tượng Document
texts = [doc.page_content for doc in document_chunks]

# Chuyển texts thành DataFrame với cột 'content'
df = pd.DataFrame({'content': texts})

# # Hiển thị DataFrame
# print(df)
dataset_df = df

genai.configure(api_key=GOOGLE_API_KEY)


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text")
        return []

    response = genai.embed_content(model="models/text-embedding-004", content=text)
    embedding = response['embedding']
        
    return embedding

tqdm.pandas()
dataset_df["embedding"] = dataset_df["content"].progress_apply(get_embedding)

# Function to establish connection to MongoDB
def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB"""
    try:
        client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python", connectTimeoutMS=20000)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

# Establish MongoDB connection
mongo_client = get_mongo_client(MONGODB_URI)

# Check if MongoDB connection was successful
if mongo_client is None:
    print("MongoDB connection failed. Exiting.")
    exit(1)

# Specify the database and collection
db = mongo_client[DB_NAME]
collection = db[DB_COLLECTION]

# insert dataset to mongoDB
documents = dataset_df.to_dict("records")
collection.insert_many(documents)

print("Data ingestion into MongoDB completed")