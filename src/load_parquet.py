# Load necessary libraries
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
import pymongo
import os
import time

# Load environment variables from .env file
load_dotenv()

# Access the key
MONGODB_URI = os.getenv('MONGODB_URI_1')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')

# File path to the parquet file
file_path = "./train-00000-of-00001.parquet"

# Read the parquet file into a pandas DataFrame
dataset_df = pd.read_parquet(file_path)

# Remove data points where the 'client' column is missing
dataset_df = dataset_df.dropna(subset=["client"])
print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text")
        return []

    response = genai.embed_content(model="models/text-embedding-004", content=text)
    embedding = response['embedding']
        
    return embedding

tqdm.pandas()
dataset_df["client_embedding"] = dataset_df["client"].progress_apply(get_embedding)

# Function to establish connection to MongoDB
def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB"""
    try:
        client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python", connectTimeoutMS=80000)
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

# Function to insert data in batches
def insert_data_in_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        try:
            collection.insert_many(batch)
            print(f"Inserted batch {i // batch_size + 1}")
        except pymongo.errors.AutoReconnect as e:
            print(f"AutoReconnect error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            collection.insert_many(batch)
        except Exception as e:
            print(f"An error occurred: {e}")

# Convert dataset to dictionary format
documents = dataset_df.to_dict("records")

# Insert dataset into MongoDB in batches
batch_size = 1000  # Adjust batch size as needed
insert_data_in_batches(documents, batch_size)

print("Data ingestion into MongoDB completed")
