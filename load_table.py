# Load Dataset
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
import pymongo
import os

# Load environment variables from .env file
load_dotenv()

# Access the key
MONGODB_URI = os.getenv('MONGODB_URI')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

file_path = "./test-00000-of-00001.parquet"
# đọc tệp parquet vào DataFrame pandas
dataset_df = pd.read_parquet(file_path)

dataset_df.head(5)
# Data Prepareration

# Remove data point where cilent column is missing
dataset_df = dataset_df.dropna(subset=["client"])
print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())


# dataset_df = dataset_df.to_string(index=False)
def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()

tqdm.pandas()
dataset_df["client_embedding"] = dataset_df["client"].progress_apply(get_embedding)

dataset_df.head()

def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB"""
    try:
      client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python")
      print("Connection to MongoDB successful")
      return client
    except pymongo.errors.ConnectionFailure as e:
      print(f"Connection failed: {e}")
      return None

mongo_uri = MONGODB_URI
if not mongo_uri:
  print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

# ingest data into MongoDB
db = mongo_client['sample_mflix']
collection = db['mental_health']

# insert dataset to mongoDB
documents = dataset_df.to_dict("records")
collection.insert_many(documents)

print("Data ingestion into MongoDB completed")