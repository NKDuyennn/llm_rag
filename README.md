# `LLM RAG` - Streamlit RAG Language Model App 🤖

## 🌟 Tổng Quan 
This Streamlit App uses Retrieval-Augmented Generation (RAG) combined with the Gemini Large Language Model (LLM) and MongoDB, a database that allows for vector storage and search. The application enables users to upload PDF files 📂, ask questions related to the content of these files ❓, and receive AI-generated answers based on the uploaded content 📚.

## Mục lục
* [Overview](#-overview)
* [Table of Contents](#table-of-contents)
* [System Architecture](#system-architecture)
* [How It Works and Demo](#-how-it-works-and-demo)
* [Project Structure](#project-structure)
* [Deployment Steps](#deployment-steps)
* [Host Streamlit App](#host-streamlit-app-for-free-with-streamlit-and-github)
* [Contact](#-contact)

## Cấu trúc hệ thống:
The diagram below illustrates the data flow through the system:
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/diagram.jpg" width="100%" />
</p>  

- **INFORMATION EXTRACTION**: I use LangChain to split the data into smaller chunks with `chunk_size=512` and `chunk_overlap=64`—these parameters can be adjusted. Then, I store the content of each chunk in the `content` column of a table and save it in a `collection` in MongoDB.
- **VECTORIZATION**: Here, I use the Gemini API to host the application for free on Streamlit. If you have the resources, you can use models on Hugging Face or others.
- **RELEVANT DOCUMENTS RETRIEVAL**: After embedding the chunks from the `content` column, I store them in the corresponding `embedding` column and create a search index using vector search for this column. Through vector search, I compare the similarity between the `user_query` and the data chunks from the PDF.
- **LLM QUERYING**: The prompt is enriched by combining `user_query + relevant_documents + history_conversation`. You can customize the number of relevant documents returned and adjust the length of the previous conversation history included in the prompt. Then, I feed this into Gemini’s LLM model, though you can use other models.
- **STREAMLIT**: The application's interface is built with Streamlit.
- **Note** 💡: This can also be applied to data sources in table format, without needing to process PDF files—you can customize the columns you want to embed `src/load_parquet.py` .

## ❓ Cách hoạt động và Demo:  
- You can use my application here: [LLM-RAG](https://nkduyen-customdata.streamlit.app/)
- **Note** 💡: You must delete the uploaded file before asking questions  
The Streamlit LLM-RAG application interface is as follows:

<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/ui.png" width="100%" />
</p>

- **Upload PDF Document** 📂: Upload the PDF file containing the data you want to enrich the model with.
- **Choose a page** 🔍: You can select from several pre-installed models:
    - **AI-Therapist**: A psychological counseling chatbot trained on the [mental-health-dataset](https://huggingface.co/datasets/fadodr/mental_health_dataset?row=75).
    - **Vision Mamba**: A chatbot that provides information related to Mamba and Vision Mamba.
- **Chat with your Custom Data** 💡: This is where you can submit your questions and receive answers based on the information you’ve added.
### Demo
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/LLM_RAG_Demo.gif" width="100%" />
</p>

## Cấu trúc của Project
The main directories of the project are organized as follows:

```
llm_rag/
|--- .devcontainer/
  |--- devcontainer.json           # Configuration file for the development environment
|--- data/                         # Data for the Chatbot to learn
|--- image/                        # Project image directory
|--- src/
  |--- app.py                      # Code for the Chat with Your Custom Data application
  |--- load_parquet.py             # Code for processing .parquet data into the database and embedding
  |--- app.py                      # Code for processing PDF data embedding and uploading to the database
  |--- streamlit_app_mamba.py      # Code for the Q&A about Mamba application
  |--- streamlit_app_therapist.py  # Code for the Chat with AI-Therapist application
|--- .env.example                  # Sample environment variable file
|--- README.md                     # This file
|--- requirements.txt              # Libraries required for the project
```

## Chuẩn bị
- Python 3.9 trở đi
- Streamlit
- MongoDB
- Sentence Transformer (Nếu không dùng API của Gemini)
- Google GenerativeAI
- Langchain

## Các bước triển khai
To deploy the project on your computer, follow these steps:

### **Step 1: Install MongoDB Atlas**
- Visit [MongoDB Atlas](https://www.mongodb.com/lp/cloud/atlas/try4?utm_source=google&utm_campaign=search_gs_pl_evergreen_atlas_core_prosp-brand_gic-null_apac-vn_ps-all_desktop_eng_lead&utm_term=mongodb%20atlas&utm_medium=cpc_paid_search&utm_ad=e&utm_ad_campaign_id=12212624377&adgroup=115749709423&cq_cmp=12212624377&gad_source=1&gclid=CjwKCAjwps-zBhAiEiwALwsVYVTSsKs0UtYI5IacyXKIAN0ccyymKRJFysZCR8tpWMNZtbMZpXdz9xoCctkQAvD_BwE)
- Create an account, create a project, create a database, and create a collection to store your data
- Create a column in the collection that will contain the `vector embedding`
- Create an `index` for that column
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/index1.png" width="60%" />
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/index2.png" width="60%" />
</p>  

- Obtain the MongoDB URI for the database you just created [Instructions](https://www.mongodb.com/docs/v5.2/reference/connection-string/)

### **Step 2: Create Environment Variables**
   Create a `.env` file in your project with the following content:
   ```
    GOOGLE_API_KEY = <Your Gemini API Key>
    MONGODB_URI = <Your MongoDB URI>
    EMBEDDING_MODEL = <Path to the Hugging Face embedding model>  # If not using the Gemini Embedding Model
    DB_NAME = <Your Database Name>
    DB_COLLECTION = <Your Database Collection Name>
   ```

### **Step 3: Install Required Libraries:**
- Open the `terminal` and ensure you are in the project directory
- Set up your virtual environment using `venv` or `conda`:
   ```
   # Using venv
   python -m venv env_llm_rag
   source env_llm_rag/bin/activate
   
   # Using conda
   conda create --name env_llm_rag
   conda activate env_llm_rag
   ```
- Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
   
### **Step 4: Upload Data to MongoDB:**
There are two types of data corresponding to two files:
- If your data is raw, in PDF format, use the code `src/load_pdf.py`
- If your data is already in table format, use the code `src/load_parquet.py` and customize the columns you want to embed
- If you want to upload data from the UI, you can skip this step

### **Step 5: Run the Streamlit Application:**
To run the file using Streamlit:
   ```
   streamlit run <file_path>.py
   ```
- Refer to the code `src/streamlit_app_mamba.py` if your data processing is complete
- Refer to the code `src/app.py` if you want to process PDF files uploaded from the UI
The Streamlit application will be deployed at **`http://localhost:8501`** after running the above command

### **Note**
In the `src/app.py` code, you need to adjust the `vector_search` function to match the `index` you created in the database and any related parameters.  

## Host Streamlit App miễn phí với Streamlit và github:
Hosting a Streamlit app for free:

### **Step 1: Create a New Repository with a Structure Similar to This Project**
- Make sure the repository includes a `requirements.txt` file and a `.py` file.

### **Step 2:**
- Create a Streamlit account and link it to your GitHub account.
- Click on `Create App`.
- Fill in the corresponding fields:
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/streamlit1.png" width="80%" />
</p>  

- Select Advanced Settings and add your environment variables here:
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/streamlit2.png" width="80%" />
</p>  

### **Step 3:**
Deploy, and you have successfully hosted your Streamlit App. You can use the app via a link like `<your-domain>.streamlit.app`.

### **Note**
Since this is a free plan, the resources provided by Streamlit are limited, so it is advisable to use an embedding model with an API Key.

## 🌐 Liên hệ:
<div align="center">
  <a href="https://www.facebook.com/nkduyen.2310/">
  <img src="https://img.shields.io/badge/Facebook-%233b5998.svg?&style=for-the-badge&logo=facebook&logoColor=white" alt="Facebook" style="margin-bottom: 5px;"/>
  </a>
  <a href="https://www.linkedin.com/in/nkduyennn2310/">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" style="margin-bottom: 5px;"/>
  </a>
  <a href="https://github.com/NKDuyennn">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" style="margin-bottom: 5px;"/>
  </a>
</div>
