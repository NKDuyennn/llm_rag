# `LLM RAG` - Streamlit RAG Language Model App 🤖

## 🌟 Tổng Quan 
Streamlit App này sử dụng Retrieval-Augmented Generation (RAG) kết hợp với Large Language Model (LLM) của Gemini và MongoDB, một cơ sở dữ liệu cho phép lưu trữ và tìm kiếm theo vector. Ứng dụng cho phép người dùng tải lên file PDF 📂, đặt câu hỏi liên quan đến nội dung của các file này ❓ và nhận câu trả lời được tạo ra bởi AI-generated dựa trên nội dung đã tải lên 📚. 

## Mục lục
* [Tổng quan](#-tổng-quan)
* [Mục lục](#mục-lục)
* [Cấu trúc hệ thống](#cấu-trúc-hệ-thống)
* [Cách hoạt động và Demo](#-cách-hoạt-động-và-demo)
* [Cấu trúc của Project](#cấu-trúc-của-project)
* [Các bước triển khai](#các-bước-triển-khai)
* [Host Streamlit App](#host-streamlit-app-miễn-phí-với-streamlit-và-github)
* [Liên hệ](#-liên-hệ)

## Cấu trúc hệ thống:
Sơ đồ dưới đây minh họa luồng dữ liệu qua hệ thống:
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/diagram.jpg" width="100%" />
</p>  

- **Lưu ý** 💡: Có thể áp dụng cả với những nguồn dữ liệu ở dạng bảng sẵn, không cần phải xử lý file PDF 

## ❓ Cách hoạt động và Demo:
Ứng dụng Streamlit LLM-RAG có giao diện như sau:

<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/ui.png" width="100%" />
</p>

- **Upload PDF Document** 📂: Tải lên file PDF chứa dữ liệu mà bạn muốn thêm thông tin cho Model
- **Choose a page** 🔍: Có thể chọn một vài mô hình mà mình đã cài đặt từ trước
    - **AI-Therapist**: Chatbot tư vấn tâm lý được đào tạo từ tập dữ liệu [mental-health-dataset](https://huggingface.co/datasets/fadodr/mental_health_dataset?row=75).
    - **Vision Mamba**: Chatbot trả lời thông tin liên quan về Mamba và Vision Mamba.
- **Chat with your Custom Data** 💡: Nơi bạn có thể gửi câu hỏi của mình và nhận được câu trả lời theo thông tin bạn đã thêm vào.

## Cấu trúc của Project
Các thư mục chính của dự án được sắp xếp như sau:

```
llm_rag/
|--- .devcontainer/
  |--- devcontainer.json           # Tệp cấu hình cho development evironment
|--- data/                           # Dữ liệu muốn Chatbot biết thêm 
|--- image/                          # Thư mục ảnh của dự án
|--- src/
  |--- app.py                      # Code ứng dụng Chat with Your Custom Data
  |--- load_parquet.py             # Code để xử lý dữ liệu dạng .parquet lên database và embedding
  |--- app.py                      # Code để xử lý dữ liệu pdf embedding và upload lên database
  |--- streamlit_app_mamba.py      # Code ứng dụng Q&A about Mamba
  |--- streamlit_app_therapist.py  # Code ứng dụng Chat with AI-Therapist
|--- .env.example                    # File biến môi trường mẫu
|--- README.md                       # File này
|--- requirements.txt                # Thư viện cần sử dụng trong dự án
```

## Chuẩn bị
- Python 3.9 trở đi
- Streamlit
- MongoDB
- Sentence Transformer (Nếu không dùng API của Gemini)
- Google GenerativeAI
- Langchain

## Các bước triển khai
Để triển khai dự án trên máy tính của bạn, hãy làm theo các bước sau

### **Bước 1: Cài đặt MongoDB Atlas**
- Truy cập [MongoDB Atlas](https://www.mongodb.com/lp/cloud/atlas/try4?utm_source=google&utm_campaign=search_gs_pl_evergreen_atlas_core_prosp-brand_gic-null_apac-vn_ps-all_desktop_eng_lead&utm_term=mongodb%20atlas&utm_medium=cpc_paid_search&utm_ad=e&utm_ad_campaign_id=12212624377&adgroup=115749709423&cq_cmp=12212624377&gad_source=1&gclid=CjwKCAjwps-zBhAiEiwALwsVYVTSsKs0UtYI5IacyXKIAN0ccyymKRJFysZCR8tpWMNZtbMZpXdz9xoCctkQAvD_BwE)
- Tạo tài khoản, tạo project, tạo database, tạo collection - nơi lưu trữ dữ liệu
- Tạo 1 cột trong collection sẽ chứa `vector embedding` đánh `chỉ mục (index)` cho cột đó
- Lấy MongoDB URI của database vừa tạo [Hướng dẫn](https://www.mongodb.com/docs/v5.2/reference/connection-string/)

### **Bước 2: Tạo biến môi trường**
   Tạo một file `.env` trong dự án của bạn gồm có:
   ```
    GOOGLE_API_KEY = <Gemnini API Key của bạn>
    MONGODB_URI = <MongoDB-URI của bạn>
    EMBEDDING_MODEL = <Đường dẫn model embedding Hugging face>  #Nếu không dùng Model Gemini Embedding
    DB_NAME = <Database Name của bạn>
    DB_COLLECTION = <Database Collection Name của bạn>
   ```

### **Bước 3: Cài đặt các thư viện cần thiết:**
- Mở `terminal` và chắc chắn đang ở thư mục của project
- Cài đặt môi trường ảo của bạn dùng `venv` hoặc `conda`:
   ```
   # Sử dụng venv
   python -m venv env_llm_rag
   source env_llm_rag/bin/activate
   
   # Sử dụng conda
   conda create --name env_llm_rag
   conda activate env_llm_rag
   ```
- Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```
   
### **Bước 4: Upload dữ liệu lên MongoDB:**
Có 2 kiểu dữ liệu tương ứng 2 file:
- Nếu dữ liệu của bạn là bản raw, pdf thì sử dụng code `src/load_pdf.py`
- Nếu kiểu dữ liệu của bạn đã là dạng bảng thì sử dụng code `src/load_parquet.py` và tùy chỉnh cột muốn embedding
- Nếu bạn muốn upload lên từ UI thì bỏ qua bước này

### **Bước 5: Chạy Ứng dụng Streamlit:**
Để chạy file sử dụng  streamlit:
   ```
   streamlit run <file_path>.py
   ```
- Tham khảo code `src/streamlit_app_mamba.py` nếu đã xử lý xong dữ liệu
- Tham khảo code `src/app.py` nếu muốn xử lý file pdf up lên từ UI
Ứng dụng Streamlit sẽ được triển khai tại **`http://localhost:8501`** sau khi chạy dòng lệnh trên

### **Lưu ý**
Trong code `src/app.py` cần chỉnh hàm `vector_search` để phù hợp với `index` bạn tạo trong database cũng như các tham số liên quan.

## Host Streamlit App miễn phí với Streamlit và github:

Để host Streamlit App miễn phí, hãy làm theo các bước sau:
### **Bước 1: Tạo 1 Repository mới có cấu trúc giống như dự án này**
### **Bước 2:**
- Tạo tài khoản Streamlit liên kết với github của bạn
- Ấn vào `Create App`
- Điền thông tin vào các ô tương ứng:
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/streamlit1.png" width="100%" />
</p>  

- Chọn vào Advanced Settings và đưa các biến môi trường của bạn vào đây:
<p align="center">
  <img src="https://github.com/NKDuyennn/llm_rag/blob/nkduyen/image/streamlit2.png" width="100%" />
</p>  

### **Bước 3:**
Deploy và bạn đã host thành công, bạn có thể sử dụng Streamlit App theo đường link dạng `<tên miền của bạn>.streamlit.app`

### **Lưu ý**
Vì dùng free, không phải trả phí nên tài nguyên Streamlit cấp cho ít, nên sử dụng embedding model với API-Key. 

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
