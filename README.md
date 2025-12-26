# CS221 - Vietnamese Text Summarization  (Streamlit)
Dự án này triển khai một **Hệ thống tóm tắt văn bản tiếng Viêt**, sử dụng Streamlit làm giao diện tương tác. Hệ thống kết hợp khả năng trích xuất trích đoạn (extractive) và trích xuất trừu tượng (abstractive) *, đồng thời tối ưu hóa trích xuất thông tin bằng các kỹ thuật như **TF-IDF**, **BM25** và **Vector Search**. Mục tiêu là cung cấp các bản tóm tắt chính xác và ngắn gọn từ các văn bản tiếng Việt dài, hỗ trợ người dùng trong việc nắm bắt thông tin nhanh chóng và hiệu quả.

## 📦 Công nghệ và Thư viện sử dụng

- **Dataset**: `OpenHust` tại [Huggingface](https://huggingface.co/datasets/OpenHust/vietnamese-summarization)
- **Embedding Model**: `multilingual-e5-large` (mô hình [finetune](https://huggingface.co/intfloat/multilingual-e5-large)) để tạo vector ngữ nghĩa cho các câu trong đoạn văn bản.
- **Extractive summarization model**: `LexRank` và `Kmeans` để trích xuất các câu quan trọng từ đoạn văn bản gốc.
- **Abstractive summarization model**: `mbart-large-50` (mô hình [finetune](https://huggingface.co/facebook/mbart-large-50)) và `bartpho-word` (mô hình [finetune](https://huggingface.co/vinai/bartpho-word)) để tạo tóm tắt trừu tượng cho các đoạn văn bản.
- **Apllication**: `Streamlit` để cung cấp một ứng dụng tương tác.

## 📂 Cấu trúc thư mục
```bash
VNese_Text_Summarization
├── application/
    └── app.py
├── assets/
├── models/
    ├── mtlge5-transformerts-default-v1
    ├── bartpho-transformers-default-v1/
    ├── mbart-transformers-default-v1/
    ├── vncorenlp/
    └── vietnamese-stopwords.txt
├── notebooks/
├── summary/
    ├── textrank.py
    ├── lexrank.py
    ├── kmeans.py
    ├── bartpho_vinai.py
    └── mbart50_fb.py
├── utils/
├── .env
├── .gitignore
├── requirements.txt
└── README.md

```
## 🚀 Cài đặt và sử dụng

### 1. Clone Repository

```bash
git clone https://github.com/HuynhNghiaKHMT/VNese_Text_Summarization.git
cd VNese_Text_Summarization
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
venv\Scripts\activate 
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

## 🏃 Demo
### 1. Chạy Demo ByteTrack cơ bản
```bash
python VNese_Text_Summarization.py
```
Lệnh này sẽ chạy demo chat trực tiếp trên máy tính của bạn với câu hỏi mẫu được cung cấp sẵn. Bạn sẽ thấy cách hệ thống tóm tắt dựa trên văn bản đầu vào bằng nhiều phương pháp. Hoặc bạn có thể sử dụng file `VNese_Text_Summarization.ipynb` để thử nghiệm.

### 2. Chạy Demo với ứng dụng Streamlit
```bash
python -m streamlit run application/app.py
```
Lệnh này sẽ chạy demo tóm tắt trực tiếp trên Streamlit app và hỗ trợ điều chỉnh các tùy chọn khác nhau. Mở trình duyệt và truy cập vào địa chỉ http://localhost:8501 để sử dụng ứng dụng.

Các Tính năng RAG Tùy chỉnh (Trong Sidebar)
| Tham số | Phạm vi | Mục đích |
| :--- | :--- | :--- |
| **Phương pháp tóm tắt** | Extractive/ Absstractive/ Hybird | Thử nghiệm nhiều phương pháp tóm tắt khác nhau. |
| **Tóm tắt trích đoạn** | LexRank/ Kmeans | Thử nghiệm nhiều phương pháp trích xuất khác nhau. |
| **Tỷ lệ trích đoạn** | 5% - 100% | Điều chỉnh số lượng câu được trích xuất từ văn bản gốc. |
| **Tóm tắt trừu tượng** | mbart50/ bartpho| Thử nghiệm nhiều mô hình tóm tắt khác nhau. |


## 🎞️ Video Demo
Dưới đây là một đoạn video/GIF ngắn minh họa hoạt động của ứng dụng VNese_Text_Summarization mà mình đã triển khai:

<!-- <img src="assets/demo.mp4" width="100%"> -->
https://github.com/user-attachments/assets/2a0fe8ad-4026-4186-b8a1-d2caba5008b0




## Reference
Dưới đây là các nghiên cứu và mô hình chính được sử dụng trong hệ thống:
1. BERT-VBD: Vietnamese Multi-Document Summarization Framework (2024). Tuan-Cuong Vuong, Trang Mai Xuan, Thien Van Luong. arXiv:2409.12134
2. Multilingual E5 Text Embeddings: A Technical Report (2024). Wang, Liang, et al. arXiv:2402.05672
3. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (2019). Mike Lewis, et al. arXiv:1910.13461
4. BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese (2022). Nguyen Luong Tran, Duong Minh Le, Dat Quoc Nguyen. arXiv:2109.09701
