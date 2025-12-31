import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from summary.kmean import kmeans_summarizer
from summary.lexrank import lexrank_summarizer
from utils.split_sentence import split_sentence
from summary.mbart50_fb import mbart_summarizer, load_mbart
from summary.bartpho_vinai import bartpho_summarizer, load_bartpho

# --- LOAD CẤU HÌNH TỪ FILE .ENV ---
load_dotenv()

# Lấy các đường dẫn model (nếu không tìm thấy sẽ dùng giá trị mặc định)
SBERT_FINETUNE = os.getenv("sbert_model_finetune")
SBERT_BASE = os.getenv("sbert_model_base")

MBART_FINETUNE = os.getenv("mbart_model_finetune")
MBART_BASE = os.getenv("mbart_model_base")

BARTPHO_FINETUNE = os.getenv("bartpho_model_finetune")
BARTPHO_BASE = os.getenv("bartpho_model_base")

# 1. Cấu hình trang
st.set_page_config(layout="wide", page_title="Vietnamese Text Summarization")

# 2. CSS để đồng nhất giao diện
st.markdown("""
    <style>
            
    /* Căn giữa tiêu đề H1 */
    .main-title {
        font-size: 50px !important;
        text-align: center;
        font-weight: bold;
    }
            
    .sub-title {
        font-size: 25px !important;
    }
    
    /* Thu hẹp khoảng cách giữa các thành phần trong Sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem; /* Giảm gap mặc định (thường là 1rem) */
    }
    
  
    /* Đồng nhất màu nền và khung cho cả ô nhập và hộp highlight */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        line-height: 1.8 !important;
    }

    .highlight-box {
        text-align: justify; 
        height: 500px; 
        overflow-y: auto; 
        border: 1px solid #d1d5db; 
        padding: 20px; 
        border-radius: 8px;
        background-color: #ffffff;
        line-height: 1.8;
        font-size: 16px;
        color: #000000;
    }
    
    /* Tùy chỉnh nút bấm chính */
    .stButton > button {
        border-radius: 5px;
        height: 3.5em;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        font-size: 18px;
    }

    /* Khoảng cách giữa các phần */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Khởi tạo trạng thái ứng dụng
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.highlighted_html = ""
    st.session_state.extractive_res = ""
    st.session_state.abstractive_res = ""

# --- SIDEBAR CẤU HÌNH ---
st.sidebar.header("⚙️ Cấu hình")
st.sidebar.markdown("---")

# Hiển thị dạng tích chọn (checkbox) và luôn hiện tham số
do_extractive = st.sidebar.checkbox("Tóm tắt trích xuất (Extractive)", value=True)
do_abstractive = st.sidebar.checkbox("Tóm tắt trừu tượng (Abstractive)", value=False)

st.sidebar.markdown("---")
extractive_method = st.sidebar.radio("Tóm tắt trích đoạn:", ["K-Means", "LexRank"])
extraction_ratio = st.sidebar.slider("Tỷ lệ trích đoạn (%)", 5, 50, 10, step=5) / 100

# st.sidebar.markdown("---")
abstractive_method = st.sidebar.radio("Tóm tắt trừu tượng:", ["fb/mbart50", "vinai/bartpho"])

st.sidebar.markdown("---")
# Nút Reset chiếm hết chiều ngang và chữ nằm giữa
if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.processed = False
    st.session_state.highlighted_html = ""
    st.session_state.extractive_res = ""
    st.session_state.abstractive_res = ""
    st.rerun()

# --- LOAD MÔ HÌNH ---
@st.cache_resource
def load_sbert_models():
    try:
        return SentenceTransformer(SBERT_FINETUNE)
    except:
        return SentenceTransformer(SBERT_BASE)

sbert_model = load_sbert_models()

@st.cache_resource
def load_mbart_models():
    try:
        return load_mbart(MBART_BASE, MBART_FINETUNE)
    except:
        return None, None, None

@st.cache_resource
def load_bartpho_models():
    try:
        return load_bartpho(BARTPHO_BASE,BARTPHO_FINETUNE)
    except:
        return None, None, None
    
MODEL1, TOKENIZER1, DEVICE1 = load_mbart_models()
MODEL2, TOKENIZER2, DEVICE2 = load_bartpho_models()

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1 class="main-title">Vietnamese Text Summarization System</h1>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<h2 class="sub-title">📝 Văn bản gốc</h2>', unsafe_allow_html=True)

    if not st.session_state.processed:
        input_text = st.text_area(
            "input_label", 
            height=500, 
            placeholder="Dán nội dung cần tóm tắt vào đây...", 
            label_visibility="collapsed"
        )
    else:
        st.markdown(f'<div class="highlight-box">{st.session_state.highlighted_html}</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<h2 class="sub-title">📝 Văn bản tóm tắt</h2>', unsafe_allow_html=True)

    st.markdown("**1. Kết quả trích đoạn (Extractive):**")
    ex_container = st.container(height=200)
    with ex_container:
        # Tạo placeholder để có thể cập nhật nội dung bên trong logic xử lý
        ex_placeholder = st.empty()
        if st.session_state.processed and st.session_state.extractive_res:
            ex_placeholder.write(st.session_state.extractive_res)
        else:
            ex_placeholder.info("Kết quả tóm tắt trích đoạn sẽ hiển thị tại đây.")

    st.markdown("**2. Kết quả trừu tượng (Abstractive):**")
    ab_container = st.container(height=200)
    with ab_container:
        # Tạo placeholder để có thể cập nhật nội dung bên trong logic xử lý
        ab_placeholder = st.empty()
        if st.session_state.processed and st.session_state.abstractive_res:
            ab_placeholder.write(st.session_state.abstractive_res)
        else:
            ab_placeholder.info("Kết quả tóm tắt trừu tượng sẽ hiển thị tại đây.")

# --- NÚT BẤM ---
st.write("---") 
_, btn_col, _ = st.columns([0.7, 0.6, 0.7]) 
with btn_col:
    summarize_btn = st.button("SUMMARIZATION", type="primary", use_container_width=True)

# --- LOGIC XỬ LÝ TẬP TRUNG ---
if summarize_btn and not st.session_state.processed:
    if not (do_extractive or do_abstractive):
        st.error("Vui lòng tích chọn ít nhất một kiểu tóm tắt!")
    elif not input_text.strip():
        st.error("Bạn chưa nhập văn bản!")
    else:
        # 1. Tách câu văn bản gốc
        sentences = split_sentence(input_text)
        indices = [] 
        
        # 2. Xử lý Extractive
        if do_extractive:
            # Hiển thị thông tin đang xử lý ngay tại ô kết quả trích đoạn
            ex_placeholder.warning("⏳ Đang thực hiện tóm tắt trích đoạn...")
            
            embeddings = sbert_model.encode(sentences)
            print("--- Đang thực hiện Extractive Summarization ---")
            if extractive_method == "K-Means":
                indices, summaries = kmeans_summarizer(sentences, embeddings, extraction_ratio)
            else:
                indices, summaries = lexrank_summarizer(sentences, embeddings, extraction_ratio)
            st.session_state.extractive_res = " ".join(summaries)
            print("--- Hoàn thành Extractive Summarization ---")
            
            # Cập nhật trạng thái sau khi xong để chuẩn bị cho bước abstractive nếu có
            ex_placeholder.success("✅ Đã hoàn thành trích đoạn!")
        
        # 3. Xử lý Abstractive
        if do_abstractive:
            # Hiển thị thông tin đang xử lý ngay tại ô kết quả trừu tượng
            ab_placeholder.warning("⏳ Đang thực hiện tóm tắt trừu tượng...")
            
            if do_extractive:
                # Chế độ Hybrid: Dùng kết quả extractive làm đầu vào
                print("--- Đang thực hiện Hybrid Summarization ---")
                
                if abstractive_method == "fb/mbart50":
                    st.session_state.abstractive_res = mbart_summarizer(MODEL1, TOKENIZER1, DEVICE1, st.session_state.extractive_res)
                else:
                    st.session_state.abstractive_res = bartpho_summarizer(MODEL2, TOKENIZER2, DEVICE2, st.session_state.extractive_res)
                
                print("--- Hoàn thành Hybrid Summarization ---")
            else:
                # Chế độ Abstractive thuần túy
                print("--- Đang thực hiện Abstractive Summarization ---")
                if abstractive_method == "fb/mbart50":
                    st.session_state.abstractive_res = mbart_summarizer(MODEL1, TOKENIZER1, DEVICE1, input_text)
                else:
                    st.session_state.abstractive_res = bartpho_summarizer(MODEL2, TOKENIZER2, DEVICE2, input_text)

                print("--- Hoàn thành Abstractive Summarization ---")
            
            ab_placeholder.success("✅ Đã hoàn thành trừu tượng!")

        # 4. Tạo HTML hiển thị văn bản gốc
        html_res = []
        for i, s in enumerate(sentences):
            if i in indices:
                html_res.append(f'<span style="background-color: #90ee90; color: black; border-radius: 3px; padding: 0 2px;">{s}</span>')
            else:
                html_res.append(f'<span>{s}</span>')
        
        st.session_state.highlighted_html = " ".join(html_res)
        st.session_state.processed = True
        # Rerun để hiển thị văn bản cuối cùng vào container
        st.rerun()