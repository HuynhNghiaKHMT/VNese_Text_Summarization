import os
import torch
from peft import PeftModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# def preprocessing(extractive, flag=False):
#     # Đảm bảo đầu vào là String để tokenizer xử lý được
#     if isinstance(extractive, list):
#         return " ".join(extractive)
#     return extractive

def chunk_text(text, chunk_size=1022, overlap=128, tokenizer=None):
    if not tokenizer:
        return [text]
    
    # Mã hóa text thành tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    # Chia nhỏ nếu văn bản quá dài
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text_decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text_decoded)
        
        if i + chunk_size >= len(tokens):
            break
            
    return chunks if chunks else [text]

def experiment(base_model, model_path, extractive):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        # 1. Load Tokenizer từ model gốc
        tokenizer = MBart50TokenizerFast.from_pretrained(base_model, src_lang="vi_VN", tgt_lang="vi_VN")
        
        # 2. Load Base Model gốc
        base_model = MBartForConditionalGeneration.from_pretrained(base_model)
        
        # 3. Load Adapter (cái mà bạn đã train) đè lên Base Model
        if model_path and os.path.exists(model_path):
            print(f"Đang nạp Adapter từ: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = base_model
            
        model = model.to(device)
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG]: {e}")
        return "Lỗi: Không thể khởi tạo mô hình."
    
    # 4. Tiền xử lý dữ liệu (Chuyển List từ K-Means thành String)
    # extractive_text = preprocessing(extractive, False)
    extractive_text = extractive

    # 5. Chia chunk và tóm tắt
    chunks = chunk_text(extractive_text, tokenizer=tokenizer)
    abstractive_summary = []
    
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(
                chunk, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(device)
            
            summary_ids = model.generate(
                **inputs,
                max_length=128,
                min_length=15,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True
            )
            
            result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            abstractive_summary.append(result)

    final_summary = " ".join(abstractive_summary)
    return final_summary