import os
import torch
from peft import PeftModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_mbart(base_model, adapter_path=None):
    """
    Hàm này chỉ gọi 1 lần duy nhất để đưa mô hình lên GPU.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- Đang khởi tạo mô hình trên: {device} ---")
    
    try:
        # 1. Load Tokenizer
        tokenizer = MBart50TokenizerFast.from_pretrained(base_model, src_lang="vi_VN", tgt_lang="vi_VN")
        
        # 2. Load Base Model
        base = MBartForConditionalGeneration.from_pretrained(base_model)
        
        # 3. Load Adapter nếu có
        if adapter_path and os.path.exists(adapter_path):
            print(f"--- Đang nạp Adapter từ: {adapter_path} ---")
            model = PeftModel.from_pretrained(base, adapter_path)
        else:
            print("--- Không tìm thấy Adapter, sử dụng Base Model ---")
            model = base
            
        model = model.to(device)
        model.eval() # Chuyển sang chế độ đánh giá
        
        return model, tokenizer, device

    except Exception as e:
        print(f"[LỖI KHI LOAD MODEL]: {e}")
        return None, None, None
    
def chunk_text(text, chunk_size=1024, overlap=128, tokenizer=None):
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

def mbart_summarizer(model, tokenizer, device, extractive_text):
    
    # 4. Chia chunk và tóm tắt
    chunks = chunk_text(extractive_text, tokenizer=tokenizer)
    abstractive_summary = []
    
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