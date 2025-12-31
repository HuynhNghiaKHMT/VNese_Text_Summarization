import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_bartpho(base_model, adapter_path=None):
    print(f"--- Đang nạp Adapter từ: {adapter_path} ---")

    model_name = adapter_path if adapter_path else base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device

def chunk_text(text, tokenizer, chunk_size=1024, overlap=128):
    """Chia văn bản thành các đoạn nhỏ để tránh tràn context 1024 của BartPho"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return chunks

def bartpho_summarizer(model, tokenizer, device, text):
    chunks = chunk_text(text, tokenizer)
    summaries = []
    
    for chunk in chunks:
        inputs = tokenizer(
            chunk, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        ).to(device)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=True
        )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    return " ".join(summaries)