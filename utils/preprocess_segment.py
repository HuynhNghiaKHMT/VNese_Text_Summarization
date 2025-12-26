import re
import py_vncorenlp

filename = "models/vietnamese-stopwords.txt"  # Điều chỉnh đường dẫn nếu cần

# Đọc stopwords từ file txt
with open(filename, 'r', encoding='utf-8') as f:
    list_stopwords = f.read().splitlines()

# Đọc py_vncorenlp từ mô hình
nlp = py_vncorenlp.VnCoreNLP(save_dir='models/vncorenlp')

class StopwordRemover:
    def __init__(self, stopwords):
        self.stopwords = set(stopwords)

    def remove_stopwords(self, text):
        pre_text = []
        words = text.split()
        for word in words:
            if word.lower() not in self.stopwords:
                pre_text.append(word)
        return ' '.join(pre_text)
        
class VNPreprocessor:
    def __init__(self, nlp):
        self.nlp = nlp

    def preprocess(self, text):
        segmented_sentences = self.nlp.word_segment(text)
        processed_text = ' '.join(segmented_sentences)
        return processed_text
        
preprocessor = VNPreprocessor(nlp)
stopword_remover = StopwordRemover(list_stopwords)

def normalize_text(text):
    text = text.lower()

    # Chỉ loại bỏ các ký tự vô nghĩa (tạm thời giữ lại dấu câu quan trọng)
    # Loại bỏ các ký tự đặc biệt, số và chữ cái tiếng Việt.
    # [^a-zA-Z0-9\s] là tìm kiếm bất kỳ ký tự nào KHÔNG phải chữ cái, số, hoặc khoảng trắng.
    # Trong môi trường tiếng Việt, chúng ta cần giữ lại dải ký tự À-ỹ.
    
    # Loại bỏ các ký tự không phải chữ cái, số, hoặc khoảng trắng, dấu câu
    # Giữ lại các ký tự tiếng Việt (À-ỹ) và dấu câu (. , ! ?)
    # Ký tự % ; : và các ký tự đặc biệt khác sẽ bị loại bỏ.
    text = re.sub(r"[^a-zA-Z0-9À-ỹ\s.,!?]", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_and_segment(text_raw):
    """Áp dụng normalize, tách từ, và loại bỏ stopword."""
    # 1. Normalize (chuyển chữ thường, loại bỏ ký tự đặc biệt)
    normalized_text = normalize_text(text_raw)
    
    # 2. Tách từ
    segmented_text = preprocessor.preprocess(normalized_text)
    
    # 3. Loại bỏ Stopword
    final_text = stopword_remover.remove_stopwords(segmented_text)
    # final_text = segmented_text
    
    return final_text