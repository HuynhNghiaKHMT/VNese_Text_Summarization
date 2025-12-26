import re

# Định nghĩa dải ký tự hoa tiếng Việt
_UPPER_VN = r"A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỵ"

def split_sentence(text):
    if not isinstance(text, str): 
        text = str(text)
    
    # Chuẩn hóa các loại dấu ngoặc và khoảng trắng thừa
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r'[ \t]+', ' ', text) # Thu gọn khoảng trắng ngang

    # Logic Regex mới:
    # (?<=[^...]) : Positive Lookbehind - Phía trước KHÔNG PHẢI là chữ hoa
    # (?=\s+[...]) : Positive Lookahead - Phía sau PHẢI là khoảng trắng + chữ hoa
    
    # 1) Tách tại dấu chấm .
    text = re.sub(
        rf'(?<=[^{_UPPER_VN}])\.(?:["\s])*?(?=\s+[{_UPPER_VN}])',
        '.<SPLIT>',
        text
    )

    # 2) Tách tại dấu ba chấm (cả dạng 3 dấu chấm ... và ký tự unicode …)
    text = re.sub(
        rf'(?<=[^{_UPPER_VN}])(?:\.{3}|…)(?:["\s])*?(?=\s+[{_UPPER_VN}])',
        '…<SPLIT>',
        text
    )

    # 3) Tách tại dấu xuống dòng
    text = re.sub(r'\n+', '<SPLIT>', text)

    # Gom kết quả
    sentences = [s.strip() for s in text.split('<SPLIT>') if s.strip()]
    filtered_sentences = [s for s in sentences if len(s.split()) >= 3]

    return filtered_sentences