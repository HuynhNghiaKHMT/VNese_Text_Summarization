import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess_segment import preprocess_and_segment

def pagerank(graph_matrix, d=0.85, max_iter=100, tol=1e-4):
    N = graph_matrix.shape[0]
    if N == 0: return np.array([])
    
    pr_scores = np.ones(N) / N
    sum_outgoing = np.sum(graph_matrix, axis=1, keepdims=True)
    M = graph_matrix.T / (sum_outgoing.T + 1e-8)
    
    for _ in range(max_iter):
        new_pr = (1 - d) / N + d * M.dot(pr_scores)
        if np.linalg.norm(new_pr - pr_scores) < tol:
            break
        pr_scores = new_pr
    return pr_scores

def calculate_similarity(sent1_tokens, sent2_tokens):
    """Công thức tính Similarity của TextRank nguyên bản."""
    common_words = set(sent1_tokens) & set(sent2_tokens)
    overlap_count = len(common_words)
    
    if overlap_count == 0:
        return 0
    
    # Tránh lỗi log(0) hoặc log(1) dẫn đến chia cho 0
    log_l1 = math.log(len(sent1_tokens)) if len(sent1_tokens) > 1 else 0.1
    log_l2 = math.log(len(sent2_tokens)) if len(sent2_tokens) > 1 else 0.1
    
    return overlap_count / (log_l1 + log_l2)

def build_word_overlap_matrix(segmented_sentences):
    """Xây dựng ma trận tương đồng dựa trên word overlap."""
    N = len(segmented_sentences)
    graph_matrix = np.zeros((N, N))
    
    # Chuyển các câu thành list các từ để so sánh
    token_lists = [s.split() for s in segmented_sentences]
    
    for i in range(N):
        for j in range(i + 1, N):
            sim = calculate_similarity(token_lists[i], token_lists[j])
            graph_matrix[i, j] = sim
            graph_matrix[j, i] = sim  # Đồ thị vô hướng
            
    return graph_matrix

def build_weights_matrix(embeddings):
    """
    TextRank giữ nguyên trọng số tương đồng (không dùng threshold)
    để tạo đồ thị có trọng số toàn phần.
    """
    if len(embeddings) <= 1:
        return None
    # Tính ma trận tương đồng Cosine
    sim_matrix = cosine_similarity(embeddings)
    
    # Loại bỏ tự tương đồng (đường chéo chính = 0)
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix

def extract_summary_textrank(df_temp, num_extract):
    if len(df_temp) <= num_extract:
        return " ".join(df_temp['text_raw'].tolist())
    
    embeddings = np.array(df_temp['embedding'].tolist())
    
    # 1. Xây dựng ma trận TextRank (không threshold)
    graph_matrix = build_weights_matrix(embeddings)
    
    if graph_matrix is None:
        return " ".join(df_temp['text_raw'].head(num_extract).tolist())
        
    # 2. Tính điểm PageRank (sử dụng lại hàm pagerank bạn đã có)
    scores = pagerank(graph_matrix)
    df_temp = df_temp.copy()
    df_temp['score'] = scores
    
    # 3. Lấy N câu điểm cao nhất và sắp xếp theo thứ tự xuất hiện
    top_sentences = df_temp.sort_values(by='score', ascending=False).head(num_extract)
    summary = " ".join(top_sentences.sort_values(by='sent_id_in_cluster')['text_raw'].tolist())
    
    return summary

def overlap_textrank_summarizer(sentences_raw, num_extract):
    if len(sentences_raw) <= num_extract:
        return " ".join(sentences_raw)
    
    # BƯỚC A: Tiền xử lý (Normalize, Segment, Stopwords)
    # Đây là bước quan trọng để TextRank truyền thống không bị nhiễu bởi 'là', 'của',...
    segmented_sentences = [preprocess_and_segment(s) for s in sentences_raw]
    
    # BƯỚC B: Xây dựng ma trận dựa trên Word Overlap
    graph_matrix = build_word_overlap_matrix(segmented_sentences)
    
    # BƯỚC C: PageRank
    scores = pagerank(graph_matrix)
    
    # BƯỚC D: Xếp hạng
    ranking_data = []
    for i in range(len(sentences_raw)):
        ranking_data.append({
            'text': sentences_raw[i],
            'score': scores[i],
            'order': i
        })
        
    top_n = sorted(ranking_data, key=lambda x: x['score'], reverse=True)[:num_extract]
    # Sắp xếp lại theo thứ tự xuất hiện ban đầu
    top_n_ordered = sorted(top_n, key=lambda x: x['order'])
    
    return " ".join([item['text'] for item in top_n_ordered])

def tfidf_textrank_summarizer(sentences_raw, num_extract):
    if len(sentences_raw) <= num_extract:
        return " ".join(sentences_raw)

    # BƯỚC 1: Tiền xử lý (Sử dụng hàm của bạn để tách từ tiếng Việt)
    # Ví dụ: "Hôm nay tôi đi học" -> "Hôm_nay tôi đi học"
    segmented_sentences = [preprocess_and_segment(s) for s in sentences_raw]

    # BƯỚC 2: Tạo ma trận TF-IDF cho các câu
    # Mỗi câu được coi là một "văn bản" nhỏ
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(segmented_sentences)

    # BƯỚC 3: Xây dựng ma trận tương đồng (Graph Matrix) bằng Cosine Similarity
    # Công thức: sim(Si, Sj) = (Vi . Vj) / (||Vi|| * ||Vj||)
    graph_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Loại bỏ tự tương đồng (đường chéo chính = 0)
    np.fill_diagonal(graph_matrix, 0)

    # BƯỚC 4: PageRank (Sử dụng hàm pagerank bạn đã có)
    scores = pagerank(graph_matrix)

    # BƯỚC 5: Xếp hạng và trích xuất
    ranking_data = []
    for i in range(len(sentences_raw)):
        ranking_data.append({
            'text': sentences_raw[i],
            'score': scores[i],
            'order': i
        })
        
    # Lấy N câu tốt nhất
    top_n = sorted(ranking_data, key=lambda x: x['score'], reverse=True)[:num_extract]
    # Sắp xếp lại theo thứ tự xuất hiện gốc để đảm bảo tính mạch lạc
    top_n_ordered = sorted(top_n, key=lambda x: x['order'])
    
    return " ".join([item['text'] for item in top_n_ordered])