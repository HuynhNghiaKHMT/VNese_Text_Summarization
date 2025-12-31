import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def build_similarity_matrix_lexrank(embeddings, threshold=0.5):
    if len(embeddings) <= 1: return None
    similarity_matrix = cosine_similarity(embeddings)
    graph_matrix = np.zeros_like(similarity_matrix, dtype=float)
    
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j and similarity_matrix[i, j] > threshold:
                graph_matrix[i, j] = similarity_matrix[i, j]
    return graph_matrix

def lexrank_summarizer(sentences, embeddings, extraction_ratio=0.2):

    # BƯỚC 1: XÁC ĐỊNH SỐ LƯỢNG CÂU PHÙ HỢP
    # LexRank thường dùng tỷ lệ trích xuất (ví dụ 20% số câu gốc)
    num_to_extract = max(1, int(len(sentences) * extraction_ratio))

    # BƯỚC 2: XÂY DỰNG ĐỒ THỊ VÀ TRÍCH XUẤT
    # 4.1. Xây dựng ma trận tương đồng (dùng TAU_THRESHOLD = 0.1)
    graph_matrix = build_similarity_matrix_lexrank(embeddings, threshold=0.1)
    
    # 4.2. Tính điểm PageRank
    scores = pagerank(graph_matrix)
    
    # 4.3. Chọn các câu điểm cao nhất
    ranking_data = []
    for i in range(len(sentences)):
        ranking_data.append({
            'index': i,
            'text': sentences[i],
            'score': scores[i]
        })
    
    # BƯỚC 3: Sắp xếp theo điểm giảm dần để lấy top, sau đó sắp xếp lại theo index để giữ trình tự văn bản
    top_n = sorted(ranking_data, key=lambda x: x['score'], reverse=True)[:num_to_extract]
    top_n_ordered = sorted(top_n, key=lambda x: x['index'])

    # Lấy ra danh sách các chỉ số index
    indices = [item['index'] for item in top_n_ordered]
    
    # final_summary = " ".join([item['text'] for item in top_n_ordered])
    final_summary = [item['text'] for item in top_n_ordered]
    
    return indices, final_summary