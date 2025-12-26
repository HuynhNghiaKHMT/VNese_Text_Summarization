from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances_argmin_min
import math

def find_optimal_k(embeddings, num_sentences):
    """
    Tìm K tối ưu nhưng bị ràng buộc bởi tỷ lệ:
    Tối thiểu 20%, Tối đa 50% số câu của tài liệu.
    """

    # 1. Tính toán biên giới hạn cho K
    # Sử dụng math.ceil để đảm bảo luôn lấy ít nhất 1 câu
    min_k = max(2, math.ceil(num_sentences * 0.2))
    max_k = max(2, math.ceil(num_sentences * 0.5))
    
    # Trường hợp đặc biệt: Tài liệu quá ngắn
    if num_sentences <= 2:
        return 1
    if min_k >= max_k:
        return min_k

    # 2. Tính toán biến thiên (Distortions) để tìm Elbow
    distortions = []
    # Chỉ tìm kiếm trong phạm vi từ 1 đến max_k để tiết kiệm tài nguyên
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)
    
    # 3. Tìm điểm Elbow trong phạm vi cho phép
    optimal_k = min_k # Mặc định khởi tạo là min_k
    
    if len(distortions) >= 3:
        kl = KneeLocator(
            K_range, 
            distortions, 
            curve="convex", 
            direction="decreasing"
        )
        if kl.elbow:
            optimal_k = kl.elbow
            
    # 4. Ép buộc K nằm trong đoạn [min_k, max_k]
    final_k = max(min_k, min(max_k, optimal_k))
    
    return final_k

def kmeans_summarizer(sentences, embeddings):

    # BƯỚC 1: TÌM K TỐI ƯU (ELBOW)
    if len(sentences) > 3:
        k_optimal = find_optimal_k(embeddings, min(len(sentences) - 1, 10))
    else:
        k_optimal = 1

    # BƯỚC 2: PHÂN CỤM VÀ TRÍCH XUẤT
    # Tận dụng logic tìm câu gần tâm cụm nhất
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    
    selected_indices = []
    for i in range(k_optimal):
        # Lấy chỉ số các câu thuộc cụm i
        cluster_indices = [idx for idx, label in enumerate(kmeans.labels_) if label == i]
        cluster_embs = embeddings[cluster_indices]
        
        # Tìm câu gần tâm nhất trong cụm đó
        closest, _ = pairwise_distances_argmin_min(centroids[i].reshape(1, -1), cluster_embs)
        selected_indices.append(cluster_indices[closest[0]])
    
    # BƯỚC : Sắp xếp lại theo thứ tự xuất hiện gốc
    selected_indices.sort()
    # final_summary = " ".join([sentences[idx] for idx in selected_indices])
    final_summary = [sentences[idx] for idx in selected_indices]
    
    return selected_indices, final_summary