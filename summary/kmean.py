import math
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

def find_optimal_k(embeddings, num_sentences):
    """
    Tìm K tối ưu nhưng bị ràng buộc bởi tỷ lệ:
    Tối thiểu 20%, Tối đa 50% số câu của tài liệu.
    """

    # 1. Tính toán biên giới hạn cho K
    # Sử dụng math.ceil để đảm bảo luôn lấy ít nhất 1 câu
    min_k = max(1, math.ceil(num_sentences * 0.2))
    max_k = max(1, math.ceil(num_sentences * 0.5))
    
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

def kmeans_summarizer(sentences, embeddings, extraction_ratio=0.2):

    # BƯỚC 1: TÌM K TỐI ƯU (ELBOW)
    k_optimal = find_optimal_k(embeddings, len(sentences))

    # BƯỚC 2: PHÂN CỤM VÀ TRÍCH XUẤT
    # Tận dụng logic tìm câu gần tâm cụm nhất
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    
    selected_indices = []
    for i in range(k_optimal):

        # Cách 1: Tìm câu gần tâm cụm nhất
        # Lấy chỉ số các câu thuộc cụm i
        # cluster_indices = [idx for idx, label in enumerate(kmeans.labels_) if label == i]
        # cluster_embs = embeddings[cluster_indices]
        
        # Tìm câu gần tâm nhất trong cụm đó
        # closest, _ = pairwise_distances_argmin_min(centroids[i].reshape(1, -1), cluster_embs)
        # selected_indices.append(cluster_indices[closest[0]])

        # Cách 2: Lấy phần trăm câu theo extraction_ratio trong mỗi cụm
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_embs = embeddings[cluster_indices]
        
        # 2. Số câu cần trích xuất trong cụm này
        num_to_extract = max(1, math.ceil(len(cluster_indices) * extraction_ratio))
        
        # 3. Tính khoảng cách từ tâm đến TẤT CẢ các câu trong cụm\, distances có dạng (1, len(cluster_indices))
        distances = pairwise_distances(centroids[i].reshape(1, -1), cluster_embs, metric='euclidean').flatten()
        
        # 4. Sắp xếp lấy num_to_extract câu gần tâm nhất
        closest_indices_in_cluster = np.argsort(distances)[:num_to_extract]
        
        # 5. Đưa các chỉ số gốc vào danh sách chọn
        for idx in closest_indices_in_cluster:
            selected_indices.append(cluster_indices[idx])
    
    # BƯỚC3: Sắp xếp lại theo thứ tự xuất hiện gốc
    selected_indices.sort()
    # final_summary = " ".join([sentences[idx] for idx in selected_indices])
    final_summary = [sentences[idx] for idx in selected_indices]
    
    return selected_indices, final_summary