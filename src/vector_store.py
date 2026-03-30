
import faiss
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, dimension=512, use_gpu=True):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = None
        self.id_mapping = None
        self.res = None
        
        if use_gpu:
            self.res = faiss.StandardGpuResources()
    
    def build_index(self, image_vectors, text_vectors):
        """
        构建联合索引
        image_vectors: [N, 512]
        text_vectors: [N, 512]
        """
        # 合并向量
        combined = np.vstack([image_vectors, text_vectors])
        n = len(image_vectors)
        
        # 创建ID映射
        id_mapping = {
            "id_to_type": {},
            "id_to_pair_idx": {},
            "num_images": n,
            "num_texts": n
        }
        
        for i in range(n):
            id_mapping["id_to_type"][str(i)] = "image"
            id_mapping["id_to_pair_idx"][str(i)] = i
            id_mapping["id_to_type"][str(n + i)] = "text"
            id_mapping["id_to_pair_idx"][str(n + i)] = i
        
        # 创建索引
        index_cpu = faiss.IndexFlatL2(self.dimension)
        
        if self.use_gpu and self.res:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, index_cpu)
        else:
            self.index = index_cpu
        
        # 添加向量
        self.index.add(combined.astype('float32'))
        self.id_mapping = id_mapping
        
        print(f"索引构建完成: {self.index.ntotal} 条向量")
        return id_mapping
    
    def search(self, query_vector, k=5):
        """
        检索相似向量
        query_vector: [1, 512] 或 [512,]
        返回: (distances, indices)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return distances[0], indices[0]
    
    def get_item_info(self, idx):
        """根据ID获取类型和pair索引"""
        idx_str = str(int(idx))
        item_type = self.id_mapping["id_to_type"].get(idx_str)
        pair_idx = self.id_mapping["id_to_pair_idx"].get(idx_str)
        return item_type, pair_idx
    
    def save(self, save_dir):
        """保存索引和映射"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 索引转CPU保存
        index_cpu = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(index_cpu, os.path.join(save_dir, "index.faiss"))
        
        # 保存映射
        with open(os.path.join(save_dir, "mapping.json"), "w") as f:
            json.dump(self.id_mapping, f)
        
        print(f"保存完成: {save_dir}")
    
    def load(self, save_dir):
        """加载索引和映射"""
        # 加载索引
        index_cpu = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        
        if self.use_gpu and self.res:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, index_cpu)
        else:
            self.index = index_cpu
        
        # 加载映射
        with open(os.path.join(save_dir, "mapping.json"), "r") as f:
            self.id_mapping = json.load(f)
        
        print(f"加载完成: {self.index.ntotal} 条向量")
