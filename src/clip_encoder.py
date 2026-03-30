
import torch
import open_clip
import numpy as np
from PIL import Image
import os

class CLIPEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"CLIP加载完成: {model_name}, 设备: {self.device}")
    
    def encode_image(self, image_path):
        """编码单张图像"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def encode_text(self, text):
        """编码文本"""
        text = str(text)
        tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def encode_batch(self, pairs, data_dir="../data/iu_xray", batch_size=32):
        """
        批量编码图像和文本
        
        Args:
            pairs: image_report_pairs.json的数据列表
            data_dir: 数据根目录
            batch_size: 批次大小
        
        Returns:
            image_vectors: [N, 512] 图像向量数组
            text_vectors: [N, 512] 文本向量数组
            valid_pairs: 成功编码的pairs子集
        """
        import torch
        from tqdm import tqdm
        
        image_vectors = []
        text_vectors = []
        valid_pairs = []
        
        print(f"开始批量编码，共{len(pairs)}条，batch_size={batch_size}...")
        
        for i in tqdm(range(0, len(pairs), batch_size)):
            batch = pairs[i:i+batch_size]
            
            batch_images = []
            batch_texts = []
            batch_valid = []
            
            for p in batch:
                img_path = os.path.join(data_dir, p['image_path'])
                if os.path.exists(img_path):
                    batch_images.append(img_path)
                    batch_texts.append(p['full_text'])
                    batch_valid.append(p)
            
            if not batch_images:
                continue
            
            # 编码图像
            images = [self.preprocess(Image.open(img).convert("RGB")) for img in batch_images]
            image_tensor = torch.stack(images).to(self.device)
            
            with torch.no_grad():
                img_feats = self.model.encode_image(image_tensor)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            # 编码文本
            text_tokens = self.tokenizer(batch_texts).to(self.device)
            with torch.no_grad():
                txt_feats = self.model.encode_text(text_tokens)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            
            image_vectors.extend(img_feats.cpu().numpy())
            text_vectors.extend(txt_feats.cpu().numpy())
            valid_pairs.extend(batch_valid)
        
        return np.array(image_vectors), np.array(text_vectors), valid_pairs
