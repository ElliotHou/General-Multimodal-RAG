import json
import os
import tempfile

import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.clip_encoder import CLIPEncoder
from src.config import AppConfig
from src.rag_pipeline import MedicalRAG


def load_faiss_index(index_path: str, use_gpu: bool = True):
    index_cpu = faiss.read_index(index_path)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index_cpu)
    return index_cpu


def load_rag_system(cfg: AppConfig) -> MedicalRAG:
    paths = cfg.paths
    for path in [paths.faiss_index, paths.id_mapping, paths.pairs]:
        if not path.exists():
            raise FileNotFoundError(f"缺少文件: {path}")

    print("加载CLIP...")
    encoder = CLIPEncoder()

    print("加载FAISS索引...")
    index = load_faiss_index(str(paths.faiss_index), use_gpu=cfg.use_gpu_faiss)

    with open(paths.id_mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(paths.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print("加载LLM...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    rag = MedicalRAG(
        encoder=encoder,
        faiss_index=index,
        id_mapping=mapping,
        pairs=pairs,
        tokenizer=tokenizer,
        llm_model=model,
        abs_threshold=cfg.abs_threshold,
        diff_threshold=cfg.diff_threshold,
        k_search=cfg.k_search,
        top_k=cfg.top_k,
    )
    print("RAG系统就绪")
    return rag


def infer(rag: MedicalRAG, image, question: str):
    if image is None:
        return "请先上传胸部X光图像。", "无"
    if not question or not question.strip():
        question = "这张胸部X光片显示什么异常？"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
    image.save(temp_path)

    try:
        answer, cases = rag.query(temp_path, question)
        retrieval_text = []
        for i, case in enumerate(cases, 1):
            retrieval_text.append(
                f"[{i}] {case.get('image_filename', 'unknown')} | "
                f"{case.get('projection', 'Unknown')} | "
                f"L2: {case.get('similarity_score', -1):.4f} | "
                f"印象: {case.get('impression', '')[:120]}"
            )
        return answer, "\n".join(retrieval_text)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
