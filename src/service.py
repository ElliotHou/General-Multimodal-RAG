import json
import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.clip_encoder import CLIPEncoder
from src.config import AppConfig
from src.llm_backend import LocalTransformersLLM, build_api_llm_from_env
from src.rag_pipeline import MedicalRAG


def load_faiss_index(index_path: str, use_gpu: bool = True):
    index_cpu = faiss.read_index(index_path)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index_cpu)
    return index_cpu


def build_llm(cfg: AppConfig):
    backend = cfg.llm_backend.lower()
    if backend == "none":
        return None
    if backend == "api":
        return build_api_llm_from_env()
    if backend != "local":
        raise ValueError(f"Unsupported llm backend: {cfg.llm_backend}")

    print("加载本地LLM...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return LocalTransformersLLM(tokenizer=tokenizer, model=model)


def load_rag_system(cfg: AppConfig) -> MedicalRAG:
    paths = cfg.paths
    required = [paths.faiss_index, paths.id_mapping, paths.pairs]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing asset: {path}")

    print("加载CLIP编码器...")
    encoder = CLIPEncoder()

    print("加载FAISS索引...")
    index = load_faiss_index(str(paths.faiss_index), use_gpu=cfg.use_gpu_faiss)

    with open(paths.id_mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(paths.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    text_vectors = None
    if paths.text_vectors.exists():
        text_vectors = np.load(paths.text_vectors)

    llm = build_llm(cfg)

    rag = MedicalRAG(
        encoder=encoder,
        faiss_index=index,
        id_mapping=mapping,
        pairs=pairs,
        llm=llm,
        text_vectors=text_vectors,
        abs_threshold=cfg.abs_threshold,
        diff_threshold=cfg.diff_threshold,
        k_search=cfg.k_search,
        top_k=cfg.top_k,
        image_weight=cfg.image_weight,
        text_weight=cfg.text_weight,
        keyword_weight=cfg.keyword_weight,
    )
    print("RAG系统就绪")
    return rag


def validate_question(question: str) -> tuple[bool, str]:
    if question is None:
        return True, ""
    if len(question) > 800:
        return False, "问题过长，请控制在800字以内。"
    return True, ""


def validate_image(image) -> tuple[bool, str]:
    if image is None:
        return True, ""
    width, height = image.size
    if width * height > 10000 * 10000:
        return False, "图像过大，请上传100MP以内的图片。"
    return True, ""


def save_temp_image(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
    image.save(temp_path)
    return temp_path


def format_retrieval(cases: list[dict]) -> str:
    lines = []
    for i, case in enumerate(cases, 1):
        sources = ", ".join(case.get("retrieval_sources", []))
        lines.append(
            f"[{i}] uid={case.get('uid', 'unknown')} | "
            f"{case.get('image_filename', 'unknown')} | "
            f"{case.get('projection', 'Unknown')} | "
            f"sources={sources} | "
            f"fusion={case.get('fusion_score', 0):.4f} | "
            f"rerank={case.get('rerank_score') if case.get('rerank_score') is not None else 'N/A'}\n"
            f"Evidence: {case.get('evidence', '')[:260]}"
        )
    return "\n\n".join(lines)


def infer(
    rag: MedicalRAG,
    image,
    question: str,
    retrieval_mode: str = "auto",
    top_k: int | None = None,
):
    question = (question or "").strip()
    if image is None and not question:
        return "请上传图像或输入问题。", "无"

    valid, message = validate_image(image)
    if not valid:
        return message, "无"
    valid, message = validate_question(question)
    if not valid:
        return message, "无"

    temp_path = save_temp_image(image)
    try:
        answer, cases = rag.query(
            image_path=temp_path,
            question=question,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
        )
        return answer, format_retrieval(cases)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def infer_from_path(
    rag: MedicalRAG,
    image_path: str | Path | None,
    question: str,
    retrieval_mode: str = "auto",
    top_k: int | None = None,
) -> tuple[str, list[dict]]:
    if image_path is not None and not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return rag.query(
        image_path=None if image_path is None else str(image_path),
        question=question,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
    )

