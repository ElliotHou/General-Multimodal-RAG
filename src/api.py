from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile

from src.config import load_config
from src.service import format_retrieval, load_rag_system


app = FastAPI(
    title="Medical Multimodal RAG API",
    description="General-purpose multimodal RAG backend with a medical report demo corpus.",
    version="2.0",
)


@lru_cache(maxsize=1)
def get_rag():
    cfg = load_config(
        data_mode=os.getenv("RAG_DATA_MODE", "full"),
        llm_backend=os.getenv("RAG_LLM_BACKEND", "local"),
        model_name=os.getenv("RAG_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
    )
    return load_rag_system(cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(
    question: Annotated[str, Form()] = "",
    retrieval_mode: Annotated[str, Form()] = "auto",
    top_k: Annotated[int, Form()] = 3,
    image: Annotated[UploadFile | None, File()] = None,
):
    rag = get_rag()
    temp_path = None
    try:
        if image is not None:
            suffix = os.path.splitext(image.filename or "")[1] or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_path = tmp.name
                tmp.write(await image.read())

        answer, cases = rag.query(
            image_path=temp_path,
            question=question,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
        )
        return {
            "answer": answer,
            "cases": cases,
            "retrieval_text": format_retrieval(cases),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

