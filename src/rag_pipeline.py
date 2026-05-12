from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize_report_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class SimpleBM25:
    """Small dependency-free BM25 index for offline demos and interviews."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.num_docs = len(documents)

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        if not query_terms or self.num_docs == 0:
            return [0.0] * self.num_docs

        scores = [0.0] * self.num_docs
        for term in query_terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for i, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-8))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0]


@dataclass
class Candidate:
    pair_idx: int
    score: float = 0.0
    sources: set[str] = field(default_factory=set)
    distances: dict[str, float] = field(default_factory=dict)
    bm25_score: float = 0.0
    rerank_score: float | None = None


class MedicalRAG:
    def __init__(
        self,
        encoder,
        faiss_index,
        id_mapping: dict,
        pairs: list[dict],
        llm: Any | None = None,
        text_vectors: np.ndarray | None = None,
        abs_threshold: float = 0.07,
        diff_threshold: float = 0.015,
        k_search: int = 20,
        top_k: int = 3,
        image_weight: float = 0.55,
        text_weight: float = 0.35,
        keyword_weight: float = 0.10,
    ):
        self.encoder = encoder
        self.index = faiss_index
        self.id_to_type = id_mapping["id_to_type"]
        self.id_to_pair_idx = id_mapping["id_to_pair_idx"]
        self.pairs = [self._normalize_case(pair) for pair in pairs]
        self.llm = llm
        self.text_vectors = self._normalize_vectors(text_vectors)
        self.abs_threshold = abs_threshold
        self.diff_threshold = diff_threshold
        self.k_search = k_search
        self.top_k = top_k
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.keyword_weight = keyword_weight
        self.bm25 = SimpleBM25([case["evidence_text"] for case in self.pairs])

    def _normalize_vectors(self, vectors: np.ndarray | None) -> np.ndarray | None:
        if vectors is None:
            return None
        vectors = vectors.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-8)

    def _normalize_case(self, pair: dict) -> dict:
        findings = normalize_report_text(pair.get("findings", ""))
        impression = normalize_report_text(pair.get("impression", ""))
        full_text = normalize_report_text(pair.get("full_text", ""))
        if not full_text:
            full_text = f"Findings: {findings}\nImpression: {impression}".strip()

        evidence = (
            f"Projection: {pair.get('projection', 'Unknown')}\n"
            f"Findings: {findings or 'Not provided.'}\n"
            f"Impression: {impression or 'Not provided.'}"
        )

        normalized = pair.copy()
        normalized.update(
            {
                "findings": findings,
                "impression": impression,
                "full_text": full_text,
                "evidence_text": evidence,
                "metadata": {
                    "uid": pair.get("uid"),
                    "projection": pair.get("projection", "Unknown"),
                    "image_filename": pair.get("image_filename", "Unknown"),
                    "image_path": pair.get("image_path", ""),
                },
            }
        )
        return normalized

    def _vector_search(self, query_vector: np.ndarray, source: str, weight: float, limit: int) -> dict[int, Candidate]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        distances, indices = self.index.search(query_vector.astype("float32"), limit)

        candidates: dict[int, Candidate] = {}
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            idx_str = str(int(idx))
            if idx_str not in self.id_to_pair_idx:
                continue
            pair_idx = int(self.id_to_pair_idx[idx_str])
            item_type = self.id_to_type.get(idx_str, "unknown")
            similarity = 1.0 / (1.0 + float(dist))
            rrf = 1.0 / (60.0 + rank)
            candidate = candidates.setdefault(pair_idx, Candidate(pair_idx=pair_idx))
            candidate.score += weight * (similarity + rrf)
            candidate.sources.add(f"{source}:{item_type}")
            best_key = f"{source}_{item_type}_distance"
            candidate.distances[best_key] = min(float(dist), candidate.distances.get(best_key, float("inf")))
        return candidates

    def _merge_candidates(self, candidate_maps: list[dict[int, Candidate]]) -> dict[int, Candidate]:
        merged: dict[int, Candidate] = {}
        for candidate_map in candidate_maps:
            for pair_idx, candidate in candidate_map.items():
                target = merged.setdefault(pair_idx, Candidate(pair_idx=pair_idx))
                target.score += candidate.score
                target.sources.update(candidate.sources)
                target.bm25_score = max(target.bm25_score, candidate.bm25_score)
                for key, value in candidate.distances.items():
                    target.distances[key] = min(value, target.distances.get(key, float("inf")))
        return merged

    def _keyword_search(self, question: str, weight: float, limit: int) -> dict[int, Candidate]:
        candidates: dict[int, Candidate] = {}
        for rank, (pair_idx, bm25_score) in enumerate(self.bm25.search(question, limit), start=1):
            candidate = candidates.setdefault(pair_idx, Candidate(pair_idx=pair_idx))
            candidate.score += weight * (bm25_score + 1.0 / (60.0 + rank))
            candidate.sources.add("keyword:bm25")
            candidate.bm25_score = bm25_score
        return candidates

    def _rerank(self, candidates: list[Candidate], question: str) -> list[Candidate]:
        if not question.strip() or self.text_vectors is None or not candidates:
            return candidates

        query_vec = self.encoder.encode_text(question).astype("float32")
        query_vec = query_vec / max(float(np.linalg.norm(query_vec)), 1e-8)
        for candidate in candidates:
            if candidate.pair_idx < len(self.text_vectors):
                semantic_score = float(np.dot(query_vec, self.text_vectors[candidate.pair_idx]))
            else:
                semantic_score = 0.0
            lexical_boost = math.log1p(candidate.bm25_score)
            candidate.rerank_score = candidate.score + 0.35 * semantic_score + 0.05 * lexical_boost

        return sorted(candidates, key=lambda c: c.rerank_score if c.rerank_score is not None else c.score, reverse=True)

    def _candidate_to_case(self, candidate: Candidate) -> dict:
        case = self.pairs[candidate.pair_idx].copy()
        distances = {key: value for key, value in candidate.distances.items() if value != float("inf")}
        best_distance = min(distances.values()) if distances else None
        case.update(
            {
                "pair_idx": candidate.pair_idx,
                "retrieval_sources": sorted(candidate.sources),
                "retrieval_type": "+".join(sorted(candidate.sources)),
                "similarity_score": best_distance if best_distance is not None else float(1.0 / (1.0 + max(candidate.score, 0.0))),
                "fusion_score": float(candidate.score),
                "rerank_score": None if candidate.rerank_score is None else float(candidate.rerank_score),
                "bm25_score": float(candidate.bm25_score),
                "distance_breakdown": distances,
                "evidence": case["evidence_text"],
            }
        )
        return case

    def retrieve(
        self,
        query_image_path: str | None = None,
        question: str = "",
        retrieval_mode: str = "auto",
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> list[dict]:
        mode = retrieval_mode.lower()
        if mode == "auto":
            if query_image_path and question.strip():
                mode = "hybrid"
            elif query_image_path:
                mode = "image"
            elif question.strip():
                mode = "text"
            else:
                return []

        limit = max(self.k_search, (top_k or self.top_k) * 6)
        candidate_maps: list[dict[int, Candidate]] = []

        if query_image_path and mode in {"image", "hybrid", "fusion"}:
            image_vec = self.encoder.encode_image(query_image_path).reshape(1, -1).astype("float32")
            weight = self.image_weight if mode in {"hybrid", "fusion"} else 1.0
            candidate_maps.append(self._vector_search(image_vec, "image_query", weight, limit))

        if question.strip() and mode in {"text", "hybrid", "fusion"}:
            text_vec = self.encoder.encode_text(question).reshape(1, -1).astype("float32")
            vector_weight = self.text_weight if mode in {"hybrid", "fusion"} else 0.75
            keyword_weight = self.keyword_weight if mode in {"hybrid", "fusion"} else 0.25
            candidate_maps.append(self._vector_search(text_vec, "text_query", vector_weight, limit))
            candidate_maps.append(self._keyword_search(question, keyword_weight, limit))

        merged = self._merge_candidates(candidate_maps)
        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        if use_rerank:
            ranked = self._rerank(ranked[:limit], question)
        else:
            ranked = ranked[:limit]
        return [self._candidate_to_case(candidate) for candidate in ranked[: (top_k or self.top_k)]]

    def _analyze_scores(self, retrieved_cases: list[dict]) -> tuple[str, list[float]]:
        if not retrieved_cases:
            return "low_confidence", []

        distances = [
            case["similarity_score"]
            for case in retrieved_cases
            if isinstance(case.get("similarity_score"), (int, float))
        ]
        if not distances:
            return "multi_similar", []

        best = min(distances)
        if best > self.abs_threshold:
            return "low_confidence", distances

        if len(distances) >= 2:
            sorted_distances = sorted(distances)
            diff = sorted_distances[1] - sorted_distances[0]
            if diff > self.diff_threshold:
                return "single_strong", distances

        return "multi_similar", distances

    def build_prompt(self, retrieved_cases: list[dict], user_question: str) -> str:
        mode, scores = self._analyze_scores(retrieved_cases)
        if mode == "low_confidence":
            strategy = "检索置信度较低。只能基于证据说明不确定性，禁止给出确定诊断。"
        elif mode == "single_strong":
            strategy = "第1条证据明显更接近。以第1条为主，其他证据只作辅助对照。"
        else:
            shown = ", ".join([f"{s:.4f}" for s in scores[:3]]) if scores else "N/A"
            strategy = f"多条证据接近，距离为 {shown}。综合多条证据，避免依赖单一病例。"

        context_parts = []
        for i, case in enumerate(retrieved_cases[:3], 1):
            context_parts.append(
                f"[证据{i}]\n"
                f"uid: {case.get('uid', 'Unknown')}\n"
                f"image: {case.get('image_filename', 'Unknown')}\n"
                f"sources: {', '.join(case.get('retrieval_sources', []))}\n"
                f"fusion_score: {case.get('fusion_score', 0):.4f}\n"
                f"{case.get('evidence', '')[:700]}"
            )
        context = "\n\n".join(context_parts)

        return (
            "你是一个专业知识库 RAG 助手，当前知识库场景是胸部 X 光报告与相似病例。\n"
            "你不是临床诊断系统。必须只基于给定证据回答，不允许编造证据外的检查结果。\n\n"
            f"检索策略: {strategy}\n\n"
            f"检索证据:\n{context}\n\n"
            f"用户问题: {user_question or '请总结图像/报告中可支持的关键信息。'}\n\n"
            "输出要求:\n"
            "1) 先给简要结论，并说明置信度。\n"
            "2) 用“证据1/证据2/证据3”引用支撑依据。\n"
            "3) 如果证据不足，明确说证据不足，不要强行诊断。\n"
            "4) 最后给出下一步建议或需要补充的信息。\n"
        )

    def generate(self, retrieved_cases: list[dict], user_question: str, max_new_tokens: int = 360) -> str:
        if not retrieved_cases:
            return "未检索到可用证据。请补充问题、上传图像，或调整检索模式。"

        mode, scores = self._analyze_scores(retrieved_cases)
        if mode == "low_confidence":
            best = scores[0] if scores else -1
            return (
                f"检索证据不足，当前最小距离为 {best:.4f}。"
                "系统不会基于低置信结果给出确定结论；建议补充更明确的问题、更多临床信息或重新上传图像。"
            )

        prompt = self.build_prompt(retrieved_cases, user_question)
        if self.llm is None:
            return prompt
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)

    def query(
        self,
        image_path: str | None = None,
        question: str = "",
        retrieval_mode: str = "auto",
        top_k: int | None = None,
    ) -> tuple[str, list[dict]]:
        cases = self.retrieve(
            query_image_path=image_path,
            question=question,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
        )
        answer = self.generate(cases, question)
        return answer, cases
