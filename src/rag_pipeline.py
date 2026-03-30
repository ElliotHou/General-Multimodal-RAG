import torch


class MedicalRAG:
    def __init__(
        self,
        encoder,
        faiss_index,
        id_mapping: dict,
        pairs: list[dict],
        tokenizer,
        llm_model,
        abs_threshold: float = 0.07,
        diff_threshold: float = 0.015,
        k_search: int = 12,
        top_k: int = 3,
    ):
        self.encoder = encoder
        self.index = faiss_index
        self.id_to_type = id_mapping["id_to_type"]
        self.id_to_pair_idx = id_mapping["id_to_pair_idx"]
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.model = llm_model
        self.abs_threshold = abs_threshold
        self.diff_threshold = diff_threshold
        self.k_search = k_search
        self.top_k = top_k

    def retrieve(self, query_image_path: str) -> list[dict]:
        query_vec = self.encoder.encode_image(query_image_path).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vec, self.k_search)

        seen_pair = set()
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            idx = int(idx)
            idx_str = str(idx)
            if idx_str not in self.id_to_pair_idx:
                continue

            pair_idx = self.id_to_pair_idx[idx_str]
            if pair_idx in seen_pair:
                continue

            case = self.pairs[pair_idx].copy()
            case["retrieval_type"] = self.id_to_type.get(idx_str, "unknown")
            case["similarity_score"] = float(dist)
            results.append(case)
            seen_pair.add(pair_idx)

            if len(results) >= self.top_k:
                break

        return results

    def _analyze_scores(self, retrieved_cases: list[dict]) -> tuple[str, list[float]]:
        if not retrieved_cases:
            return "low_confidence", []

        scores = [c["similarity_score"] for c in retrieved_cases]
        best = scores[0]

        if best > self.abs_threshold:
            return "low_confidence", scores

        if len(scores) >= 2:
            diff = scores[1] - scores[0]
            if diff > self.diff_threshold:
                return "single_strong", scores

        return "multi_similar", scores

    def generate(self, retrieved_cases: list[dict], user_question: str, max_new_tokens: int = 320) -> str:
        if not retrieved_cases:
            return "未检索到相似病例。"

        mode, scores = self._analyze_scores(retrieved_cases)
        if mode == "low_confidence":
            return f"未找到足够相似的病例（最小距离 {scores[0]:.4f}），建议结合更多临床信息判断。"

        if mode == "single_strong":
            strategy = (
                f"第1个病例（距离{scores[0]:.4f}）显著优于其他病例。"
                "请主要参考第1个病例，避免把其他病例异常误加到结论。"
            )
        else:
            shown = ", ".join([f"{s:.4f}" for s in scores[:3]])
            strategy = (
                f"多个病例相似度接近（距离分别为 {shown}）。"
                "请综合参考，避免只依赖单一病例。"
            )

        context_parts = []
        for i, case in enumerate(retrieved_cases[:3], 1):
            context_parts.append(
                f"[相似病例{i}] "
                f"影像: {case.get('image_filename', 'Unknown')} | "
                f"视角: {case.get('projection', 'Unknown')} | "
                f"L2距离: {case.get('similarity_score', -1):.4f}\n"
                f"Findings: {case.get('findings', '')[:220]}\n"
                f"Impression: {case.get('impression', '')[:220]}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            "你是资深放射科医师，请根据相似病例与用户问题给出专业中文分析。\n"
            f"策略: {strategy}\n\n"
            f"相似病例上下文:\n{context}\n\n"
            f"用户问题: {user_question}\n\n"
            "输出要求：\n"
            "1) 先给简要影像结论\n"
            "2) 再给支持该结论的影像依据\n"
            "3) 最后给建议（如随访或进一步检查）\n"
            "4) 不要逐条重复病例原文\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        return response

    def query(self, image_path: str, question: str) -> tuple[str, list[dict]]:
        cases = self.retrieve(image_path)
        answer = self.generate(cases, question)
        return answer, cases
