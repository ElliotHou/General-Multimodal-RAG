from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ABNORMAL_TERMS = {
    "opacity",
    "opacities",
    "effusion",
    "pneumothorax",
    "cardiomegaly",
    "consolidation",
    "atelectasis",
    "edema",
    "fracture",
    "mass",
    "nodule",
    "emphysema",
    "fibrosis",
    "pneumonia",
}


def is_abnormal(case: dict) -> bool:
    text = f"{case.get('findings', '')} {case.get('impression', '')}".lower()
    return any(term in text for term in ABNORMAL_TERMS)


def dcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def evaluate_mode(rag, pairs: list[dict], mode: str, sample_size: int, top_k: int) -> dict:
    recall_1 = 0
    recall_k = 0
    reciprocal_ranks = []
    ndcg_scores = []
    label_hits = 0
    latencies = []

    for case in pairs[:sample_size]:
        full_image_path = rag.pairs[case["pair_idx"]].get("_abs_image_path")
        question = case["question"]

        query_image = None
        query_text = ""
        retrieval_mode = mode
        use_rerank = False
        if mode == "image":
            query_image = full_image_path
        elif mode == "text":
            query_text = question
        else:
            query_image = full_image_path
            query_text = question
            retrieval_mode = "hybrid"
            use_rerank = mode in {"hybrid+rerank", "fusion+rerank"}

        start = time.perf_counter()
        results = rag.retrieve(
            query_image_path=query_image,
            question=query_text,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        latencies.append(time.perf_counter() - start)

        target_uid = case["uid"]
        target_label = case["is_abnormal"]
        hit_rank = None
        for rank, result in enumerate(results, start=1):
            if result.get("uid") == target_uid:
                hit_rank = rank
                break

        if hit_rank == 1:
            recall_1 += 1
        if hit_rank is not None:
            recall_k += 1
            reciprocal_ranks.append(1.0 / hit_rank)
            ndcg_scores.append(dcg(hit_rank))
        else:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)

        if results and is_abnormal(results[0]) == target_label:
            label_hits += 1

    denom = max(sample_size, 1)
    return {
        "mode": mode,
        "sample_size": sample_size,
        "recall@1": recall_1 / denom,
        f"recall@{top_k}": recall_k / denom,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        f"ndcg@{top_k}": mean(ndcg_scores) if ndcg_scores else 0.0,
        "normal_abnormal_top1_acc": label_hits / denom,
        "avg_latency_ms": 1000 * mean(latencies) if latencies else 0.0,
    }


def build_eval_samples(rag, sample_size: int) -> list[dict]:
    samples = []
    for pair_idx, case in enumerate(rag.pairs[:sample_size]):
        question = case.get("impression") or case.get("findings") or case.get("full_text")
        samples.append(
            {
                "pair_idx": pair_idx,
                "uid": case.get("uid"),
                "question": question,
                "is_abnormal": is_abnormal(case),
            }
        )
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes for the multimodal RAG system.")
    parser.add_argument("--data-mode", default="full", choices=["sample", "full"])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["image", "text", "hybrid", "hybrid+rerank"])
    return parser.parse_args()


def main():
    args = parse_args()

    from src.config import load_config
    from src.service import load_rag_system

    cfg = load_config(data_mode=args.data_mode, llm_backend="none")
    rag = load_rag_system(cfg)

    for case in rag.pairs:
        case["_abs_image_path"] = str(cfg.image_root / case["image_filename"])

    sample_size = min(args.sample_size, len(rag.pairs))
    samples = build_eval_samples(rag, sample_size)
    results = [evaluate_mode(rag, samples, mode, sample_size, args.top_k) for mode in args.modes]

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nMarkdown table:")
    print("| mode | recall@1 | recall@{} | MRR | nDCG@{} | label acc | latency(ms) |".format(args.top_k, args.top_k))
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            "| {mode} | {r1:.4f} | {rk:.4f} | {mrr:.4f} | {ndcg:.4f} | {acc:.4f} | {lat:.1f} |".format(
                mode=row["mode"],
                r1=row["recall@1"],
                rk=row[f"recall@{args.top_k}"],
                mrr=row["mrr"],
                ndcg=row[f"ndcg@{args.top_k}"],
                acc=row["normal_abnormal_top1_acc"],
                lat=row["avg_latency_ms"],
            )
        )


if __name__ == "__main__":
    main()
