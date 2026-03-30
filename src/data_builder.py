import json

import faiss
import numpy as np
import pandas as pd

from src.clip_encoder import CLIPEncoder
from src.config import AppConfig
from src.vector_store import VectorStore


def _safe_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def build_image_report_pairs(cfg: AppConfig, save_to_source: bool = True) -> list[dict]:
    reports_csv = cfg.data_dir / "indiana_reports.csv"
    projections_csv = cfg.data_dir / "indiana_projections.csv"

    reports_df = pd.read_csv(reports_csv)
    projections_df = pd.read_csv(projections_csv)

    report_map = {}
    for _, r in reports_df.iterrows():
        uid = int(r["uid"])
        findings = _safe_text(r.get("findings", ""))
        impression = _safe_text(r.get("impression", ""))
        full_text = f"Findings: {findings}\nImpression: {impression}".strip()
        report_map[uid] = {
            "findings": findings,
            "impression": impression,
            "full_text": full_text,
        }

    pairs = []

    for _, p in projections_df.iterrows():
        uid = int(p["uid"])
        if uid not in report_map:
            continue

        image_filename = str(p["filename"])
        image_path_rel = f"images/images_normalized/{image_filename}"
        image_path_abs = cfg.data_dir / image_path_rel
        if not image_path_abs.exists():
            continue

        pairs.append(
            {
                "uid": uid,
                "image_filename": image_filename,
                "projection": str(p.get("projection", "Unknown")),
                "findings": report_map[uid]["findings"],
                "impression": report_map[uid]["impression"],
                "full_text": report_map[uid]["full_text"],
                "image_path": image_path_rel,
            }
        )

    if save_to_source:
        with open(cfg.pair_source, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"pairs built: {len(pairs)}")
    print(f"saved source: {cfg.pair_source}")
    return pairs


def encode_and_save_full(cfg: AppConfig, pairs: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if pairs is None:
        with open(cfg.pair_source, "r", encoding="utf-8") as f:
            pairs = json.load(f)

    encoder = CLIPEncoder()
    image_vectors, text_vectors, valid_pairs = encoder.encode_batch(
        pairs=pairs,
        data_dir=str(cfg.data_dir),
        batch_size=cfg.batch_size,
    )

    np.save(cfg.paths.image_vectors, image_vectors)
    np.save(cfg.paths.text_vectors, text_vectors)

    with open(cfg.paths.pairs, "w", encoding="utf-8") as f:
        json.dump(valid_pairs, f, ensure_ascii=False, indent=2)

    print(f"saved image vectors: {cfg.paths.image_vectors} {image_vectors.shape}")
    print(f"saved text vectors : {cfg.paths.text_vectors} {text_vectors.shape}")
    print(f"saved valid pairs  : {cfg.paths.pairs} {len(valid_pairs)}")

    return image_vectors, text_vectors, valid_pairs


def build_faiss_and_save(cfg: AppConfig, image_vectors: np.ndarray, text_vectors: np.ndarray) -> dict:
    use_gpu = cfg.use_gpu_faiss and faiss.get_num_gpus() > 0
    store = VectorStore(dimension=image_vectors.shape[1], use_gpu=use_gpu)
    mapping = store.build_index(image_vectors, text_vectors)

    if use_gpu:
        index_cpu = faiss.index_gpu_to_cpu(store.index)
    else:
        index_cpu = store.index

    faiss.write_index(index_cpu, str(cfg.paths.faiss_index))
    with open(cfg.paths.id_mapping, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"saved faiss index: {cfg.paths.faiss_index}")
    print(f"saved id mapping : {cfg.paths.id_mapping}")
    return mapping


def estimate_thresholds(
    cfg: AppConfig,
    sample_size: int | None = None,
    search_k: int = 4,
) -> dict:
    with open(cfg.paths.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    index_cpu = faiss.read_index(str(cfg.paths.faiss_index))
    use_gpu = cfg.use_gpu_faiss and faiss.get_num_gpus() > 0
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu

    encoder = CLIPEncoder()
    n_cases = len(pairs) if sample_size is None else min(sample_size, len(pairs))

    all_scores = []
    for case in pairs[:n_cases]:
        img_path = cfg.image_root / case["image_filename"]
        if not img_path.exists():
            continue

        query = encoder.encode_image(str(img_path)).reshape(1, -1).astype("float32")
        distances, _ = index.search(query, search_k)
        dists = [float(x) for x in distances[0] if x > 1e-8]
        all_scores.extend(dists)

    scores = np.array(all_scores, dtype=np.float32)
    if len(scores) == 0:
        raise RuntimeError("No valid distance scores generated.")

    p50 = float(np.percentile(scores, 50))
    p80 = float(np.percentile(scores, 80))
    p90 = float(np.percentile(scores, 90))
    mean = float(np.mean(scores))
    max_value = float(np.max(scores))

    recommended_abs = p90
    recommended_diff = max(0.005, p80 - p50) # 避免阈值过小

    stats = {
        "num_queries": n_cases,
        "num_scores": int(len(scores)),
        "mean": mean,
        "p50": p50,
        "p80": p80,
        "p90": p90,
        "max": max_value,
        "recommended_abs_threshold": recommended_abs,
        "recommended_diff_threshold": recommended_diff,
    }

    with open(cfg.paths.score_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("distance stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved stats: {cfg.paths.score_stats}")
    return stats


def build_all_assets(
    cfg: AppConfig,
    rebuild_pairs: bool = False,
    threshold_sample_size: int | None = None,
) -> dict:
    if rebuild_pairs or (not cfg.pair_source.exists()):
        pairs = build_image_report_pairs(cfg, save_to_source=True)
    else:
        with open(cfg.pair_source, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"use existing pair source: {cfg.pair_source} {len(pairs)}")

    image_vectors, text_vectors, valid_pairs = encode_and_save_full(cfg, pairs)
    mapping = build_faiss_and_save(cfg, image_vectors, text_vectors)
    stats = estimate_thresholds(cfg, sample_size=threshold_sample_size)

    return {
        "valid_pairs": len(valid_pairs),
        "index_total": int(image_vectors.shape[0] + text_vectors.shape[0]),
        "mapping_images": mapping.get("num_images", 0),
        "mapping_texts": mapping.get("num_texts", 0),
        "abs_threshold": stats["recommended_abs_threshold"],
        "diff_threshold": stats["recommended_diff_threshold"],
    }
