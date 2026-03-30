from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DataPaths:
    pairs: Path
    image_vectors: Path
    text_vectors: Path
    faiss_index: Path
    id_mapping: Path
    score_stats: Path


@dataclass
class AppConfig:
    root_dir: Path
    data_dir: Path
    model_name: str
    data_mode: str
    batch_size: int
    abs_threshold: float
    diff_threshold: float
    use_gpu_faiss: bool = True
    k_search: int = 12
    top_k: int = 3

    @property
    def paths(self) -> DataPaths:
        suffix = "_full" if self.data_mode == "full" else "_100"  # full:采纳全部数据集内容 _100：采纳前一百项（主要是最初构建项目时使用的）
        return DataPaths(
            pairs=self.data_dir / f"valid_pairs{suffix}.json",
            image_vectors=self.data_dir / f"image_vectors{suffix}.npy",
            text_vectors=self.data_dir / f"text_vectors{suffix}.npy",
            faiss_index=self.data_dir / f"faiss_index{suffix}.bin",
            id_mapping=self.data_dir / f"id_mapping{suffix}.json",
            score_stats=self.data_dir / f"score_stats{suffix}.json",
        )

    @property
    def pair_source(self) -> Path:
        return self.data_dir / "image_report_pairs.json"

    @property
    def image_root(self) -> Path:
        return self.data_dir / "images" / "images_normalized"


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(
    data_mode: str = "full",
    batch_size: int = 32,
    abs_threshold: float | None = None,
    diff_threshold: float | None = None,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
) -> AppConfig:
    root = resolve_project_root()
    data_dir = root / "data" / "iu_xray"

    cfg = AppConfig(
        root_dir=root,
        data_dir=data_dir,
        model_name=model_name,
        data_mode=data_mode,
        batch_size=batch_size,
        abs_threshold=abs_threshold if abs_threshold is not None else 0.07,
        diff_threshold=diff_threshold if diff_threshold is not None else 0.015,
    )

    stats_path = cfg.paths.score_stats
    if stats_path.exists() and (abs_threshold is None or diff_threshold is None):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if abs_threshold is None:
            cfg.abs_threshold = float(stats.get("recommended_abs_threshold", cfg.abs_threshold))
        if diff_threshold is None:
            cfg.diff_threshold = float(stats.get("recommended_diff_threshold", cfg.diff_threshold))

    return cfg
