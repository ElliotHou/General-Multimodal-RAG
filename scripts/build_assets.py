import argparse

from src.config import load_config
from src.data_builder import build_all_assets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-mode", default="full", choices=["sample", "full"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--threshold-sample-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = "full" if args.data_mode == "full" else "sample"

    cfg = load_config(
        data_mode=mode,
        batch_size=args.batch_size,
    )

    result = build_all_assets(
        cfg=cfg,
        rebuild_pairs=args.rebuild_pairs,
        threshold_sample_size=args.threshold_sample_size,
    )

    print("build summary:")
    for key, value in result.items():
        print(f"{key}: {value}")
