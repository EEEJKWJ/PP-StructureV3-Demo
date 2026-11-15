"""从已有实验逐样本结果重新计算指标。"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.evaluation.runner import ExperimentStore


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="计算已有表格识别实验的指标",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="包含 experiment.json 和 results.jsonl 的实验目录",
    )
    parser.add_argument(
        "--skip-teds",
        action="store_true",
        help="跳过 TEDS/S-TEDS，仅计算其他指标",
    )
    args = parser.parse_args(argv)

    logger = logging.getLogger("pp-struct-metrics")
    store = ExperimentStore(args.experiment_dir, logger)
    report = store.write_metrics_report(
        include_teds=not args.skip_teds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"metrics_file={store.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
