"""在同一 FinTabNet 范围运行多个模型并导出对比报告。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.datasets.fintabnet import FinTabNetDataset
from app.datasets.registry import DatasetRegistry
from app.evaluation.comparison import (
    create_comparison,
    generate_comparison_report,
    load_comparison,
)
from app.evaluation.runner import BatchRunner, ExperimentStore


def build_parser():
    parser = argparse.ArgumentParser(
        description="运行多个表格模型的同数据集对比",
    )
    parser.add_argument(
        "--models",
        help="逗号分隔模型 ID；续跑时使用 comparison.json",
    )
    parser.add_argument("--dataset-id", default="fintabnet-test")
    parser.add_argument(
        "--parquet",
        help="直接指定 Parquet 文件或分片目录",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="experiments/comparisons",
    )
    parser.add_argument("--comparison-id")
    parser.add_argument(
        "--resume",
        help="已有对比目录，跳过各模型已完成样本",
    )
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def _dataset_from_metadata(metadata):
    path = metadata["dataset"]["input_path"]
    return FinTabNetDataset(path)


def main(argv=None):
    import logging

    import server

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offset < 0 or args.limit < 0:
        parser.error("offset/limit 不能小于 0")
    if args.batch_size <= 0:
        parser.error("batch-size 必须大于 0")
    if args.retries < 0:
        parser.error("retries 不能小于 0")

    logger = logging.getLogger("pp-struct-compare")
    if args.resume:
        comparison_dir = Path(args.resume).resolve()
        metadata = load_comparison(comparison_dir)
        dataset = _dataset_from_metadata(metadata)
        model_ids = metadata["model_ids"]
        selection = metadata["selection"]
        retries = metadata.get("retries", 0)
    else:
        model_ids = [
            model_id.strip()
            for model_id in (args.models or "").split(",")
            if model_id.strip()
        ]
        if len(set(model_ids)) < 2:
            parser.error("--models 至少包含两个不同模型")
        if args.parquet:
            dataset = FinTabNetDataset(args.parquet)
        else:
            dataset = DatasetRegistry().get_dataset(args.dataset_id)
        selection = {
            "offset": args.offset,
            "limit": args.limit,
            "batch_size": args.batch_size,
        }
        retries = args.retries
        comparison_dir = create_comparison(
            output_root=args.output_dir,
            dataset=dataset.describe(),
            model_ids=model_ids,
            selection=selection,
            retries=retries,
            comparison_id=args.comparison_id,
        )
        metadata = load_comparison(comparison_dir)

    print(
        f"comparison={metadata['comparison_id']} "
        f"samples={selection['offset']}:"
        f"{selection['offset'] + selection['limit']}",
        flush=True,
    )

    for model_id in model_ids:
        print(f"正在激活模型: {model_id}", flush=True)
        activation = server._activate_table_model(model_id)
        if not activation["ok"]:
            raise RuntimeError(
                f"模型 {model_id} 激活失败: "
                f"{activation.get('error')}"
            )
        active = server._table_service.registry.active_adapter
        model_info = dict(active.model_info)
        experiment_dir = comparison_dir / model_id
        selection_for_store = {
            "limit": selection["limit"],
            "start": selection["offset"],
            "batch_size": selection["batch_size"],
        }
        if experiment_dir.exists():
            store = ExperimentStore(experiment_dir, logger)
        else:
            store = ExperimentStore.create(
                output_root=comparison_dir,
                logger=logger,
                dataset=dataset.describe(),
                model=model_info,
                selection=selection_for_store,
                retries=retries,
                experiment_id=model_id,
            )

        runner = BatchRunner(
            dataset=dataset,
            recognizer=server.recognize_table_structure,
            store=store,
            model_info=model_info,
            logger=logger,
            retries=retries,
        )
        summary = runner.run(
            limit=selection["limit"],
            start=selection["offset"],
            batch_size=selection["batch_size"],
            retry_failed=args.retry_failed,
            progress=lambda item, skipped, prefix=model_id: print(
                (
                    f"[{prefix}] SKIP {item.sample_id}"
                    if skipped
                    else f"[{prefix}] #{item['global_index']} "
                    f"{item['status']} cells={item['cell_count']} "
                    f"{item['elapsed_ms'] / 1000:.2f}s"
                ),
                flush=True,
            ),
        )
        print(
            f"[{model_id}] completed={summary['completed']} "
            f"success={summary['successes']}",
            flush=True,
        )

    report = generate_comparison_report(comparison_dir, logger)
    print(json.dumps(
        {
            "comparison": report["comparison"],
            "sample_count": report["sample_count"],
            "models": [
                {
                    "id": item["model"]["id"],
                    "metrics": item["metrics"],
                }
                for item in report["models"]
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"comparison_dir={comparison_dir}")
    print(f"report_json={comparison_dir / 'report.json'}")
    print(f"report_csv={comparison_dir / 'report.csv'}")
    print(f"report_html={comparison_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
