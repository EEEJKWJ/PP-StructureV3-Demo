"""运行可恢复的 FinTabNet 批量表格识别实验。"""

import argparse
import json
from pathlib import Path

from app.datasets.fintabnet import FinTabNetDataset
from app.evaluation.runner import BatchRunner, ExperimentStore


DEFAULT_PARQUET = (
    r"D:\dataset\fintabnet-pubtables-full\test"
    r"\test-00000-of-00006.parquet"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="运行 FinTabNet 批量表格识别实验",
    )
    parser.add_argument(
        "--parquet",
        help="Parquet 文件或分片目录；续跑时默认使用实验记录路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="本次运行最多检查的样本数",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--model",
        help="模型注册清单中的模型 ID",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="单样本首次失败后的重试次数",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="新实验的输出根目录",
    )
    parser.add_argument(
        "--experiment-id",
        help="新实验 ID；省略时自动生成",
    )
    parser.add_argument(
        "--resume",
        help="已有实验目录，恢复未完成样本",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="续跑时重新执行失败的样本",
    )
    return parser


def _load_store(args, logger):
    if not args.resume:
        return None
    store = ExperimentStore(args.resume, logger)
    dataset = store.experiment["dataset"]
    model = store.experiment["model"]
    selection = store.experiment["selection"]
    args.parquet = dataset["input_path"]
    args.limit = selection["limit"]
    args.start = selection["start"]
    args.batch_size = selection["batch_size"]
    args.retries = store.experiment.get("retries", 0)
    args.model = model.get("id")
    return store


def _activate_model(server, model_id):
    if model_id:
        activation = server._activate_table_model(model_id)
        if not activation["ok"]:
            raise RuntimeError(
                activation.get("error")
                or f"无法激活模型: {model_id}"
            )
    else:
        server._get_table_engine()

    active = server._table_service.registry.active_adapter
    if active is None:
        raise RuntimeError("没有可用的表格模型")
    return dict(active.model_info)


def _print_progress(item, skipped):
    if skipped:
        print(f"SKIP {item.sample_id}", flush=True)
        return
    status = item["status"].upper()
    print(
        f"{item['global_index']:05d} {item['file_name']}: "
        f"{status} cells={item['cell_count']} "
        f"html={int(item['html_present'])} "
        f"attempts={item['attempts']} "
        f"time={item['elapsed_ms'] / 1000:.2f}s",
        flush=True,
    )


def main(argv=None):
    import logging

    import server

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error("--retries 不能小于 0")
    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    if args.start < 0:
        parser.error("--start 不能小于 0")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")

    logger = logging.getLogger("pp-struct-batch")
    store = _load_store(args, logger)
    dataset_path = args.parquet or DEFAULT_PARQUET
    dataset = FinTabNetDataset(dataset_path)
    if store is not None:
        recorded_fingerprint = store.experiment["dataset"]["fingerprint"]
        if dataset.describe()["fingerprint"] != recorded_fingerprint:
            raise RuntimeError(
                "续跑数据集指纹与实验记录不一致"
            )

    model_info = _activate_model(server, args.model)
    selection = {
        "limit": args.limit,
        "start": args.start,
        "batch_size": args.batch_size,
    }
    if store is None:
        store = ExperimentStore.create(
            output_root=args.output_dir,
            logger=logger,
            dataset=dataset.describe(),
            model=model_info,
            selection=selection,
            retries=args.retries,
            experiment_id=args.experiment_id,
        )

    runner = BatchRunner(
        dataset=dataset,
        recognizer=server.recognize_table_structure,
        store=store,
        model_info=model_info,
        logger=logger,
        retries=args.retries,
    )
    summary = runner.run(
        limit=args.limit,
        start=args.start,
        batch_size=args.batch_size,
        retry_failed=args.retry_failed,
        progress=_print_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"experiment_dir={Path(store.directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
