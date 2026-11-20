"""在同一官方整页上下文上对比官方表格和自定义表格。"""

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.datasets.registry import DatasetRegistry
from app.evaluation.metrics import (
    bbox_iou,
    grounded_cell_metrics,
    nearest_rank_percentile,
)
from app.evaluation.runner import atomic_write_json
from app.evaluation.table_canonical import canonical_metrics
from app.evaluation.teds import calculate_html_metrics
from app.services.official_parser import apply_custom_table_results


_GROUNDED_TRUTH_UNSET = object()


def select_table_block(page, ground_truth_bbox):
    """选择与真值表格区域最接近的预测 table block。"""
    candidates = [
        block
        for block in page.get("blocks") or []
        if block.get("label") == "table"
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda block: bbox_iou(
            block.get("bbox"),
            ground_truth_bbox,
        ),
    )


def block_metrics(
    block,
    truth,
    grounded_truth=_GROUNDED_TRUTH_UNSET,
):
    if block is None:
        return {
            "status": "no_table_block",
            "teds": None,
            "teds_s": None,
            "text_recall": None,
            "mean_best_iou": None,
            "match_rate_iou_0_5": None,
            "cer": None,
            "wer": None,
        }
    html_metrics = calculate_html_metrics(
        block.get("table_html"),
        truth.get("html"),
    )
    if grounded_truth is _GROUNDED_TRUTH_UNSET:
        grounded_truth = truth
    cell_metrics = (
        grounded_cell_metrics(
            block.get("cells") or [],
            grounded_truth.get("cells") or [],
        )
        if grounded_truth is not None
        else None
    ) or {}
    has_result = bool(
        block.get("table_html") or block.get("cells")
    )
    metrics = {
        "status": "success" if has_result else "empty_table_result",
        "teds": html_metrics.get("teds"),
        "teds_s": html_metrics.get("teds_s"),
        "text_recall": cell_metrics.get("text_recall"),
        "mean_best_iou": cell_metrics.get("mean_best_iou"),
        "match_rate_iou_0_5": cell_metrics.get(
            "match_rate_iou_0_5"
        ),
        "cer": cell_metrics.get("cer"),
        "wer": cell_metrics.get("wer"),
        "grounding_status": (
            "available"
            if grounded_truth is not None
            else "unavailable_page_alignment"
        ),
    }
    if grounded_truth is not None:
        metrics.update(
            canonical_metrics(
                block.get("table_html"),
                block.get("cells") or [],
                grounded_truth,
            )
        )
    else:
        metrics.update({
            "teds_canonical": None,
            "teds_s_canonical": None,
            "text_recall_canonical": None,
            "cer_canonical": None,
            "wer_canonical": None,
        })
    return metrics


def mean_metric(rows, key):
    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]
    return round(sum(values) / len(values), 4) if values else None


def aggregate_engine(results, engine):
    records = [item[engine] for item in results]
    successes = [
        item for item in records if item["status"] == "success"
    ]
    latency_key = (
        "elapsed_ms"
        if engine == "official"
        else "injection_ms"
    )
    latencies = [
        item.get(latency_key)
        for item in records
        if item.get(latency_key) is not None
    ]
    return {
        "result_count": len(records),
        "success_count": len(successes),
        "success_rate": (
            round(len(successes) / len(records), 4)
            if records
            else None
        ),
        "fallback_count": sum(
            bool(item.get("fallback_detection"))
            for item in records
        ),
        "teds": mean_metric(successes, "teds"),
        "teds_s": mean_metric(successes, "teds_s"),
        "text_recall": mean_metric(successes, "text_recall"),
        "mean_best_iou": mean_metric(
            successes,
            "mean_best_iou",
        ),
        "match_rate_iou_0_5": mean_metric(
            successes,
            "match_rate_iou_0_5",
        ),
        "cer": mean_metric(successes, "cer"),
        "wer": mean_metric(successes, "wer"),
        f"{'parse' if engine == 'official' else 'injection'}"
        "_p50_ms": (
            round(nearest_rank_percentile(latencies, 50), 2)
            if latencies
            else None
        ),
        f"{'parse' if engine == 'official' else 'injection'}"
        "_p95_ms": (
            round(nearest_rank_percentile(latencies, 95), 2)
            if latencies
            else None
        ),
    }


def paired_summary(results):
    """只比较双方都成功的样本。"""
    both = [
        item
        for item in results
        if item["official"]["status"] == "success"
        and item["custom"]["status"] == "success"
    ]
    official_rows = [item["official"] for item in both]
    custom_rows = [item["custom"] for item in both]
    return {
        "both_success": len(both),
        "official_only_success": sum(
            item["official"]["status"] == "success"
            and item["custom"]["status"] != "success"
            for item in results
        ),
        "custom_only_success": sum(
            item["custom"]["status"] == "success"
            and item["official"]["status"] != "success"
            for item in results
        ),
        "both_failed": sum(
            item["official"]["status"] != "success"
            and item["custom"]["status"] != "success"
            for item in results
        ),
        "official": {
            key: mean_metric(official_rows, key)
            for key in (
                "teds",
                "teds_s",
                "text_recall",
                "mean_best_iou",
                "match_rate_iou_0_5",
                "cer",
                "wer",
            )
        },
        "custom": {
            key: mean_metric(custom_rows, key)
            for key in (
                "teds",
                "teds_s",
                "text_recall",
                "mean_best_iou",
                "match_rate_iou_0_5",
                "cer",
                "wer",
            )
        },
    }


def write_report(comparison_dir, metadata, results):
    report = {
        "format_version": 1,
        "comparison": metadata,
        "sample_count": len(results),
        "official": aggregate_engine(results, "official"),
        "custom": aggregate_engine(results, "custom"),
        "paired": paired_summary(results),
        "latency_scope": {
            "official": "full page parse including official table",
            "custom": "custom table injection and fallback detection only",
        },
    }
    atomic_write_json(comparison_dir / "report.json", report)
    rows = []
    for item in results:
        rows.append(
            "<tr>"
            f"<td>{item['file_name']}</td>"
            f"<td>{item['official']['status']}</td>"
            f"<td>{item['official']['teds']}</td>"
            f"<td>{item['custom']['status']}</td>"
            f"<td>{item['custom']['teds']}</td>"
            f"<td>{item['custom'].get('fallback_detection')}</td>"
            "</tr>"
        )
    official = report["official"]
    custom = report["custom"]
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>官方表格与自定义表格对比</title>
<style>
body{{font:14px/1.5 system-ui;margin:24px;background:#f5f7f9}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #ccd5dd;padding:6px;text-align:left}}
.summary{{display:flex;gap:16px;margin:16px 0}}
.card{{background:white;border:1px solid #ccd5dd;padding:12px}}
</style></head><body>
<h1>官方表格与自定义表格对比</h1>
<div class="summary">
<div class="card">官方成功 {official['success_count']} /
{official['result_count']}，TEDS {official['teds']}，
S-TEDS {official['teds_s']}，IoU {official['mean_best_iou']}</div>
<div class="card">自定义成功 {custom['success_count']} /
{custom['result_count']}，TEDS {custom['teds']}，
S-TEDS {custom['teds_s']}，IoU {custom['mean_best_iou']}，
补框 {custom['fallback_count']}</div>
</div>
<p>双方都成功 {report['paired']['both_success']} 条；
仅官方成功 {report['paired']['official_only_success']} 条；
仅自定义成功 {report['paired']['custom_only_success']} 条；
双方失败 {report['paired']['both_failed']} 条。</p>
<table><thead><tr><th>文件</th><th>官方状态</th><th>官方 TEDS</th>
<th>自定义状态</th><th>自定义 TEDS</th><th>本地补框</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (comparison_dir / "report.html").write_text(
        document,
        encoding="utf-8",
    )
    return report


def load_results(path):
    latest = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            custom = item.setdefault("custom", {})
            if (
                "injection_ms" not in custom
                and "elapsed_ms" in custom
            ):
                custom["injection_ms"] = custom.pop("elapsed_ms")
            latest[item["sample_id"]] = item
    return latest


def main(argv=None):
    import server

    parser = argparse.ArgumentParser(
        description="比较官方表格和当前自定义表格",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument(
        "--output-dir",
        default="experiments/official-table-comparisons",
    )
    parser.add_argument("--comparison-id")
    parser.add_argument("--resume")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.start < 0:
        parser.error("limit/start 参数无效")

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("official-table-compare")
    output_root = Path(args.output_dir).resolve()
    if args.resume:
        comparison_dir = Path(args.resume).resolve()
        metadata = json.loads(
            (comparison_dir / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        args.start = metadata["selection"]["start"]
        args.limit = metadata["selection"]["limit"]
        args.model = metadata["model"].get("id")
    else:
        comparison_id = args.comparison_id or time.strftime(
            "official-tables-%Y%m%dT%H%M%S"
        )
        comparison_dir = output_root / comparison_id
        comparison_dir.mkdir(parents=True, exist_ok=False)
        metadata = None

    if args.model:
        activation = server._activate_table_model(args.model)
        if not activation["ok"]:
            raise RuntimeError(activation.get("error"))
    else:
        server._get_table_engine()
    active = server._table_service.registry.active_adapter
    model_info = dict(active.model_info)

    dataset = DatasetRegistry().get_dataset(
        "fintabnet-c-structure-test"
    )
    if metadata is None:
        metadata = {
            "comparison_id": comparison_dir.name,
            "dataset": dataset.describe(),
            "model": model_info,
            "selection": {
                "start": args.start,
                "limit": args.limit,
            },
            "official_engine": "PP-StructureV3",
            "custom_table_source": "active model registry",
        }
        atomic_write_json(
            comparison_dir / "comparison.json",
            metadata,
        )

    results_path = comparison_dir / "results.jsonl"
    latest = load_results(results_path)
    samples = dataset.iter_samples(
        start=args.start,
        limit=args.limit,
    )
    with tempfile.TemporaryDirectory(
        dir=PROJECT_DIR,
        prefix=".official_table_compare_",
    ) as temp_dir:
        for index, sample in enumerate(
            samples,
            start=args.start,
        ):
            if sample.sample_id in latest and not (
                args.retry_failed
                and latest[sample.sample_id]["custom"]["status"]
                != "success"
            ):
                continue
            image_path = Path(temp_dir) / f"{index}.jpg"
            image_path.write_bytes(sample.image_bytes)
            started = time.perf_counter()
            page = server._official_parser.parse_page(
                str(image_path),
                page_num=index + 1,
                is_pdf=False,
                predict_options={
                    "use_table_recognition": True,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                },
            )
            parse_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            official_block = select_table_block(
                page,
                sample.table_bbox,
            )
            injection_started = time.perf_counter()
            custom_page = apply_custom_table_results(
                copy.deepcopy(page),
                str(image_path),
                server._table_service.recognize,
                server._detect_table_blocks_with_fallback,
                True,
            )
            injection_ms = round(
                (time.perf_counter() - injection_started) * 1000,
                2,
            )
            custom_block = select_table_block(
                custom_page,
                sample.table_bbox,
            )
            custom_is_override = bool(
                custom_block
                and custom_block.get("table_source") == "custom"
            )
            official = block_metrics(
                official_block,
                sample.ground_truth,
            )
            official["elapsed_ms"] = parse_ms
            official["fallback_detection"] = False
            custom = block_metrics(
                custom_block,
                sample.ground_truth,
            )
            if not custom_is_override:
                custom["status"] = "override_failed"
            custom["injection_ms"] = injection_ms
            custom["fallback_detection"] = bool(
                custom_block
                and custom_block.get("table_fallback_detection")
            )
            custom["full_image_fallback"] = bool(
                custom_block
                and custom_block.get("full_image_fallback")
            )
            custom["model"] = (
                custom_block.get("table_model")
                if custom_block
                else None
            )
            item = {
                "sample_id": sample.sample_id,
                "file_name": sample.file_name,
                "parse_ms": parse_ms,
                "official": official,
                "custom": custom,
            }
            latest[sample.sample_id] = item
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"{sample.sample_id}: official={official['status']} "
                f"custom={custom['status']} "
                f"fallback={custom['fallback_detection']}",
                flush=True,
            )

    results = sorted(
        latest.values(),
        key=lambda item: item["sample_id"],
    )
    report = write_report(comparison_dir, metadata, results)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"comparison_dir={comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
