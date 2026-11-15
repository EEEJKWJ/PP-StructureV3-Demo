"""多模型实验对比和报告导出。"""

import csv
import html
import json
import logging
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path

from app.evaluation.metrics import grounded_cell_metrics
from app.evaluation.runner import (
    ExperimentStore,
    atomic_write_json,
    utc_now,
)


COMPARISON_FORMAT_VERSION = 1
COMPARISON_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$"
)


def create_comparison(
    output_root,
    dataset,
    model_ids,
    selection,
    retries,
    comparison_id=None,
):
    """创建对比元数据目录。"""
    model_ids = list(dict.fromkeys(model_ids))
    if len(model_ids) < 2:
        raise ValueError("模型对比至少需要两个不同模型")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_id = comparison_id or (
        time.strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    if not COMPARISON_ID_PATTERN.fullmatch(comparison_id):
        raise ValueError("对比 ID 无效")
    directory = output_root / comparison_id
    if directory.exists():
        raise FileExistsError(f"对比目录已存在: {directory}")
    directory.mkdir(parents=True)
    metadata = {
        "format_version": COMPARISON_FORMAT_VERSION,
        "comparison_id": comparison_id,
        "created_at": utc_now(),
        "dataset": dataset,
        "model_ids": model_ids,
        "selection": selection,
        "retries": retries,
    }
    atomic_write_json(directory / "comparison.json", metadata)
    return directory


def load_comparison(comparison_dir):
    comparison_dir = Path(comparison_dir).resolve()
    path = comparison_dir / "comparison.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 comparison.json: {comparison_dir}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def classify_result(result):
    """把单样本结果归到稳定错误类别。"""
    if result is None:
        return "missing"
    if result.get("status") == "success" and result.get("cell_count"):
        return "success"
    errors = " ".join(result.get("errors") or [])
    if "模型返回空结果" in errors or "返回空结果" in errors:
        return "empty_result"
    if "未检出表格单元格" in errors:
        return "no_cells"
    if result.get("status") == "failed":
        return "model_exception"
    if not result.get("cell_count"):
        return "no_cells"
    return "unknown"


def _sample_quality(result):
    if not result:
        return None
    ground_truth = result.get("ground_truth") or {}
    gt_cells = ground_truth.get("cells") or []
    if not gt_cells:
        return None
    return grounded_cell_metrics(
        result.get("cells") or [],
        gt_cells,
    )


def _quality_key(metrics):
    if not metrics:
        return None
    cer = metrics.get("cer")
    return (
        metrics.get("text_recall") or 0,
        metrics.get("match_rate_iou_0_5") or 0,
        -(cer if cer is not None else 999),
    )


def _quality_winner(model_ids, model_results):
    qualities = {
        model_id: _sample_quality(model_results[model_id])
        for model_id in model_ids
    }
    if any(
        model_results[model_id].get("status") != "success"
        or qualities[model_id] is None
        for model_id in model_ids
    ):
        return None, "not_comparable_no_ground_truth", qualities
    ranked = sorted(
        model_ids,
        key=lambda model_id: _quality_key(qualities[model_id]),
        reverse=True,
    )
    if _quality_key(qualities[ranked[0]]) == _quality_key(
        qualities[ranked[1]]
    ):
        return None, "tie", qualities
    return ranked[0], "winner", qualities


def generate_comparison_report(comparison_dir, logger=None):
    """读取各模型实验并生成全部对比文件。"""
    logger = logger or logging.getLogger("pp-struct-comparison")
    comparison_dir = Path(comparison_dir).resolve()
    metadata = load_comparison(comparison_dir)
    model_ids = metadata["model_ids"]

    model_summaries = []
    all_results = {}
    experiment_by_model = {}
    fingerprints = set()
    for model_id in model_ids:
        experiment_dir = comparison_dir / model_id
        store = ExperimentStore(experiment_dir, logger)
        if not store.metrics_path.exists():
            store.write_metrics_report()
        metrics_report = json.loads(
            store.metrics_path.read_text(encoding="utf-8")
        )
        experiment = store.experiment
        fingerprints.add(
            experiment.get("dataset", {}).get("fingerprint")
        )
        all_results[model_id] = store.load_latest_results()
        experiment_by_model[model_id] = experiment
        model_summaries.append({
            "model": experiment.get("model"),
            "metrics": metrics_report.get("metrics"),
            "metric_status": metrics_report.get("metric_status"),
            "summary": {
                "completed": len(all_results[model_id]),
                "successes": sum(
                    1
                    for result in all_results[model_id].values()
                    if classify_result(result) == "success"
                ),
            },
        })

    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("各模型实验的数据集指纹不一致")

    sample_ids = []
    seen = set()
    for model_id in model_ids:
        for result in sorted(
            all_results[model_id].values(),
            key=lambda item: item.get("global_index", 0),
        ):
            sample_id = result.get("sample_id")
            if sample_id and sample_id not in seen:
                seen.add(sample_id)
                sample_ids.append(sample_id)

    completion_wins = Counter()
    quality_wins = Counter()
    latency_wins = Counter()
    error_counts = {model_id: Counter() for model_id in model_ids}
    quality_comparable = 0
    samples = []

    for sample_id in sample_ids:
        model_results = {
            model_id: all_results[model_id].get(sample_id)
            for model_id in model_ids
        }
        success_ids = [
            model_id
            for model_id in model_ids
            if classify_result(model_results[model_id]) == "success"
        ]
        completion_winner = (
            success_ids[0] if len(success_ids) == 1 else None
        )
        if completion_winner:
            completion_wins[completion_winner] += 1

        quality_winner, quality_status, qualities = _quality_winner(
            model_ids,
            model_results,
        )
        if quality_status != "not_comparable_no_ground_truth":
            quality_comparable += 1
        if quality_winner:
            quality_wins[quality_winner] += 1

        latency_winner = None
        if len(success_ids) == len(model_ids):
            latency_winner = min(
                model_ids,
                key=lambda model_id: float(
                    model_results[model_id].get("elapsed_ms", 0)
                ),
            )
            latency_wins[latency_winner] += 1

        model_rows = {}
        for model_id in model_ids:
            result = model_results[model_id] or {}
            error_type = classify_result(model_results[model_id])
            error_counts[model_id][error_type] += 1
            model_rows[model_id] = {
                "status": result.get("status", "missing"),
                "error_type": error_type,
                "cell_count": result.get("cell_count", 0),
                "html_present": result.get("html_present", False),
                "elapsed_ms": result.get("elapsed_ms"),
                "row_count": result.get("row_count", 0),
                "col_count": result.get("col_count", 0),
                "quality": qualities.get(model_id),
            }

        first_result = next(
            (
                result
                for result in model_results.values()
                if result is not None
            ),
            None,
        ) or {}
        samples.append({
            "sample_id": sample_id,
            "file_name": first_result.get("file_name"),
            "global_index": first_result.get("global_index"),
            "models": model_rows,
            "completion_winner": completion_winner,
            "quality_winner": quality_winner,
            "quality_status": quality_status,
            "latency_winner": latency_winner,
        })

    summary = {
        "format_version": COMPARISON_FORMAT_VERSION,
        "comparison_id": metadata["comparison_id"],
        "generated_at": utc_now(),
        "dataset": metadata["dataset"],
        "selection": metadata["selection"],
        "model_ids": model_ids,
        "sample_count": len(samples),
        "models": model_summaries,
        "comparison": {
            "completion_wins": dict(completion_wins),
            "quality_comparable_samples": quality_comparable,
            "quality_wins": dict(quality_wins),
            "latency_wins": dict(latency_wins),
            "error_types": {
                model_id: dict(counts)
                for model_id, counts in error_counts.items()
            },
        },
    }

    _write_samples(comparison_dir, samples)
    atomic_write_json(
        comparison_dir / "comparison_summary.json",
        summary,
    )
    report = {**summary, "samples": samples}
    atomic_write_json(comparison_dir / "report.json", report)
    _write_csv(comparison_dir / "report.csv", model_ids, samples)
    _write_html(comparison_dir / "report.html", summary, samples)
    return summary


def _write_samples(comparison_dir, samples):
    path = comparison_dir / "comparison_samples.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_csv(path, model_ids, samples):
    base_columns = [
        "sample_id",
        "file_name",
        "global_index",
        "completion_winner",
        "quality_winner",
        "quality_status",
        "latency_winner",
    ]
    model_columns = []
    for model_id in model_ids:
        model_columns.extend([
            f"{model_id}_status",
            f"{model_id}_error_type",
            f"{model_id}_cell_count",
            f"{model_id}_elapsed_ms",
        ])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=base_columns + model_columns,
        )
        writer.writeheader()
        for sample in samples:
            row = {
                key: sample.get(key)
                for key in base_columns
            }
            for model_id in model_ids:
                result = sample["models"][model_id]
                for key in (
                    "status",
                    "error_type",
                    "cell_count",
                    "elapsed_ms",
                ):
                    row[f"{model_id}_{key}"] = result.get(key)
            writer.writerow(row)
    os.replace(temporary, path)


def _write_html(path, summary, samples):
    model_headers = "".join(
        f"<th>{html.escape(model_id)}</th>"
        for model_id in summary["model_ids"]
    )
    model_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['model'].get('name', '')))}</td>"
        f"<td>{item['summary']['completed']}</td>"
        f"<td>{item['summary']['successes']}</td>"
        f"<td>{item['metrics'].get('table_detection_rate')}</td>"
        f"<td>{item['metrics'].get('average_cells')}</td>"
        f"<td>{item['metrics'].get('latency_p50_ms')}</td>"
        f"<td>{item['metrics'].get('latency_p95_ms')}</td>"
        "</tr>"
        for item in summary["models"]
    )
    sample_rows = []
    for sample in samples:
        cells = "".join(
            f"<td>{html.escape(str(sample['models'][model_id]['error_type']))}<br>"
            f"{sample['models'][model_id]['cell_count']}</td>"
            for model_id in summary["model_ids"]
        )
        sample_rows.append(
            "<tr>"
            f"<td>{html.escape(str(sample.get('global_index', '')))}</td>"
            f"<td>{html.escape(str(sample.get('file_name', '')))}</td>"
            f"{cells}"
            f"<td>{html.escape(str(sample.get('completion_winner') or ''))}</td>"
            f"<td>{html.escape(str(sample.get('quality_winner') or ''))}</td>"
            f"<td>{html.escape(str(sample.get('latency_winner') or ''))}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="UTF-8">
<title>模型对比 {html.escape(summary['comparison_id'])}</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;color:#333}}
table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}
th,td{{border:1px solid #ddd;padding:6px 8px;font-size:12px;text-align:left}}
th{{background:#f5f5f5}}
code{{background:#f3f3f3;padding:2px 5px;border-radius:3px}}
</style></head><body>
<h1>模型对比报告</h1>
<p>数据集：{html.escape(summary['dataset'].get('name', ''))} ·
指纹：<code>{html.escape(summary['dataset'].get('fingerprint', ''))}</code></p>
<p>样本数：{summary['sample_count']} ·
结构质量可比较：{summary['comparison']['quality_comparable_samples']}</p>
<h2>模型指标</h2>
<table><thead><tr><th>模型</th><th>完成</th><th>成功</th>
<th>检出率</th><th>平均单元格</th><th>P50</th><th>P95</th></tr></thead>
<tbody>{model_rows}</tbody></table>
<h2>逐样本对比</h2>
<table><thead><tr><th>#</th><th>文件</th>{model_headers}
<th>完成胜者</th><th>质量胜者</th><th>延迟胜者</th></tr></thead>
<tbody>{''.join(sample_rows)}</tbody></table>
</body></html>"""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, path)
