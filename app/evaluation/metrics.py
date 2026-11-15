"""从逐样本表格识别结果计算可复现指标。"""

import math
import re
from datetime import datetime, timezone

from app.evaluation.teds import calculate_html_metrics


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def nearest_rank_percentile(values, percentile):
    """使用 nearest-rank 计算百分位；空输入返回 None。"""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def bbox_iou(first, second):
    """计算两个 bbox 的 IoU。"""
    if not first or not second or len(first) < 4 or len(second) < 4:
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0,
        float(first[3]) - float(first[1]),
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0,
        float(second[3]) - float(second[1]),
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def normalize_cell_text(text):
    """单元格精确文本匹配使用的规范化。"""
    return re.sub(r"\s+", "", str(text or "")).casefold()


def ordered_cell_text(cells):
    ordered = sorted(
        cells or [],
        key=lambda cell: (
            int(cell.get("row", 0) or 0),
            int(cell.get("col", 0) or 0),
        ),
    )
    return " ".join(str(cell.get("text", "") or "") for cell in ordered)


def levenshtein_distance(first, second):
    """计算两个序列的 Levenshtein 编辑距离。"""
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_item in enumerate(first, 1):
        current = [first_index]
        for second_index, second_item in enumerate(second, 1):
            insertion = current[second_index - 1] + 1
            deletion = previous[second_index] + 1
            substitution = previous[second_index - 1] + (
                0 if first_item == second_item else 1
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def word_tokens(text):
    """中文按字符、其他语言按连续字母数字切分。"""
    return re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+",
        str(text or "").casefold(),
    )


def structure_edges(cells):
    """从单元格占用网格提取右邻和下邻结构边。"""
    cells = list(cells or [])
    coverage = {}
    positions = {}
    for index, cell in enumerate(cells):
        row = int(cell.get("row", 0) or 0)
        col = int(cell.get("col", 0) or 0)
        rowspan = max(1, int(cell.get("rowspan", 1) or 1))
        colspan = max(1, int(cell.get("colspan", 1) or 1))
        for row_index in range(row, row + rowspan):
            for col_index in range(col, col + colspan):
                coverage[(row_index, col_index)] = index
                positions.setdefault(index, []).append(
                    (row_index, col_index)
                )

    edges = set()
    canonical = {
        index: min(cell_positions)
        for index, cell_positions in positions.items()
    }
    for (row, col), cell_index in coverage.items():
        right = coverage.get((row, col + 1))
        below = coverage.get((row + 1, col))
        if right is not None and right != cell_index:
            edges.add((canonical[cell_index], canonical[right]))
        if below is not None and below != cell_index:
            edges.add((canonical[cell_index], canonical[below]))
    return coverage, edges


def grounded_cell_metrics(predictions, ground_truth):
    """计算一组预测与真值单元格的空间和文本指标。"""
    predictions = list(predictions or [])
    ground_truth = list(ground_truth or [])
    if not ground_truth:
        return None

    candidates = []
    best_iou_sum = 0.0
    for gt_index, gt_cell in enumerate(ground_truth):
        best_iou = 0.0
        for pred_index, prediction in enumerate(predictions):
            iou = bbox_iou(
                prediction.get("bbox"),
                gt_cell.get("bbox"),
            )
            best_iou = max(best_iou, iou)
            candidates.append((
                iou,
                gt_index,
                pred_index,
            ))
        best_iou_sum += best_iou

    used_gt = set()
    used_predictions = set()
    spatial_matches = []
    for iou, gt_index, pred_index in sorted(
        candidates,
        reverse=True,
    ):
        if gt_index in used_gt or pred_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(pred_index)
        if iou >= 0.5:
            spatial_matches.append((gt_index, pred_index, iou))

    exact_text_matches = sum(
        1
        for gt_index, pred_index, _ in spatial_matches
        if normalize_cell_text(
            ground_truth[gt_index].get("text")
        )
        == normalize_cell_text(
            predictions[pred_index].get("text")
        )
    )

    predicted_text = re.sub(
        r"\s+",
        "",
        ordered_cell_text(predictions),
    )
    truth_text = re.sub(
        r"\s+",
        "",
        ordered_cell_text(ground_truth),
    )
    cer = (
        levenshtein_distance(predicted_text, truth_text)
        / len(truth_text)
        if truth_text
        else None
    )
    predicted_words = word_tokens(ordered_cell_text(predictions))
    truth_words = word_tokens(ordered_cell_text(ground_truth))
    wer = (
        levenshtein_distance(predicted_words, truth_words)
        / len(truth_words)
        if truth_words
        else None
    )
    _, prediction_edges = structure_edges(predictions)
    _, truth_edges = structure_edges(ground_truth)
    edge_true_positive = len(prediction_edges & truth_edges)
    if not prediction_edges and not truth_edges:
        edge_precision = 1.0
        edge_recall = 1.0
    else:
        edge_precision = (
            edge_true_positive / len(prediction_edges)
            if prediction_edges
            else 0.0
        )
        edge_recall = (
            edge_true_positive / len(truth_edges)
            if truth_edges
            else 0.0
        )
    edge_f1 = (
        2 * edge_precision * edge_recall
        / (edge_precision + edge_recall)
        if edge_precision + edge_recall > 0
        else 0.0
    )
    return {
        "mean_best_iou": best_iou_sum / len(ground_truth),
        "match_rate_iou_0_5": len(spatial_matches) / len(ground_truth),
        "text_recall": exact_text_matches / len(ground_truth),
        "cer": cer,
        "wer": wer,
        "structure_edge_precision": edge_precision,
        "structure_edge_recall": edge_recall,
        "structure_edge_f1": edge_f1,
    }


def _mean_defined(values):
    available = [value for value in values if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def _round_optional(value, digits=4):
    return round(value, digits) if value is not None else None


def compute_metrics(results, experiment=None, include_teds=True):
    """从结果字典或列表生成完整指标报告。"""
    if isinstance(results, dict):
        results = list(results.values())
    else:
        results = list(results or [])

    total = len(results)
    successes = [
        result
        for result in results
        if result.get("status") == "success"
        and int(result.get("cell_count", 0) or 0) > 0
    ]
    failures = total - len(successes)
    annotated = [
        result
        for result in results
        if int(result.get("table_box_count", 0) or 0) > 0
    ]
    detected_annotated = [
        result
        for result in annotated
        if result.get("status") == "success"
        and int(result.get("cell_count", 0) or 0) > 0
    ]

    latencies = [
        float(result["elapsed_ms"])
        for result in results
        if result.get("elapsed_ms") is not None
    ]
    p50 = nearest_rank_percentile(latencies, 50)
    p95 = nearest_rank_percentile(latencies, 95)
    latency_sum = sum(latencies)
    throughput = total / (latency_sum / 1000) if latency_sum > 0 else None
    average_cells = _mean_defined(
        [
            float(result.get("cell_count", 0) or 0)
            for result in successes
        ]
    )
    average_attempts = _mean_defined(
        [
            float(result.get("attempts", 0) or 0)
            for result in results
        ]
    )

    grounded_samples = []
    html_truth_count = 0
    html_metrics = []
    html_parse_errors = 0
    html_missing_predictions = 0
    teds_skipped = 0
    for result in results:
        truth = result.get("ground_truth") or {}
        gt_cells = truth.get("cells") or []
        if gt_cells:
            metric = grounded_cell_metrics(
                result.get("cells") or [],
                gt_cells,
            )
            if metric is not None:
                grounded_samples.append(metric)
        if truth.get("html"):
            html_truth_count += 1
            if not include_teds:
                teds_skipped += 1
                continue
            metric = result.get("html_metrics")
            if not isinstance(metric, dict):
                metric = calculate_html_metrics(
                    result.get("html"),
                    truth.get("html"),
                )
            if metric.get("status") == "available":
                html_metrics.append(metric)
            elif metric.get("status") == "parse_error":
                html_parse_errors += 1
            elif metric.get("status") == "missing_prediction_html":
                html_missing_predictions += 1

    cell_mean_iou = _mean_defined(
        [item["mean_best_iou"] for item in grounded_samples]
    )
    cell_match_rate = _mean_defined(
        [item["match_rate_iou_0_5"] for item in grounded_samples]
    )
    text_recall = _mean_defined(
        [item["text_recall"] for item in grounded_samples]
    )
    cer = _mean_defined([item["cer"] for item in grounded_samples])
    wer = _mean_defined([item["wer"] for item in grounded_samples])
    edge_precision = _mean_defined(
        [item["structure_edge_precision"] for item in grounded_samples]
    )
    edge_recall = _mean_defined(
        [item["structure_edge_recall"] for item in grounded_samples]
    )
    edge_f1 = _mean_defined(
        [item["structure_edge_f1"] for item in grounded_samples]
    )
    teds_values = _mean_defined(
        [item["teds"] for item in html_metrics]
    )
    teds_s_values = _mean_defined(
        [item["teds_s"] for item in html_metrics]
    )

    detection_rate = (
        len(detected_annotated) / len(annotated)
        if annotated
        else None
    )
    success_rate = len(successes) / total if total else None
    html_successes = sum(
        bool(result.get("html_present")) for result in successes
    )
    html_rate = html_successes / total if total else None

    metrics = {
        "result_count": total,
        "success_count": len(successes),
        "failure_count": failures,
        "annotated_sample_count": len(annotated),
        "detected_sample_count": len(detected_annotated),
        "table_detection_rate": _round_optional(detection_rate),
        "average_cells": _round_optional(average_cells, 2),
        "latency_p50_ms": _round_optional(p50, 2),
        "latency_p95_ms": _round_optional(p95, 2),
        "throughput_samples_per_second": _round_optional(
            throughput,
            3,
        ),
        "success_rate": _round_optional(success_rate),
        "html_rate": _round_optional(html_rate),
        "average_attempts": _round_optional(average_attempts, 2),
        "cell_mean_best_iou": _round_optional(cell_mean_iou),
        "cell_match_rate_iou_0_5": _round_optional(cell_match_rate),
        "cell_text_recall": _round_optional(text_recall),
        "cer": _round_optional(cer),
        "wer": _round_optional(wer),
        "structure_edge_precision": _round_optional(edge_precision),
        "structure_edge_recall": _round_optional(edge_recall),
        "structure_edge_f1": _round_optional(edge_f1),
        "teds": _round_optional(teds_values),
        "teds_s": _round_optional(teds_s_values),
    }

    ground_truth_status = (
        "available"
        if grounded_samples
        else "not_available_no_ground_truth"
    )
    no_detection_status = (
        "available" if successes else "not_applicable_no_detections"
    )
    if not include_teds and html_truth_count:
        structure_status = "skipped"
    elif html_truth_count == 0:
        structure_status = "not_available_no_ground_truth"
    elif html_metrics:
        structure_status = "available"
    elif html_parse_errors:
        structure_status = "parse_error"
    elif html_missing_predictions:
        structure_status = "missing_prediction_html"
    else:
        structure_status = "not_available"

    metric_status = {
        "table_detection_rate": (
            "available" if annotated else "not_applicable_no_table_annotations"
        ),
        "average_cells": no_detection_status,
        "latency_p50_ms": "available" if latencies else "not_available",
        "latency_p95_ms": "available" if latencies else "not_available",
        "throughput_samples_per_second": (
            "available" if throughput is not None else "not_available"
        ),
        "success_rate": "available" if total else "not_available",
        "html_rate": "available" if total else "not_available",
        "cell_mean_best_iou": ground_truth_status,
        "cell_match_rate_iou_0_5": ground_truth_status,
        "cell_text_recall": ground_truth_status,
        "cer": ground_truth_status,
        "wer": ground_truth_status,
        "structure_edge_precision": ground_truth_status,
        "structure_edge_recall": ground_truth_status,
        "structure_edge_f1": ground_truth_status,
        "teds": structure_status,
        "teds_s": structure_status,
    }

    source = {
        "generated_at": utc_now(),
        "result_count": total,
    }
    if experiment:
        source.update({
            "experiment_id": experiment.get("experiment_id"),
            "dataset": experiment.get("dataset"),
            "model": experiment.get("model"),
        })

    return {
        "format_version": 1,
        "source": source,
        "metrics": metrics,
        "metric_status": metric_status,
        "ground_truth_coverage": {
            "cell_samples": len(grounded_samples),
            "html_samples": html_truth_count,
            "sample_count": total,
        },
        "teds_coverage": {
            "sample_count": html_truth_count,
            "available": len(html_metrics),
            "parse_errors": html_parse_errors,
            "missing_prediction_html": html_missing_predictions,
            "skipped": teds_skipped,
        },
        "definitions": {
            "table_detection_rate": (
                "successful cell detections / completed samples "
                "with table annotations"
            ),
            "average_cells": (
                "mean cell_count over successfully detected samples"
            ),
            "latency_percentiles": (
                "nearest-rank over elapsed_ms of all completed results"
            ),
            "throughput_samples_per_second": (
                "results / sum(elapsed_ms); excludes model startup "
                "and dataset loading"
            ),
            "grounded_cell_metrics": (
                "spatial matching at IoU >= 0.5; exact normalized "
                "cell text recall; reading-order CER/WER"
            ),
            "structure_edges": (
                "precision/recall/F1 over right and lower adjacency "
                "edges in the occupied cell grid"
            ),
            "teds": (
                "ordered tree edit similarity including canonical cell text"
            ),
            "teds_s": (
                "ordered tree edit similarity with structure-only labels"
            ),
        },
    }
