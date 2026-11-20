"""运行官方、自定义和表格局部预处理五组实验。"""

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.datasets.registry import DatasetRegistry
from app.evaluation.metrics import bbox_iou, nearest_rank_percentile
from app.evaluation.runner import atomic_write_json
from app.services.page_alignment import (
    align_ground_truth,
    estimate_page_alignment,
)
from app.services.official_parser import apply_custom_table_results
from app.services.table_preprocess import (
    crop_table_from_image,
    enhance_table_crop,
    processed_bbox_to_page,
)
from app.services.table_text_fusion import build_fused_table
from scripts.compare_official_table_engines import (
    block_metrics,
    select_table_block,
)


VARIANTS = {"A", "B", "C", "D", "E"}


def _map_cells_to_page(cells, crop_info, transform):
    mapped = []
    for cell in cells or []:
        new_cell = dict(cell)
        bbox = cell.get("bbox")
        if bbox and list(bbox) != [0, 0, 0, 0]:
            new_cell["bbox"] = processed_bbox_to_page(
                bbox,
                crop_info,
                transform,
            )
        mapped.append(new_cell)
    return mapped


def _identity_transform(width, height):
    return {
        "matrix": [[1, 0, 0], [0, 1, 0]],
        "inverse_matrix": [[1, 0, 0], [0, 1, 0]],
        "source_width": width,
        "source_height": height,
        "output_width": width,
        "output_height": height,
        "scale": 1.0,
        "skew_angle": 0.0,
    }


def _resolve_table_bbox(
    page,
    preferred_bbox,
    image_path,
    fallback_detector,
    allow_full_image,
):
    """选择与当前评估坐标系中的真值表格最接近的 bbox。"""
    block = select_table_block(page, preferred_bbox or [0, 0, 0, 0])
    if block is not None:
        return block.get("bbox") or []
    fallback_blocks = fallback_detector(image_path) or []
    tables = [
        item
        for item in fallback_blocks
        if item.get("label") == "table"
    ]
    if tables:
        return max(
            tables,
            key=lambda item: bbox_iou(
                item.get("bbox"),
                preferred_bbox,
            ),
        )["bbox"]
    if allow_full_image:
        image = cv2.imread(image_path)
        if image is not None:
            return [0, 0, image.shape[1], image.shape[0]]
    return None


def _cell_structure_signature(cells):
    return [
        (
            int(cell.get("row", 0) or 0),
            int(cell.get("col", 0) or 0),
            int(cell.get("rowspan", 1) or 1),
            int(cell.get("colspan", 1) or 1),
        )
        for cell in cells or []
    ]


def _attach_table_statistics(result, truth, official_cells=None):
    """补充过分割和合并单元格保持统计。"""
    cells = result.get("cells") or []
    truth_cells = truth.get("cells") or []
    predicted_count = len(cells)
    truth_count = len(truth_cells)
    result["predicted_cell_count"] = predicted_count
    result["truth_cell_count"] = truth_count
    result["cell_count_ratio"] = (
        round(predicted_count / truth_count, 4)
        if truth_count
        else None
    )
    result["over_segmentation_rate"] = (
        round(max(0, predicted_count - truth_count) / truth_count, 4)
        if truth_count
        else None
    )
    result["empty_cell_count"] = sum(
        not str(cell.get("text") or "").strip()
        for cell in cells
    )
    result["merged_cell_count"] = sum(
        int(cell.get("rowspan", 1) or 1) > 1
        or int(cell.get("colspan", 1) or 1) > 1
        for cell in cells
    )
    result["truth_merged_cell_count"] = sum(
        int(cell.get("rowspan", 1) or 1) > 1
        or int(cell.get("colspan", 1) or 1) > 1
        for cell in truth_cells
    )
    for key in (
        "teds",
        "teds_s",
        "text_recall",
        "cer",
        "wer",
    ):
        result[f"{key}_raw"] = result.get(key)
    if official_cells is not None:
        result["official_cell_count"] = len(official_cells)
        result["structure_preserved"] = (
            _cell_structure_signature(cells)
            == _cell_structure_signature(official_cells)
        )
    return result


def _run_current_table(server, image, crop_info, transform, temp_path):
    cv2.imwrite(str(temp_path), image)
    started = time.perf_counter()
    result = server._table_service.recognize(
        str(temp_path),
        [0, 0, image.shape[1], image.shape[0]],
    )
    elapsed_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )
    if result is None:
        return None
    cells = _map_cells_to_page(
        result.get("cells") or [],
        crop_info,
        transform,
    )
    return {
        "cells": cells,
        "table_html": result.get("html") or "",
        "table_model": result.get("model"),
        "elapsed_ms": elapsed_ms,
        "status": "success" if cells else "empty_table_result",
    }


def _run_fused_table(
    official_block,
    enhanced,
    crop_info,
    transform,
    ocr_service,
):
    started = time.perf_counter()
    ocr_lines = ocr_service.recognize_crop(enhanced)
    result = build_fused_table(
        official_block,
        ocr_lines,
        crop_info,
        transform,
    )
    elapsed_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )
    if result is None:
        return None
    return {
        **result,
        "elapsed_ms": elapsed_ms,
        "status": (
            "success"
            if result.get("cells") or result.get("html")
            else "empty_table_result"
        ),
        "ocr_line_count": len(ocr_lines),
        "assignment": result.get("assignment"),
    }


def _run_custom_baseline(server, image_path, sample):
    """在未做官方页面矫正的原图上复现当前自定义基线。"""
    page = server._official_parser.parse_page(
        str(image_path),
        page_num=sample.global_index + 1,
        is_pdf=False,
        predict_options={
            "use_table_recognition": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        },
    )
    started = time.perf_counter()
    page = apply_custom_table_results(
        copy.deepcopy(page),
        str(image_path),
        server._table_service.recognize,
        server._detect_table_blocks_with_fallback,
        True,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    block = select_table_block(page, sample.table_bbox)
    result = block_metrics(block, sample.ground_truth)
    if (
        block is None
        or block.get("table_source") != "custom"
    ):
        result["status"] = "override_failed"
    result["elapsed_ms"] = elapsed_ms
    result["fallback_detection"] = bool(
        block and block.get("table_fallback_detection")
    )
    result["full_image_fallback"] = bool(
        block and block.get("full_image_fallback")
    )
    result["source"] = "live current custom baseline"
    statistics = _attach_table_statistics(
        {"cells": (block.get("cells") if block else None) or []},
        sample.ground_truth,
    )
    result.update(statistics)
    result.pop("cells", None)
    return result


def _aggregate_variant(records, variant):
    rows = [record["variants"][variant] for record in records]
    successes = [
        row for row in rows if row["status"] == "success"
    ]
    latencies = [
        row["elapsed_ms"]
        for row in rows
        if row.get("elapsed_ms") is not None
    ]

    def mean(key):
        values = [
            row[key]
            for row in successes
            if row.get(key) is not None
        ]
        return round(sum(values) / len(values), 4) if values else None

    grounded = [
        row
        for row in successes
        if row.get("grounding_status") == "available"
    ]
    return {
        "result_count": len(rows),
        "success_count": len(successes),
        "success_rate": (
            round(len(successes) / len(rows), 4)
            if rows
            else None
        ),
        "grounding_available_count": len(grounded),
        "alignment_failure_count": sum(
            row.get("grounding_status")
            == "unavailable_page_alignment"
            for row in rows
        ),
        "teds": mean("teds"),
        "teds_s": mean("teds_s"),
        "teds_canonical": mean("teds_canonical"),
        "teds_s_canonical": mean("teds_s_canonical"),
        "text_recall": mean("text_recall"),
        "text_recall_canonical": mean("text_recall_canonical"),
        "mean_best_iou": mean("mean_best_iou"),
        "match_rate_iou_0_5": mean("match_rate_iou_0_5"),
        "cer": mean("cer"),
        "wer": mean("wer"),
        "cer_canonical": mean("cer_canonical"),
        "wer_canonical": mean("wer_canonical"),
        "predicted_cell_count": mean("predicted_cell_count"),
        "truth_cell_count": mean("truth_cell_count"),
        "cell_count_ratio": mean("cell_count_ratio"),
        "over_segmentation_rate": mean("over_segmentation_rate"),
        "empty_cell_count": mean("empty_cell_count"),
        "merged_cell_count": mean("merged_cell_count"),
        "truth_merged_cell_count": mean("truth_merged_cell_count"),
        "structure_preserved_rate": (
            round(
                sum(
                    row.get("structure_preserved") is True
                    for row in successes
                )
                / len(successes),
                4,
            )
            if any(
                row.get("structure_preserved") is not None
                for row in successes
            )
            else None
        ),
        "elapsed_p50_ms": (
            round(nearest_rank_percentile(latencies, 50), 2)
            if latencies
            else None
        ),
        "elapsed_p95_ms": (
            round(nearest_rank_percentile(latencies, 95), 2)
            if latencies
            else None
        ),
    }


def _common_success_summary(records, variants):
    """计算所有参与路线均成功样本上的配对指标。"""
    common = [
        record
        for record in records
        if all(
            record["variants"].get(variant, {}).get("status") == "success"
            for variant in variants
        )
    ]
    summary = {"sample_count": len(common), "variants": {}}
    for variant in variants:
        rows = [record["variants"][variant] for record in common]
        summary["variants"][variant] = {
            key: (
                round(
                    sum(row[key] for row in rows if row.get(key) is not None)
                    / len([
                        row for row in rows if row.get(key) is not None
                    ]),
                    4,
                )
                if any(row.get(key) is not None for row in rows)
                else None
            )
            for key in (
                "teds",
                "teds_s",
                "teds_canonical",
                "teds_s_canonical",
                "text_recall",
                "text_recall_canonical",
                "mean_best_iou",
                "match_rate_iou_0_5",
                "cer",
                "cer_canonical",
                "wer",
                "wer_canonical",
                "cell_count_ratio",
                "over_segmentation_rate",
            )
        }
    return summary


def _write_report(experiment_dir, metadata, records):
    aggregate = {
        variant: _aggregate_variant(records, variant)
        for variant in sorted(VARIANTS)
    }
    selected_variants = sorted(metadata["variants"])
    report = {
        "format_version": 1,
        "experiment": metadata,
        "sample_count": len(records),
        "variants": aggregate,
        "common_success": _common_success_summary(
            records,
            selected_variants,
        ),
    }
    atomic_write_json(experiment_dir / "report.json", report)
    rows = []
    for variant, metrics in aggregate.items():
        rows.append(
            "<tr>"
            f"<td>{variant}</td>"
            f"<td>{metrics['success_count']}/"
            f"{metrics['result_count']}</td>"
            f"<td>{metrics['teds']}</td>"
            f"<td>{metrics['teds_s']}</td>"
            f"<td>{metrics['teds_canonical']}</td>"
            f"<td>{metrics['text_recall']}</td>"
            f"<td>{metrics['mean_best_iou']}</td>"
            f"<td>{metrics['match_rate_iou_0_5']}</td>"
            f"<td>{metrics['over_segmentation_rate']}</td>"
            f"<td>{metrics['elapsed_p50_ms']}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>表格预处理五组实验</title>
<style>
body{{font:14px/1.5 system-ui;margin:24px;background:#f5f7f9}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #ccd5dd;padding:7px;text-align:left}}
</style></head><body>
<h1>表格预处理五组实验</h1>
<table><thead><tr><th>组</th><th>成功</th><th>TEDS</th>
<th>S-TEDS</th><th>Canonical TEDS</th><th>文本召回</th>
<th>平均 IoU</th><th>IoU≥0.5</th><th>过分割率</th>
<th>耗时 P50</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (experiment_dir / "report.html").write_text(
        document,
        encoding="utf-8",
    )
    return report


def _load_latest(results_path):
    latest = {}
    if not results_path.exists():
        return latest
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            latest[item["sample_id"]] = item
    return latest


def main(argv=None):
    import server

    parser = argparse.ArgumentParser(
        description="运行官方、自定义和表格预处理五组实验",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--variants",
        default="A,B,C,D,E",
    )
    parser.add_argument(
        "--experiment-id",
        default=time.strftime("table-preprocess-%Y%m%dT%H%M%S"),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/table-preprocess",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--custom-baseline-results",
        default=(
            "experiments/official-table-comparisons/"
            "official-vs-custom-100-20261119/results.jsonl"
        ),
    )
    args = parser.parse_args(argv)
    selected = {
        item.strip().upper()
        for item in args.variants.split(",")
        if item.strip()
    }
    if not selected or not selected <= VARIANTS:
        parser.error("variants 必须是 A/B/C/D/E")

    experiment_dir = (
        Path(args.resume).resolve()
        if args.resume
        else Path(args.output_dir).resolve() / args.experiment_id
    )
    if args.resume:
        metadata = json.loads(
            (experiment_dir / "experiment.json").read_text(
                encoding="utf-8"
            )
        )
        args.limit = metadata["selection"]["limit"]
        args.start = metadata["selection"]["start"]
        selected = set(metadata["variants"])
    else:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "experiment_id": experiment_dir.name,
            "selection": {
                "start": args.start,
                "limit": args.limit,
            },
            "variants": sorted(selected),
            "dataset_id": "fintabnet-c-structure-test",
            "custom_baseline_results": args.custom_baseline_results,
        }
        atomic_write_json(
            experiment_dir / "experiment.json",
            metadata,
        )

    logging.basicConfig(level=logging.WARNING)
    server._get_table_engine()
    dataset = DatasetRegistry().get_dataset(
        "fintabnet-c-structure-test"
    )
    results_path = experiment_dir / "results.jsonl"
    latest = _load_latest(results_path)
    baseline_latest = _load_latest(
        Path(args.custom_baseline_results).resolve()
    )
    with tempfile.TemporaryDirectory(
        dir=PROJECT_DIR,
        prefix=".table_preprocess_",
    ) as temp_dir:
        temp_root = Path(temp_dir)
        for index, sample in enumerate(
            dataset.iter_samples(
                start=args.start,
                limit=args.limit,
            ),
            start=args.start,
        ):
            if sample.sample_id in latest:
                continue
            original_path = temp_root / f"{index}_original.jpg"
            original_path.write_bytes(sample.image_bytes)
            original_image = cv2.imdecode(
                np.frombuffer(sample.image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if original_image is None:
                raise ValueError(f"无法读取样本图像: {sample.file_name}")
            parse_started = time.perf_counter()
            page, corrected_image = (
                server._official_parser.parse_page_with_image(
                    str(original_path),
                    page_num=index + 1,
                    is_pdf=False,
                    predict_options={
                        "use_table_recognition": True,
                        "use_doc_orientation_classify": True,
                        "use_doc_unwarping": True,
                    },
                )
            )
            parse_ms = round(
                (time.perf_counter() - parse_started) * 1000,
                2,
            )
            corrected = (
                corrected_image
                if corrected_image is not None
                else original_image
            )
            alignment = estimate_page_alignment(original_image, corrected)
            truth_with_bbox = {
                **sample.ground_truth,
                "table_bbox": sample.table_bbox,
            }
            aligned_truth = align_ground_truth(
                truth_with_bbox,
                alignment,
            )
            grounded_truth = (
                aligned_truth
                if alignment["status"] == "available"
                else None
            )
            selection_bbox = (
                aligned_truth.get("table_bbox")
                if grounded_truth is not None
                else sample.table_bbox
            )
            corrected_path = temp_root / f"{index}_corrected.png"
            cv2.imwrite(str(corrected_path), corrected)
            official_block = select_table_block(
                page,
                selection_bbox,
            )
            table_bbox = _resolve_table_bbox(
                page,
                selection_bbox,
                str(corrected_path),
                server._detect_table_blocks_with_fallback,
                False,
            )
            table_bbox_source = (
                "official"
                if official_block is not None
                and bbox_iou(
                    table_bbox,
                    official_block.get("bbox"),
                ) >= 0.999
                else "fallback"
                if table_bbox is not None
                else None
            )
            variants = {}
            if "A" in selected:
                official_cells = (
                    official_block.get("cells") or []
                    if official_block is not None
                    else []
                )
                official_metrics = block_metrics(
                    {
                        "cells": official_cells,
                        "table_html": (
                            official_block.get("table_html")
                            if official_block is not None
                            else ""
                        ),
                    },
                    sample.ground_truth,
                    grounded_truth,
                )
                if official_block is None:
                    official_metrics["status"] = "no_table_block"
                official_metrics["elapsed_ms"] = parse_ms
                official_statistics = _attach_table_statistics(
                    {"cells": official_cells},
                    sample.ground_truth,
                )
                official_metrics.update(official_statistics)
                official_metrics.pop("cells", None)
                variants["A"] = official_metrics

            if table_bbox is not None:
                crop, crop_info = crop_table_from_image(
                    corrected,
                    table_bbox,
                    padding=8,
                )
                enhanced, transform = enhance_table_crop(crop)
                identity = _identity_transform(
                    crop.shape[1],
                    crop.shape[0],
                )
                if "E" in selected:
                    raw_result = _run_current_table(
                        server,
                        crop,
                        crop_info,
                        identity,
                        temp_root / f"{index}_E.png",
                    )
                    if raw_result is None:
                        raw_result = {
                            "status": "override_failed",
                            "elapsed_ms": None,
                        }
                    raw_result.update(block_metrics(
                        {
                            "cells": raw_result.get("cells"),
                            "table_html": raw_result.get("table_html"),
                        },
                        sample.ground_truth,
                        grounded_truth,
                    ))
                    raw_result["status"] = (
                        "success"
                        if raw_result.get("cells")
                        or raw_result.get("table_html")
                        else "override_failed"
                    )
                    _attach_table_statistics(
                        raw_result,
                        sample.ground_truth,
                    )
                    variants["E"] = raw_result

                if "C" in selected:
                    enhanced_result = _run_current_table(
                        server,
                        enhanced,
                        crop_info,
                        transform,
                        temp_root / f"{index}_C.png",
                    )
                    if enhanced_result is None:
                        enhanced_result = {
                            "status": "override_failed",
                            "elapsed_ms": None,
                        }
                    enhanced_result.update(block_metrics(
                        {
                            "cells": enhanced_result.get("cells"),
                            "table_html": enhanced_result.get(
                                "table_html"
                            ),
                        },
                        sample.ground_truth,
                        grounded_truth,
                    ))
                    enhanced_result["status"] = (
                        "success"
                        if enhanced_result.get("cells")
                        or enhanced_result.get("table_html")
                        else "override_failed"
                    )
                    _attach_table_statistics(
                        enhanced_result,
                        sample.ground_truth,
                    )
                    variants["C"] = enhanced_result

                if "D" in selected:
                    if official_block is not None:
                        fused = _run_fused_table(
                            official_block,
                            enhanced,
                            crop_info,
                            transform,
                            server._ocr_service,
                        )
                    else:
                        fused = None
                    if fused is None:
                        fused = {
                            "status": "no_official_cells",
                            "elapsed_ms": None,
                        }
                    fused.update(block_metrics(
                        {
                            "cells": fused.get("cells"),
                            "table_html": fused.get("html"),
                        },
                        sample.ground_truth,
                        grounded_truth,
                    ))
                    fused["status"] = (
                        "success"
                        if fused.get("cells")
                        or fused.get("table_html")
                        else "no_official_cells"
                    )
                    _attach_table_statistics(
                        fused,
                        sample.ground_truth,
                        official_cells=(
                            official_block.get("cells") or []
                            if official_block is not None
                            else None
                        ),
                    )
                    variants["D"] = fused

            if "B" in selected:
                try:
                    variants["B"] = _run_custom_baseline(
                        server,
                        original_path,
                        sample,
                    )
                except Exception as exc:
                    baseline = baseline_latest.get(sample.sample_id)
                    if baseline is not None:
                        variants["B"] = {
                            **baseline["custom"],
                            "elapsed_ms": baseline["custom"].get(
                                "injection_ms"
                            ),
                            "source": "imported current custom baseline",
                            "baseline_rerun_error": str(exc),
                            "grounding_status": "available",
                        }
                    else:
                        variants["B"] = {
                            "status": "baseline_failed",
                            "elapsed_ms": None,
                            "error": str(exc),
                        }

            for variant in selected:
                variants.setdefault(variant, {
                    "status": "no_table_crop",
                    "elapsed_ms": None,
                })

            item = {
                "sample_id": sample.sample_id,
                "file_name": sample.file_name,
                "parse_ms": parse_ms,
                "page_alignment": alignment,
                "official_table_bbox": (
                    official_block.get("bbox")
                    if official_block is not None
                    else None
                ),
                "table_crop_bbox": table_bbox,
                "table_bbox_source": table_bbox_source,
                "variants": variants,
            }
            latest[sample.sample_id] = item
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"{sample.sample_id}: "
                + ", ".join(
                    f"{key}={value['status']}"
                    for key, value in variants.items()
                ),
                flush=True,
            )

    records = list(latest.values())
    report = _write_report(experiment_dir, metadata, records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"experiment_dir={experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
