"""分析五组表格预处理实验的配对结果和场景子集。"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.datasets.registry import DatasetRegistry
from app.evaluation.metrics import nearest_rank_percentile
from app.evaluation.runner import atomic_write_json
from scripts.run_table_preprocess_experiment import (
    _aggregate_variant,
    _common_success_summary,
)


METRIC_KEYS = (
    "teds",
    "teds_s",
    "teds_canonical",
    "teds_s_canonical",
    "text_recall",
    "text_recall_canonical",
    "mean_best_iou",
    "match_rate_iou_0_5",
    "cer",
    "wer",
    "over_segmentation_rate",
)


def _load_records(experiment_dir):
    latest = {}
    with (experiment_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                latest[item["sample_id"]] = item
    return list(latest.values())


def _bbox(element):
    box = element.find("bndbox")
    return [
        float(box.findtext("xmin")),
        float(box.findtext("ymin")),
        float(box.findtext("xmax")),
        float(box.findtext("ymax")),
    ]


def _header_levels(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = root.findall("object")
    rows = sorted(
        (
            _bbox(item)
            for item in objects
            if (item.findtext("name") or "") == "table row"
        ),
        key=lambda box: (box[1], box[0]),
    )
    headers = [
        _bbox(item)
        for item in objects
        if (item.findtext("name") or "") == "table column header"
    ]
    levels = []
    for _, y1, _, y2 in headers:
        levels.append(sum(
            1
            for _, row_y1, _, row_y2 in rows
            if row_y2 > y1 + 0.5 and row_y1 < y2 - 0.5
        ))
    return max(levels, default=1)


def _sample_features(dataset, limit):
    features = {}
    for sample in dataset.iter_samples(limit=limit):
        cells = sample.ground_truth.get("cells") or []
        empty_ratio = (
            sum(not str(cell.get("text") or "").strip() for cell in cells)
            / len(cells)
            if cells
            else 0.0
        )
        features[sample.sample_id] = {
            "cell_count": len(cells),
            "empty_cell_ratio": empty_ratio,
            "header_levels": _header_levels(sample.shard_path),
        }
    return features


def _pairwise(records, variants):
    common = [
        record
        for record in records
        if all(
            record["variants"].get(variant, {}).get("status") == "success"
            for variant in variants
        )
    ]
    result = {"sample_count": len(common), "comparisons": {}}
    for variant in variants:
        if variant == "B":
            continue
        comparison = {}
        for metric in METRIC_KEYS:
            higher = lower = equal = 0
            deltas = []
            for record in common:
                left = record["variants"][variant].get(metric)
                right = record["variants"]["B"].get(metric)
                if left is None or right is None:
                    continue
                delta = left - right
                deltas.append(delta)
                if delta > 1e-6:
                    higher += 1
                elif delta < -1e-6:
                    lower += 1
                else:
                    equal += 1
            lower_is_better = metric in ("cer", "wer", "over_segmentation_rate")
            comparison[metric] = {
                "higher_than_B": higher,
                "lower_than_B": lower,
                "equal_to_B": equal,
                "favorable_for_variant": (
                    lower if lower_is_better else higher
                ),
                "favorable_for_B": (
                    higher if lower_is_better else lower
                ),
                "mean_delta": (
                    round(sum(deltas) / len(deltas), 4)
                    if deltas
                    else None
                ),
            }
        result["comparisons"][variant] = comparison
    return result


def _top_differences(records, variant, metric="teds", limit=10):
    rows = []
    for record in records:
        current = record["variants"].get(variant, {})
        baseline = record["variants"].get("B", {})
        if (
            current.get("status") != "success"
            or baseline.get("status") != "success"
            or current.get(metric) is None
            or baseline.get(metric) is None
        ):
            continue
        rows.append({
            "sample_id": record["sample_id"],
            "variant": current[metric],
            "baseline_b": baseline[metric],
            "delta": round(current[metric] - baseline[metric], 6),
        })
    return sorted(rows, key=lambda item: item["delta"])[:limit]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="分析五组表格预处理实验",
    )
    parser.add_argument("experiment_dir")
    args = parser.parse_args(argv)
    experiment_dir = Path(args.experiment_dir).resolve()
    metadata = json.loads(
        (experiment_dir / "experiment.json").read_text(encoding="utf-8")
    )
    records = _load_records(experiment_dir)
    variants = sorted(metadata["variants"])
    dataset = DatasetRegistry().get_dataset(metadata["dataset_id"])
    features = _sample_features(
        dataset,
        metadata["selection"]["limit"],
    )

    cell_counts = [
        features[record["sample_id"]]["cell_count"]
        for record in records
    ]
    empty_ratios = [
        features[record["sample_id"]]["empty_cell_ratio"]
        for record in records
    ]
    large_threshold = nearest_rank_percentile(cell_counts, 75)
    sparse_threshold = nearest_rank_percentile(empty_ratios, 75)
    subsets = {
        "all": list(records),
        "large_table": [
            record
            for record in records
            if features[record["sample_id"]]["cell_count"]
            >= large_threshold
        ],
        "multi_level_header": [
            record
            for record in records
            if features[record["sample_id"]]["header_levels"] >= 2
        ],
        "sparse_table": [
            record
            for record in records
            if features[record["sample_id"]]["empty_cell_ratio"]
            >= sparse_threshold
        ],
    }
    subset_report = {}
    for name, subset in subsets.items():
        subset_report[name] = {
            "sample_count": len(subset),
            "variants": {
                variant: _aggregate_variant(subset, variant)
                for variant in variants
            },
            "common_success": _common_success_summary(
                subset,
                variants,
            ),
        }

    report = {
        "format_version": 1,
        "experiment_id": metadata["experiment_id"],
        "thresholds": {
            "large_table_cell_count": large_threshold,
            "sparse_empty_cell_ratio": sparse_threshold,
            "multi_level_header_levels": 2,
        },
        "page_alignment": [
            {
                "status": status,
                "method": method,
                "count": count,
            }
            for (status, method), count in Counter(
                (
                    record["page_alignment"]["status"],
                    record["page_alignment"]["method"],
                )
                for record in records
            ).most_common()
        ],
        "table_bbox_source": dict(Counter(
            record.get("table_bbox_source")
            for record in records
        )),
        "pairwise_vs_B": _pairwise(records, variants),
        "subsets": subset_report,
        "largest_degradation_vs_B": {
            variant: _top_differences(records, variant)
            for variant in variants
            if variant != "B"
        },
    }
    atomic_write_json(experiment_dir / "analysis.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"analysis={experiment_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
