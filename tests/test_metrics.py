import logging
import tempfile
import unittest

from app.evaluation.metrics import (
    bbox_iou,
    compute_metrics,
    grounded_cell_metrics,
    nearest_rank_percentile,
)
from app.evaluation.runner import ExperimentStore


def result(
    sample_id,
    *,
    status="success",
    cells=1,
    table_box_count=1,
    elapsed_ms=100,
    html=True,
    attempts=1,
    ground_truth=None,
):
    return {
        "sample_id": sample_id,
        "global_index": int(sample_id.split(":")[-1]),
        "status": status,
        "cell_count": cells,
        "table_box_count": table_box_count,
        "elapsed_ms": elapsed_ms,
        "html_present": html,
        "attempts": attempts,
        "cells": [],
        "ground_truth": ground_truth,
    }


class MetricPrimitiveTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        values = [1, 2, 3, 4, 100]

        self.assertEqual(nearest_rank_percentile(values, 50), 3)
        self.assertEqual(nearest_rank_percentile(values, 95), 100)
        self.assertIsNone(nearest_rank_percentile([], 95))

    def test_bbox_iou(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(
            bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]),
            1 / 3,
        )

    def test_perfect_grounded_cell_metrics(self):
        cells = [{
            "row": 0,
            "col": 0,
            "text": "A B",
            "bbox": [0, 0, 10, 10],
        }]

        metrics = grounded_cell_metrics(cells, cells)

        self.assertEqual(metrics["mean_best_iou"], 1)
        self.assertEqual(metrics["match_rate_iou_0_5"], 1)
        self.assertEqual(metrics["text_recall"], 1)
        self.assertEqual(metrics["cer"], 0)
        self.assertEqual(metrics["wer"], 0)

    def test_partial_grounded_metrics_keep_zero_and_error_values(self):
        predictions = [
            {
                "row": 0,
                "col": 0,
                "text": "A",
                "bbox": [0, 0, 10, 10],
            },
            {
                "row": 1,
                "col": 0,
                "text": "X",
                "bbox": [10, 0, 20, 10],
            },
        ]
        ground_truth = [
            {
                "row": 0,
                "col": 0,
                "text": "A",
                "bbox": [0, 0, 10, 10],
            },
            {
                "row": 1,
                "col": 0,
                "text": "C",
                "bbox": [10, 0, 20, 10],
            },
        ]

        metrics = grounded_cell_metrics(predictions, ground_truth)

        self.assertEqual(metrics["mean_best_iou"], 1)
        self.assertEqual(metrics["match_rate_iou_0_5"], 1)
        self.assertEqual(metrics["text_recall"], 0.5)
        self.assertAlmostEqual(metrics["cer"], 0.5)
        self.assertAlmostEqual(metrics["wer"], 0.5)


class AggregateMetricTests(unittest.TestCase):
    def test_detection_rate_uses_annotated_samples_as_denominator(self):
        results = [
            result("s:0", cells=5),
            result("s:1", status="failed", cells=0),
            result("s:2", cells=7),
            result("s:3", table_box_count=0, cells=9),
        ]

        report = compute_metrics(results)
        metrics = report["metrics"]

        self.assertEqual(metrics["annotated_sample_count"], 3)
        self.assertEqual(metrics["detected_sample_count"], 2)
        self.assertAlmostEqual(
            metrics["table_detection_rate"],
            2 / 3,
            places=4,
        )
        self.assertEqual(metrics["success_count"], 3)

    def test_average_cells_excludes_failures_and_not_applicable(self):
        results = [
            result("s:0", cells=4),
            result("s:1", cells=8),
            result("s:2", status="failed", cells=0),
        ]

        metrics = compute_metrics(results)["metrics"]

        self.assertEqual(metrics["average_cells"], 6)
        self.assertEqual(metrics["failure_count"], 1)
        self.assertEqual(metrics["latency_p50_ms"], 100)
        self.assertEqual(metrics["latency_p95_ms"], 100)
        self.assertAlmostEqual(
            metrics["throughput_samples_per_second"],
            10.0,
        )

    def test_no_table_annotations_marks_detection_not_applicable(self):
        report = compute_metrics([
            result("s:0", table_box_count=0, cells=2),
        ])

        self.assertIsNone(report["metrics"]["table_detection_rate"])
        self.assertEqual(
            report["metric_status"]["table_detection_rate"],
            "not_applicable_no_table_annotations",
        )

    def test_missing_ground_truth_has_explicit_statuses(self):
        report = compute_metrics([result("s:0")])

        self.assertIsNone(report["metrics"]["cell_mean_best_iou"])
        self.assertIsNone(report["metrics"]["teds"])
        self.assertEqual(
            report["metric_status"]["cell_mean_best_iou"],
            "not_available_no_ground_truth",
        )
        self.assertEqual(
            report["metric_status"]["teds"],
            "not_available_no_ground_truth",
        )

    def test_html_truth_marks_teds_as_not_implemented(self):
        report = compute_metrics([
            result(
                "s:0",
                ground_truth={
                    "cells": [],
                    "html": "<table></table>",
                },
            ),
        ])

        self.assertEqual(
            report["metric_status"]["teds"],
            "not_implemented",
        )
        self.assertEqual(
            report["ground_truth_coverage"]["html_samples"],
            1,
        )

    def test_store_recomputes_metrics_and_syncs_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = logging.getLogger("test-metrics-store")
            store = ExperimentStore.create(
                output_root=temp_dir,
                logger=logger,
                dataset={"fingerprint": "data", "name": "Fake"},
                model={"id": "fake-model"},
                selection={"limit": 1, "start": 0, "batch_size": 1},
                retries=0,
                experiment_id="metrics-test",
            )
            store.append_result(result("s:0"))

            report = store.write_metrics_report()

            self.assertTrue(store.metrics_path.exists())
            self.assertTrue(store.summary_path.exists())
            summary = __import__("json").loads(
                store.summary_path.read_text(encoding="utf-8")
            )

        self.assertEqual(report["metrics"]["result_count"], 1)
        self.assertEqual(summary["metrics"], report["metrics"])
        self.assertEqual(
            summary["metric_status"],
            report["metric_status"],
        )


if __name__ == "__main__":
    unittest.main()
