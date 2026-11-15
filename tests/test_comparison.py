import csv
import json
import logging
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import server
from app.evaluation.comparison import (
    classify_result,
    create_comparison,
    generate_comparison_report,
)
from app.evaluation.comparison_reports import ComparisonReports
from app.evaluation.runner import ExperimentStore


DATASET = {
    "name": "Fake",
    "version": "1",
    "fingerprint": "fingerprint-a",
    "input_path": "fake.parquet",
}


def sample_result(
    sample_id,
    index,
    *,
    success=True,
    elapsed=10,
    cells=None,
    ground_truth=None,
    errors=None,
):
    cells = cells if cells is not None else [{
        "row": 0,
        "col": 0,
        "text": "A",
        "bbox": [0, 0, 10, 10],
    }]
    return {
        "sample_id": sample_id,
        "file_name": f"{index}.jpg",
        "global_index": index,
        "status": "success" if success else "failed",
        "cell_count": len(cells) if success else 0,
        "html_present": success,
        "elapsed_ms": elapsed,
        "row_count": 1 if success else 0,
        "col_count": 1 if success else 0,
        "cells": cells if success else [],
        "ground_truth": ground_truth,
        "errors": errors or [],
    }


def build_model_experiment(
    comparison_dir,
    model_id,
    results,
    fingerprint="fingerprint-a",
):
    store = ExperimentStore.create(
        output_root=comparison_dir,
        logger=logging.getLogger("test-comparison"),
        dataset={**DATASET, "fingerprint": fingerprint},
        model={"id": model_id, "name": model_id},
        selection={"limit": len(results), "start": 0, "batch_size": 1},
        retries=0,
        experiment_id=model_id,
    )
    for result in results:
        store.append_result(result)
    store.write_metrics_report()
    return store


class ComparisonReportTests(unittest.TestCase):
    def _make_comparison(self, temp_dir, first, second, second_fp="fingerprint-a"):
        comparison_dir = create_comparison(
            output_root=temp_dir,
            dataset=DATASET,
            model_ids=["model-a", "model-b"],
            selection={"offset": 0, "limit": 2, "batch_size": 1},
            retries=0,
            comparison_id="comparison-test",
        )
        build_model_experiment(
            comparison_dir,
            "model-a",
            first,
        )
        build_model_experiment(
            comparison_dir,
            "model-b",
            second,
            fingerprint=second_fp,
        )
        return comparison_dir

    def test_completion_and_latency_winners_without_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._make_comparison(
                temp_dir,
                [
                    sample_result("s:0", 0, success=True, elapsed=20),
                    sample_result("s:1", 1, success=True, elapsed=10),
                ],
                [
                    sample_result("s:0", 0, success=False, elapsed=5),
                    sample_result("s:1", 1, success=True, elapsed=15),
                ],
            )
            summary = generate_comparison_report(comparison_dir)
            samples = [
                json.loads(line)
                for line in (
                    comparison_dir / "comparison_samples.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            summary["comparison"]["completion_wins"],
            {"model-a": 1},
        )
        self.assertEqual(
            summary["comparison"]["latency_wins"],
            {"model-a": 1},
        )
        self.assertEqual(
            summary["comparison"]["quality_comparable_samples"],
            0,
        )
        self.assertEqual(
            samples[0]["completion_winner"],
            "model-a",
        )
        self.assertIsNone(samples[1]["quality_winner"])
        self.assertEqual(
            samples[1]["quality_status"],
            "not_comparable_no_ground_truth",
        )

    def test_quality_winner_uses_ground_truth_cells(self):
        truth = [{
            "row": 0,
            "col": 0,
            "text": "A",
            "bbox": [0, 0, 10, 10],
        }]
        exact = sample_result(
            "s:0",
            0,
            cells=truth,
            ground_truth={"cells": truth},
        )
        wrong = sample_result(
            "s:0",
            0,
            cells=[{
                "row": 0,
                "col": 0,
                "text": "B",
                "bbox": [0, 0, 10, 10],
            }],
            ground_truth={"cells": truth},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._make_comparison(
                temp_dir,
                [exact],
                [wrong],
            )

            summary = generate_comparison_report(comparison_dir)

        self.assertEqual(
            summary["comparison"]["quality_wins"],
            {"model-a": 1},
        )
        self.assertEqual(
            summary["comparison"]["quality_comparable_samples"],
            1,
        )

    def test_dataset_fingerprint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._make_comparison(
                temp_dir,
                [sample_result("s:0", 0)],
                [sample_result("s:0", 0)],
                second_fp="different",
            )

            with self.assertRaises(ValueError):
                generate_comparison_report(comparison_dir)

    def test_exports_are_generated_and_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._make_comparison(
                temp_dir,
                [sample_result("s:0", 0)],
                [sample_result("s:0", 0)],
            )
            report_path = comparison_dir / "report.json"
            generate_comparison_report(comparison_dir)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            with (
                comparison_dir / "report.csv"
            ).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            report_html = (
                comparison_dir / "report.html"
            ).read_text(encoding="utf-8")

        self.assertEqual(report["comparison_id"], "comparison-test")
        self.assertEqual(len(report["samples"]), 1)
        self.assertEqual(rows[0]["sample_id"], "s:0")
        self.assertIn("model-a_status", rows[0])
        self.assertIn("<html", report_html)

    def test_error_classification(self):
        self.assertEqual(
            classify_result(sample_result("s:0", 0)),
            "success",
        )
        self.assertEqual(
            classify_result({
                "status": "failed",
                "cell_count": 0,
                "errors": ["模型返回空结果"],
            }),
            "empty_result",
        )
        self.assertEqual(
            classify_result({
                "status": "failed",
                "cell_count": 0,
                "errors": ["未检出表格单元格"],
            }),
            "no_cells",
        )

    def test_report_service_lists_and_exports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._make_comparison(
                temp_dir,
                [sample_result("s:0", 0)],
                [sample_result("s:0", 0)],
            )
            generate_comparison_report(comparison_dir)
            reports = ComparisonReports(temp_dir)

            listed = reports.list_reports()
            detail = reports.get_report("comparison-test")
            export = reports.export_path("comparison-test", "csv")

        self.assertEqual(len(listed), 1)
        self.assertEqual(detail["comparison_id"], "comparison-test")
        self.assertEqual(export.suffix, ".csv")

    def test_comparison_api_list_detail_and_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._make_comparison(
                temp_dir,
                [sample_result("s:0", 0)],
                [sample_result("s:0", 0)],
            )
            generate_comparison_report(comparison_dir)
            original = server._comparison_reports
            server._comparison_reports = ComparisonReports(temp_dir)
            try:
                client = TestClient(server.app)
                listed = client.get("/api/comparisons")
                detail = client.get(
                    "/api/comparisons/comparison-test"
                )
                export = client.get(
                    "/api/comparisons/comparison-test/export/json"
                )
            finally:
                server._comparison_reports = original

        self.assertTrue(listed.json()["ok"])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(export.status_code, 200)
        self.assertIn(
            "application/json",
            export.headers["content-type"],
        )

    def test_invalid_export_format_and_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = ComparisonReports(temp_dir)

            with self.assertRaises(ValueError):
                reports.export_path("comparison-test", "exe")
            with self.assertRaises(ValueError):
                reports.get_report("../outside")


if __name__ == "__main__":
    unittest.main()
