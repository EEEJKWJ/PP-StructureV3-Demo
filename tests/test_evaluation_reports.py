import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import server
from app.evaluation.reports import EvaluationReports


def write_experiment(root, experiment_id, generated_at, detection_rate):
    directory = Path(root) / experiment_id
    directory.mkdir(parents=True)
    experiment = {
        "experiment_id": experiment_id,
        "created_at": "2026-11-15T00:00:00+00:00",
        "model": {"id": "model-a", "name": "Model A"},
        "dataset": {"version": "test", "fingerprint": "data"},
    }
    metrics = {
        "format_version": 1,
        "source": {
            "generated_at": generated_at,
            "experiment_id": experiment_id,
            "model": experiment["model"],
            "dataset": experiment["dataset"],
            "result_count": 3,
        },
        "metrics": {
            "result_count": 3,
            "table_detection_rate": detection_rate,
            "average_cells": 12.5,
            "latency_p50_ms": 10,
            "latency_p95_ms": 20,
            "throughput_samples_per_second": 2,
            "teds": None,
        },
        "metric_status": {"teds": "not_available_no_ground_truth"},
        "ground_truth_coverage": {
            "cell_samples": 0,
            "html_samples": 0,
            "sample_count": 3,
        },
        "definitions": {},
    }
    (directory / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False),
        encoding="utf-8",
    )
    return directory


class EvaluationReportTests(unittest.TestCase):
    def test_lists_only_experiments_with_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_experiment(
                temp_dir,
                "with-metrics",
                "2026-11-15T01:00:00+00:00",
                0.9,
            )
            incomplete = Path(temp_dir) / "without-metrics"
            incomplete.mkdir()
            (incomplete / "experiment.json").write_text(
                '{"experiment_id": "without-metrics"}',
                encoding="utf-8",
            )
            reports = EvaluationReports(temp_dir)

            listed = reports.list_reports()

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["experiment_id"], "with-metrics")
        self.assertEqual(
            listed[0]["metrics"]["table_detection_rate"],
            0.9,
        )

    def test_latest_returns_most_recent_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_experiment(
                temp_dir,
                "older",
                "2026-11-15T01:00:00+00:00",
                0.5,
            )
            write_experiment(
                temp_dir,
                "newer",
                "2026-11-15T02:00:00+00:00",
                1.0,
            )
            report = EvaluationReports(temp_dir).latest_metrics()

        self.assertEqual(report["source"]["experiment_id"], "newer")
        self.assertEqual(
            report["metrics"]["table_detection_rate"],
            1.0,
        )

    def test_invalid_id_and_missing_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = EvaluationReports(temp_dir)

            with self.assertRaises(ValueError):
                reports.get_metrics("../outside")
            with self.assertRaises(FileNotFoundError):
                reports.get_metrics("missing")

    def test_report_api_lists_and_returns_latest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_experiment(
                temp_dir,
                "api-report",
                "2026-11-15T03:00:00+00:00",
                0.75,
            )
            original = server._evaluation_reports
            server._evaluation_reports = EvaluationReports(temp_dir)
            try:
                client = TestClient(server.app)
                listed = client.get("/api/evaluations")
                latest = client.get(
                    "/api/evaluations/latest/metrics"
                )
                specific = client.get(
                    "/api/evaluations/api-report/metrics"
                )
            finally:
                server._evaluation_reports = original

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(
            listed.json()["reports"][0]["experiment_id"],
            "api-report",
        )
        self.assertEqual(latest.status_code, 200)
        self.assertTrue(latest.json()["ok"])
        self.assertEqual(
            specific.json()["metrics"]["table_detection_rate"],
            0.75,
        )

    def test_empty_report_api_returns_explicit_placeholder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = server._evaluation_reports
            server._evaluation_reports = EvaluationReports(temp_dir)
            try:
                response = TestClient(server.app).get(
                    "/api/evaluations/latest/metrics"
                )
            finally:
                server._evaluation_reports = original

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertIn("暂无评测结果", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
