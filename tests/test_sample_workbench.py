import io
import logging
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from fastapi.testclient import TestClient

import server
from app.datasets.registry import DatasetRegistry
from app.evaluation.workbench import SampleWorkbenchService


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(
        buffer,
        format="JPEG",
    )
    return buffer.getvalue()


def write_shard(path, count):
    rows = []
    for index in range(count):
        rows.append({
            "image": {"bytes": image_bytes(), "path": ""},
            "category_ids": [1],
            "image_id": index + 1,
            "boxes": [[1, 2, 30, 20]],
            "width": 32,
            "height": 24,
            "file_name": f"sample_{index}.jpg",
            "ocr_words": ["A", "B"],
            "ocr_boxes": [[1, 1, 5, 5], [6, 1, 10, 5]],
        })
    pq.write_table(pa.Table.from_pylist(rows), path)


class SampleWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test-workbench")

    def test_missing_dataset_is_reported_unavailable(self):
        registry = DatasetRegistry(
            fintabnet_path=Path("missing-dataset")
        )

        datasets = registry.list_datasets()

        self.assertFalse(datasets[0]["available"])
        self.assertIn("不存在", datasets[0]["error"])

    def test_list_samples_paginates_with_stable_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 5)
            registry = DatasetRegistry(fintabnet_path=shard)
            service = SampleWorkbenchService(registry, self.logger)

            page = service.list_samples(
                "fintabnet-test",
                offset=2,
                limit=2,
            )

        self.assertEqual(page["total_samples"], 5)
        self.assertEqual(page["offset"], 2)
        self.assertEqual(len(page["samples"]), 2)
        self.assertEqual(
            page["samples"][0]["sample_id"],
            "shard:2:sample_2.jpg",
        )
        self.assertTrue(page["has_more"])

    def test_get_sample_and_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 2)
            service = SampleWorkbenchService(
                DatasetRegistry(fintabnet_path=shard),
                self.logger,
            )

            sample = service.get_sample(
                "fintabnet-test",
                "shard:1:sample_1.jpg",
            )
            content = service.get_image(
                "fintabnet-test",
                sample.sample_id,
            )

        self.assertEqual(sample.width, 32)
        self.assertEqual(sample.table_bbox, [1.0, 2.0, 30.0, 20.0])
        self.assertTrue(content.startswith(b"\xff\xd8"))
        self.assertEqual(sample.ground_truth["status"], "not_available")

    def test_run_sample_returns_prediction_and_no_truth_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 1)
            service = SampleWorkbenchService(
                DatasetRegistry(fintabnet_path=shard),
                self.logger,
            )

            result = service.run_sample(
                "fintabnet-test",
                "shard:0:sample_0.jpg",
                lambda image_path, bbox: {
                    "cells": [{
                        "row": 0,
                        "col": 0,
                        "text": "A",
                        "bbox": [0, 0, 1, 1],
                    }],
                    "html": "<table><tr><td>A</td></tr></table>",
                    "model": {"id": "fake"},
                    "elapsed_ms": 5,
                },
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "success")
        self.assertIn("<table>", result["prediction"]["html"])
        self.assertEqual(
            result["ground_truth"]["status"],
            "not_available",
        )

    def test_run_sample_distinguishes_no_table_and_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 1)
            service = SampleWorkbenchService(
                DatasetRegistry(fintabnet_path=shard),
                self.logger,
            )

            no_table = service.run_sample(
                "fintabnet-test",
                "shard:0:sample_0.jpg",
                lambda image_path, bbox: None,
            )

            def failing(image_path, bbox):
                raise RuntimeError("model down")

            failed = service.run_sample(
                "fintabnet-test",
                "shard:0:sample_0.jpg",
                failing,
            )

        self.assertEqual(no_table["status"], "no_table")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("model down", failed["error"])

    def test_workbench_api_lists_samples_and_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 2)
            original_registry = server._dataset_registry
            original_workbench = server._sample_workbench
            registry = DatasetRegistry(fintabnet_path=shard)
            server._dataset_registry = registry
            server._sample_workbench = SampleWorkbenchService(
                registry,
                self.logger,
            )
            try:
                client = TestClient(server.app)
                datasets = client.get("/api/datasets")
                samples = client.get(
                    "/api/datasets/fintabnet-test/samples",
                    params={"offset": 0, "limit": 1},
                )
                sample_id = samples.json()["samples"][0]["sample_id"]
                encoded = quote(sample_id, safe="")
                image = client.get(
                    f"/api/datasets/fintabnet-test/samples/"
                    f"{encoded}/image"
                )
            finally:
                server._dataset_registry = original_registry
                server._sample_workbench = original_workbench

        self.assertTrue(datasets.json()["datasets"][0]["available"])
        self.assertEqual(samples.status_code, 200)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/jpeg")

    def test_invalid_pagination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 1)
            service = SampleWorkbenchService(
                DatasetRegistry(fintabnet_path=shard),
                self.logger,
            )

            with self.assertRaises(ValueError):
                service.list_samples(
                    "fintabnet-test",
                    offset=-1,
                )
            with self.assertRaises(ValueError):
                service.list_samples(
                    "fintabnet-test",
                    limit=101,
                )

    def test_workbench_api_runs_single_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.parquet"
            write_shard(shard, 1)
            original_registry = server._dataset_registry
            original_workbench = server._sample_workbench
            original_recognizer = server.recognize_table_structure
            registry = DatasetRegistry(fintabnet_path=shard)
            server._dataset_registry = registry
            server._sample_workbench = SampleWorkbenchService(
                registry,
                self.logger,
            )
            server.recognize_table_structure = (
                lambda image_path, bbox: {
                    "cells": [{
                        "row": 0,
                        "col": 0,
                        "text": "A",
                        "bbox": [0, 0, 1, 1],
                    }],
                    "html": "<table><tr><td>A</td></tr></table>",
                    "model": {"id": "api-fake"},
                    "elapsed_ms": 2,
                }
            )
            try:
                encoded = quote(
                    "shard:0:sample_0.jpg",
                    safe="",
                )
                response = TestClient(server.app).post(
                    f"/api/datasets/fintabnet-test/samples/"
                    f"{encoded}/run"
                )
            finally:
                server._dataset_registry = original_registry
                server._sample_workbench = original_workbench
                server.recognize_table_structure = original_recognizer

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(
            response.json()["prediction"]["model"]["id"],
            "api-fake",
        )


if __name__ == "__main__":
    unittest.main()
