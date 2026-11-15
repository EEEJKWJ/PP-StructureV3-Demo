import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from app.datasets.base import TableDataset, TableSample
from app.datasets.fintabnet import FinTabNetDataset
from app.evaluation.runner import BatchRunner, ExperimentStore


def jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(
        buffer,
        format="JPEG",
    )
    return buffer.getvalue()


def write_fintabnet_shard(path, rows):
    records = []
    for file_name, categories, boxes in rows:
        records.append({
            "image": {
                "bytes": jpeg_bytes(),
                "path": file_name,
            },
            "category_ids": categories,
            "image_id": len(records) + 1,
            "boxes": boxes,
            "width": 20,
            "height": 20,
            "file_name": file_name,
            "ocr_words": ["A", "B"],
            "ocr_boxes": [
                [1, 1, 5, 5],
                [6, 1, 10, 5],
            ],
        })
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


class FakeDataset(TableDataset):
    def __init__(self, count):
        self.samples = [
            TableSample(
                sample_id=f"fake:{index}:{index}.jpg",
                file_name=f"{index}.jpg",
                row_index=index,
                global_index=index,
                shard_path="fake.parquet",
                image_bytes=jpeg_bytes(),
                width=20,
                height=20,
                table_bbox=[0, 0, 20, 20],
                table_box_count=1,
                used_full_image_fallback=False,
                image_id=index,
                ocr_word_count=2,
                ocr_box_count=2,
            )
            for index in range(count)
        ]

    def describe(self):
        return {
            "name": "Fake",
            "version": "1",
            "input_path": "fake",
            "shard_count": 1,
            "total_samples": len(self.samples),
            "shards": [],
            "fingerprint": "fake-fingerprint",
            "fingerprint_method": "test",
        }

    def iter_samples(self, limit=None, start=0, batch_size=64):
        selected = 0
        for sample in self.samples[start:]:
            yield sample
            selected += 1
            if limit is not None and selected >= limit:
                return


def success_result(sample_id="sample"):
    return {
        "cells": [{
            "row": 0,
            "col": 0,
            "text": sample_id,
            "bbox": [0, 0, 1, 1],
        }],
        "html": "<table><tr><td>A</td></tr></table>",
        "model": {"id": "fake-model"},
        "row_count": 1,
        "col_count": 1,
    }


class FinTabNetDatasetTests(unittest.TestCase):
    def test_directory_description_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_fintabnet_shard(
                root / "part-0.parquet",
                [
                    ("a.jpg", [1], [[1, 2, 10, 12]]),
                    ("b.jpg", [1], [[2, 3, 12, 13]]),
                ],
            )
            write_fintabnet_shard(
                root / "part-1.parquet",
                [
                    ("c.jpg", [1], [[3, 4, 14, 15]]),
                    ("d.jpg", [1], [[4, 5, 16, 17]]),
                ],
            )

            dataset = FinTabNetDataset(root)
            samples = list(dataset.iter_samples())

        description = dataset.describe()
        self.assertEqual(description["shard_count"], 2)
        self.assertEqual(description["total_samples"], 4)
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].sample_id, "part-0:0:a.jpg")
        self.assertEqual(samples[2].sample_id, "part-1:0:c.jpg")

    def test_range_and_single_file_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "single.parquet"
            write_fintabnet_shard(
                shard,
                [
                    ("a.jpg", [1], [[1, 2, 10, 12]]),
                    ("b.jpg", [1], [[2, 3, 12, 13]]),
                    ("c.jpg", [1], [[3, 4, 14, 15]]),
                ],
            )
            dataset = FinTabNetDataset(shard)

            samples = list(
                dataset.iter_samples(start=1, limit=1, batch_size=1)
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].file_name, "b.jpg")
        self.assertEqual(samples[0].global_index, 1)

    def test_missing_table_box_uses_full_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "fallback.parquet"
            write_fintabnet_shard(
                shard,
                [("a.jpg", [2], [[1, 2, 10, 12]])],
            )

            sample = next(FinTabNetDataset(shard).iter_samples())

        self.assertTrue(sample.used_full_image_fallback)
        self.assertEqual(sample.table_bbox, [0, 0, 20, 20])
        self.assertEqual(sample.table_box_count, 0)


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test-batch-runner")
        self.model_info = {"id": "fake-model", "name": "Fake"}

    def _create_store(self, output_root, dataset, retries=0):
        return ExperimentStore.create(
            output_root=output_root,
            logger=self.logger,
            dataset=dataset.describe(),
            model=self.model_info,
            selection={"limit": 2, "start": 0, "batch_size": 1},
            retries=retries,
            experiment_id="test-experiment",
        )

    def test_run_writes_experiment_results_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = FakeDataset(2)
            store = self._create_store(temp_dir, dataset)
            calls = []

            def recognizer(image_path, bbox):
                calls.append((image_path, bbox))
                return success_result(Path(image_path).name)

            runner = BatchRunner(
                dataset,
                recognizer,
                store,
                self.model_info,
                self.logger,
            )
            summary = runner.run(limit=2, batch_size=1)

            self.assertTrue(store.experiment_path.exists())
            self.assertTrue(store.results_path.exists())
            self.assertTrue(store.summary_path.exists())
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["successes"], 2)
            self.assertEqual(summary["html_successes"], 2)
            self.assertEqual(summary["processed_this_run"], 2)

    def test_resume_skips_completed_successes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = FakeDataset(2)
            store = self._create_store(temp_dir, dataset)
            first_calls = []

            def first_recognizer(image_path, bbox):
                first_calls.append(image_path)
                return success_result()

            BatchRunner(
                dataset,
                first_recognizer,
                store,
                self.model_info,
                self.logger,
            ).run(limit=1, batch_size=1)

            resumed_store = ExperimentStore(store.directory, self.logger)
            resumed_calls = []

            def resumed_recognizer(image_path, bbox):
                resumed_calls.append(image_path)
                return success_result()

            summary = BatchRunner(
                dataset,
                resumed_recognizer,
                resumed_store,
                self.model_info,
                self.logger,
            ).run(limit=2, batch_size=1)

        self.assertEqual(len(first_calls), 1)
        self.assertEqual(len(resumed_calls), 1)
        self.assertEqual(summary["skipped_completed"], 1)
        self.assertEqual(summary["completed"], 2)

    def test_retry_counts_attempts_when_then_successful(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = FakeDataset(1)
            store = self._create_store(
                temp_dir,
                dataset,
                retries=1,
            )
            attempts = {"count": 0}

            def flaky_recognizer(image_path, bbox):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("temporary")
                return success_result()

            runner = BatchRunner(
                dataset,
                flaky_recognizer,
                store,
                self.model_info,
                self.logger,
                retries=1,
            )
            runner.run(limit=1, batch_size=1)
            result = store.load_latest_results()["fake:0:0.jpg"]

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(attempts["count"], 2)

    def test_retry_failed_reprocesses_failed_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = FakeDataset(1)
            store = self._create_store(temp_dir, dataset)

            def failing_recognizer(image_path, bbox):
                raise RuntimeError("always fails")

            BatchRunner(
                dataset,
                failing_recognizer,
                store,
                self.model_info,
                self.logger,
            ).run(limit=1, batch_size=1)
            first_results = store.load_latest_results()

            resumed_store = ExperimentStore(store.directory, self.logger)
            summary = BatchRunner(
                dataset,
                lambda image_path, bbox: success_result(),
                resumed_store,
                self.model_info,
                self.logger,
            ).run(
                limit=1,
                batch_size=1,
                retry_failed=True,
            )
            final_results = resumed_store.load_latest_results()

        self.assertEqual(
            first_results["fake:0:0.jpg"]["status"],
            "failed",
        )
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["processed_this_run"], 1)
        self.assertEqual(
            final_results["fake:0:0.jpg"]["status"],
            "success",
        )

    def test_truncated_result_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = FakeDataset(1)
            store = self._create_store(temp_dir, dataset)
            valid = {
                "sample_id": "fake:0:0.jpg",
                "status": "success",
                "global_index": 0,
            }
            store.results_path.write_text(
                json.dumps(valid) + "\n{truncated",
                encoding="utf-8",
            )

            results = store.load_latest_results()

        self.assertEqual(results["fake:0:0.jpg"], valid)

    def test_empty_image_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = FakeDataset(1)
            dataset.samples[0].image_bytes = b""
            store = self._create_store(temp_dir, dataset)
            runner = BatchRunner(
                dataset,
                lambda image_path, bbox: success_result(),
                store,
                self.model_info,
                self.logger,
            )

            summary = runner.run(limit=1, batch_size=1)
            result = store.load_latest_results()["fake:0:0.jpg"]

        self.assertEqual(summary["failures"], 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("样本图片为空", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
