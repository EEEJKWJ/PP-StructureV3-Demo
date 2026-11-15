"""可恢复的表格识别批量实验运行器。"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.metrics import compute_metrics
from app.evaluation.teds import calculate_html_metrics


EXPERIMENT_FORMAT_VERSION = 1
EXPERIMENT_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class ExperimentStore:
    """管理一个独立实验目录及可恢复结果。"""

    def __init__(self, directory, logger):
        self.directory = Path(directory).resolve()
        self.log = logger
        self.experiment_path = self.directory / "experiment.json"
        self.results_path = self.directory / "results.jsonl"
        self.failures_path = self.directory / "failures.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.metrics_path = self.directory / "metrics.json"
        self.temp_dir = self.directory / ".tmp"
        if not self.experiment_path.exists():
            raise FileNotFoundError(
                f"实验目录缺少 experiment.json: {self.directory}"
            )
        self.experiment = json.loads(
            self.experiment_path.read_text(encoding="utf-8")
        )
        self.results_path.touch(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    @classmethod
    def create(
        cls,
        output_root,
        logger,
        dataset,
        model,
        selection,
        retries,
        experiment_id=None,
    ):
        output_root = Path(output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        experiment_id = experiment_id or (
            time.strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
            raise ValueError(f"实验 ID 无效: {experiment_id}")

        directory = output_root / experiment_id
        if directory.exists():
            raise FileExistsError(f"实验目录已存在: {directory}")
        directory.mkdir(parents=True)
        metadata = {
            "format_version": EXPERIMENT_FORMAT_VERSION,
            "experiment_id": experiment_id,
            "created_at": utc_now(),
            "dataset": dataset,
            "model": model,
            "selection": selection,
            "retries": retries,
        }
        atomic_write_json(directory / "experiment.json", metadata)
        (directory / "results.jsonl").touch()
        (directory / ".tmp").mkdir()
        store = cls(directory, logger)
        store.log.info("实验目录: %s", directory)
        return store

    def append_result(self, result):
        line = json.dumps(result, ensure_ascii=False)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_latest_results(self):
        """读取结果，同一 sample_id 以最后一条记录为准。"""
        latest = {}
        invalid_lines = 0
        with self.results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                sample_id = result.get("sample_id")
                if sample_id:
                    latest[sample_id] = result
        if invalid_lines:
            self.log.warning(
                "结果文件有 %s 条无效或截断记录，已跳过",
                invalid_lines,
            )
        return latest

    def completed_sample_ids(self, retry_failed=False):
        completed = set()
        for sample_id, result in self.load_latest_results().items():
            status = result.get("status")
            if status == "success" or (
                status == "failed" and not retry_failed
            ):
                completed.add(sample_id)
        return completed

    def write_failures_and_summary(self, run_id):
        results = self.load_latest_results()
        failures = [
            result
            for result in results.values()
            if result.get("status") != "success"
        ]

        temporary = self.failures_path.with_name(
            self.failures_path.name + ".tmp"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            for failure in sorted(
                failures,
                key=lambda item: item.get("global_index", 0),
            ):
                handle.write(
                    json.dumps(failure, ensure_ascii=False) + "\n"
                )
        os.replace(temporary, self.failures_path)

        successes = [
            result
            for result in results.values()
            if result.get("status") == "success"
        ]
        elapsed_values = [
            float(result.get("elapsed_ms", 0) or 0)
            for result in results.values()
        ]
        summary = {
            "format_version": EXPERIMENT_FORMAT_VERSION,
            "experiment_id": self.experiment["experiment_id"],
            "run_id": run_id,
            "updated_at": utc_now(),
            "dataset": self.experiment["dataset"],
            "model": self.experiment["model"],
            "selection": self.experiment["selection"],
            "completed": len(results),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": (
                len(successes) / len(results) if results else 0.0
            ),
            "html_successes": sum(
                bool(result.get("html_present"))
                for result in successes
            ),
            "total_cells": sum(
                int(result.get("cell_count", 0) or 0)
                for result in successes
            ),
            "total_elapsed_ms": round(sum(elapsed_values), 2),
        }
        self.write_metrics_report(
            summary=summary,
            results=results,
        )
        return summary

    def write_metrics_report(
        self,
        summary=None,
        results=None,
        include_teds=True,
    ):
        """重算指标并同步写入独立报告和 summary。"""
        results = results or self.load_latest_results()
        report = compute_metrics(
            results.values(),
            experiment=self.experiment,
            include_teds=include_teds,
        )
        atomic_write_json(self.metrics_path, report)

        if summary is None:
            if self.summary_path.exists():
                summary = json.loads(
                    self.summary_path.read_text(encoding="utf-8")
                )
            else:
                summary = {
                    "format_version": EXPERIMENT_FORMAT_VERSION,
                    "experiment_id": self.experiment["experiment_id"],
                    "dataset": self.experiment["dataset"],
                    "model": self.experiment["model"],
                    "selection": self.experiment["selection"],
                }
        summary["metrics"] = report["metrics"]
        summary["metric_status"] = report["metric_status"]
        summary["metric_definitions"] = report["definitions"]
        summary["ground_truth_coverage"] = report[
            "ground_truth_coverage"
        ]
        summary["metrics_updated_at"] = report["source"]["generated_at"]
        atomic_write_json(self.summary_path, summary)
        return report


class BatchRunner:
    """逐样本运行表格识别并持久化可恢复结果。"""

    def __init__(
        self,
        dataset,
        recognizer,
        store,
        model_info,
        logger,
        retries=0,
    ):
        if retries < 0:
            raise ValueError("retries 不能小于 0")
        self.dataset = dataset
        self.recognizer = recognizer
        self.store = store
        self.model_info = model_info
        self.log = logger
        self.retries = retries

    def run(
        self,
        limit=None,
        start=0,
        batch_size=64,
        retry_failed=False,
        progress=None,
    ):
        run_id = (
            time.strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        completed = self.store.completed_sample_ids(
            retry_failed=retry_failed
        )
        processed = 0
        skipped = 0
        for sample in self.dataset.iter_samples(
            limit=limit,
            start=start,
            batch_size=batch_size,
        ):
            if sample.sample_id in completed:
                skipped += 1
                if progress is not None:
                    progress(sample, skipped=True)
                continue
            result = self._run_sample(sample, run_id)
            result["status"] = (
                "success" if result["cells"] else "failed"
            )
            self.store.append_result(result)
            processed += 1
            if progress is not None:
                progress(result, skipped=False)

        summary = self.store.write_failures_and_summary(run_id)
        summary["processed_this_run"] = processed
        summary["skipped_completed"] = skipped
        return summary

    def _run_sample(self, sample, run_id):
        started_at = time.perf_counter()
        attempts = 0
        errors = []
        last_result = None
        image_path = self.store.temp_dir / (
            f"sample_{uuid.uuid4().hex}.jpg"
        )

        try:
            if not sample.image_bytes:
                raise ValueError("样本图片为空")
            image_path.write_bytes(sample.image_bytes)
            for attempt in range(1, self.retries + 2):
                attempts = attempt
                try:
                    result = self.recognizer(
                        str(image_path),
                        sample.table_bbox,
                    )
                    if result is None:
                        raise RuntimeError("模型返回空结果")
                    last_result = result
                    if result.get("cells"):
                        break
                    raise RuntimeError("未检出表格单元格")
                except Exception as exc:
                    errors.append(
                        f"attempt {attempt}: "
                        f"{exc.__class__.__name__}: {exc}"
                    )
                    self.log.warning(
                        "%s 第 %s 次识别失败: %s",
                        sample.sample_id,
                        attempt,
                        exc,
                    )
        except Exception as exc:
            errors.append(
                f"prepare: {exc.__class__.__name__}: {exc}"
            )
            self.log.warning(
                "%s 样本准备失败: %s",
                sample.sample_id,
                exc,
            )
        finally:
            image_path.unlink(missing_ok=True)

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
        cells = (
            last_result.get("cells", [])
            if isinstance(last_result, dict)
            else []
        )
        table_html = (
            last_result.get("html", "")
            if isinstance(last_result, dict)
            else ""
        )
        truth_html = (
            (sample.ground_truth or {}).get("html")
            if sample.ground_truth
            else ""
        )
        html_metrics = calculate_html_metrics(
            table_html,
            truth_html,
        )
        result = {
            "run_id": run_id,
            **sample.locating_info(),
            "attempts": attempts,
            "elapsed_ms": elapsed_ms,
            "model": (
                last_result.get("model", self.model_info)
                if isinstance(last_result, dict)
                else self.model_info
            ),
            "cell_count": len(cells),
            "row_count": (
                last_result.get("row_count", 0)
                if isinstance(last_result, dict)
                else 0
            ),
            "col_count": (
                last_result.get("col_count", 0)
                if isinstance(last_result, dict)
                else 0
            ),
            "html_present": bool(table_html),
            "cells": cells,
            "html": table_html,
            "html_metrics": html_metrics,
            "ground_truth": sample.ground_truth,
            "errors": errors,
        }
        return result
