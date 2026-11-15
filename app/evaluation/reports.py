"""只读批量实验指标报告。"""

import json
import re
from pathlib import Path


EXPERIMENT_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$"
)


class EvaluationReports:
    """枚举实验目录并读取 metrics.json。"""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()

    @staticmethod
    def _read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def list_reports(self):
        if not self.root_dir.is_dir():
            return []
        reports = []
        for directory in self.root_dir.iterdir():
            if (
                not directory.is_dir()
                or not EXPERIMENT_ID_PATTERN.fullmatch(directory.name)
            ):
                continue
            metrics_path = directory / "metrics.json"
            experiment_path = directory / "experiment.json"
            if not metrics_path.is_file() or not experiment_path.is_file():
                continue
            try:
                metrics = self._read_json(metrics_path)
                experiment = self._read_json(experiment_path)
            except (OSError, json.JSONDecodeError):
                continue
            summary_path = directory / "summary.json"
            summary = {}
            if summary_path.is_file():
                try:
                    summary = self._read_json(summary_path)
                except (OSError, json.JSONDecodeError):
                    summary = {}
            source = metrics.get("source") or {}
            reports.append({
                "experiment_id": experiment.get("experiment_id"),
                "created_at": experiment.get("created_at"),
                "updated_at": (
                    source.get("generated_at")
                    or summary.get("updated_at")
                ),
                "model": source.get("model")
                or experiment.get("model"),
                "dataset": source.get("dataset")
                or experiment.get("dataset"),
                "metrics": metrics.get("metrics"),
                "metric_status": metrics.get("metric_status"),
                "ground_truth_coverage": metrics.get(
                    "ground_truth_coverage"
                ),
                "_sort_mtime": metrics_path.stat().st_mtime_ns,
            })
        reports.sort(
            key=lambda item: (
                item.get("updated_at") or "",
                item["_sort_mtime"],
            ),
            reverse=True,
        )
        for report in reports:
            report.pop("_sort_mtime", None)
        return reports

    def latest_metrics(self):
        reports = self.list_reports()
        if not reports:
            return None
        return self.get_metrics(reports[0]["experiment_id"])

    def get_metrics(self, experiment_id):
        if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id or ""):
            raise ValueError("实验 ID 无效")
        directory = (self.root_dir / experiment_id).resolve()
        if self.root_dir not in directory.parents:
            raise ValueError("实验路径越界")
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"实验指标不存在: {experiment_id}"
            )
        return self._read_json(metrics_path)
