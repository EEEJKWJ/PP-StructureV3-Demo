"""只读模型对比报告服务。"""

import json
import re
from pathlib import Path


REPORT_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$"
)


class ComparisonReports:
    """枚举并读取模型对比报告。"""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()

    def list_reports(self):
        if not self.root_dir.is_dir():
            return []
        reports = []
        for directory in self.root_dir.iterdir():
            if (
                not directory.is_dir()
                or not REPORT_ID_PATTERN.fullmatch(directory.name)
            ):
                continue
            summary_path = directory / "comparison_summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            reports.append(summary)
        reports.sort(
            key=lambda report: report.get("generated_at") or "",
            reverse=True,
        )
        return reports

    def get_report(self, comparison_id):
        directory = self._directory(comparison_id)
        report_path = directory / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"对比报告不存在: {comparison_id}"
            )
        return json.loads(report_path.read_text(encoding="utf-8"))

    def export_path(self, comparison_id, export_format):
        if export_format not in {"json", "csv", "html"}:
            raise ValueError("导出格式必须是 json、csv 或 html")
        directory = self._directory(comparison_id)
        path = directory / f"report.{export_format}"
        if not path.is_file():
            raise FileNotFoundError(f"导出文件不存在: {path.name}")
        return path

    def _directory(self, comparison_id):
        if not REPORT_ID_PATTERN.fullmatch(comparison_id or ""):
            raise ValueError("对比 ID 无效")
        directory = (self.root_dir / comparison_id).resolve()
        if self.root_dir not in directory.parents:
            raise ValueError("对比路径越界")
        return directory
