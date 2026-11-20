"""表格文本 canonical 视图及其配套指标。"""

import html
import re
import unicodedata

from app.evaluation.metrics import grounded_cell_metrics
from app.evaluation.teds import calculate_html_metrics


_DOT_LEADER = re.compile(r"(?:\s*\.{3,}\s*)+")
_WHITESPACE = re.compile(r"\s+")


def canonicalize_table_text(text):
    """去除排版噪声，同时保留数字分隔符和有效标点。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    value = _DOT_LEADER.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def canonicalize_cells(cells):
    """复制单元格并只规范化文字字段。"""
    return [
        {
            **cell,
            "text": canonicalize_table_text(cell.get("text")),
        }
        for cell in cells or []
    ]


def canonicalize_html(table_html):
    """规范化 HTML 文本节点，保留标签和跨行列属性。"""
    if not table_html:
        return ""

    def replace_text(match):
        normalized = canonicalize_table_text(
            html.unescape(match.group(1))
        )
        return f">{html.escape(normalized)}<"

    return re.sub(r">([^<>]+)<", replace_text, str(table_html))


def canonical_metrics(predicted_html, predicted_cells, truth):
    """返回 canonical TEDS、文字召回、CER 和 WER。"""
    truth_cells = truth.get("cells") or []
    html_metrics = calculate_html_metrics(
        canonicalize_html(predicted_html),
        canonicalize_html(truth.get("html")),
    )
    cell_metrics = grounded_cell_metrics(
        canonicalize_cells(predicted_cells),
        canonicalize_cells(truth_cells),
    ) or {}
    return {
        "teds_canonical": html_metrics.get("teds"),
        "teds_s_canonical": html_metrics.get("teds_s"),
        "text_recall_canonical": cell_metrics.get("text_recall"),
        "cer_canonical": cell_metrics.get("cer"),
        "wer_canonical": cell_metrics.get("wer"),
    }
