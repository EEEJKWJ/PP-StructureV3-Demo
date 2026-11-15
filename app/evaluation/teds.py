"""基于有序树编辑距离的 TEDS 与 S-TEDS。"""

import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser

from apted import APTED, Config


@dataclass
class TableTreeNode:
    kind: str
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    children: list["TableTreeNode"] = field(default_factory=list)


class _TableTreeParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = None
        self.current_row = None
        self.current_cell = None
        self.cell_text = []
        self._ignored_table_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = dict(attrs)
        if tag == "table":
            if self.root is None:
                self.root = TableTreeNode("table")
            else:
                self._ignored_table_depth += 1
            return
        if self.root is None or self._ignored_table_depth:
            return
        if tag == "tr":
            self._finish_cell()
            self.current_row = TableTreeNode("row")
            self.root.children.append(self.current_row)
        elif tag in ("td", "th"):
            if self.current_row is None:
                self.current_row = TableTreeNode("row")
                self.root.children.append(self.current_row)
            self._finish_cell()
            self.current_cell = TableTreeNode(
                "cell",
                rowspan=self._span(attrs.get("rowspan")),
                colspan=self._span(attrs.get("colspan")),
            )
            self.cell_text = []
        elif tag == "br" and self.current_cell is not None:
            self.cell_text.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table" and self.root is not None:
            if self._ignored_table_depth:
                self._ignored_table_depth -= 1
            return
        if self._ignored_table_depth:
            return
        if tag in ("td", "th"):
            self._finish_cell()
        elif tag == "tr":
            self._finish_cell()
            self.current_row = None

    def handle_data(self, data):
        if (
            self.root is not None
            and not self._ignored_table_depth
            and self.current_cell is not None
        ):
            self.cell_text.append(data)

    def close(self):
        super().close()
        self._finish_cell()

    @staticmethod
    def _span(value):
        try:
            return min(1000, max(1, int(value or 1)))
        except (TypeError, ValueError):
            return 1

    def _finish_cell(self):
        if self.current_cell is None:
            return
        self.current_cell.text = normalize_text(
            "".join(self.cell_text)
        )
        if self.current_row is not None:
            self.current_row.children.append(self.current_cell)
        self.current_cell = None
        self.cell_text = []


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def parse_table_tree(table_html):
    if not table_html or not str(table_html).strip():
        raise ValueError("HTML 为空")
    parser = _TableTreeParser()
    parser.feed(str(table_html))
    parser.close()
    if parser.root is None:
        raise ValueError("HTML 中没有 table")
    if not parser.root.children or not any(
        row.children for row in parser.root.children
    ):
        raise ValueError("HTML 表格没有单元格")
    return parser.root


def _node_name(node, include_text):
    if node.kind != "cell":
        return node.kind
    structure = f"cell:{node.rowspan}:{node.colspan}"
    return f"{structure}:{node.text}" if include_text else structure


class _TEDSConfig(Config):
    def __init__(self, include_text):
        self.include_text = include_text

    def rename(self, first, second):
        return int(
            _node_name(first, self.include_text)
            != _node_name(second, self.include_text)
        )


def _tree_size(root):
    return 1 + sum(_tree_size(child) for child in root.children)


def tree_edit_distance(first_root, second_root, include_text):
    """使用 APTED 计算有序树编辑距离。"""
    return APTED(
        first_root,
        second_root,
        _TEDSConfig(include_text),
    ).compute_edit_distance()


def calculate_html_metrics(prediction_html, truth_html):
    """计算一个样本的 TEDS 和 S-TEDS。"""
    if not truth_html:
        return {
            "teds": None,
            "teds_s": None,
            "status": "not_available_no_html_truth",
            "error": "",
        }
    if not prediction_html:
        return {
            "teds": None,
            "teds_s": None,
            "status": "missing_prediction_html",
            "error": "",
        }
    try:
        prediction_tree = parse_table_tree(prediction_html)
        truth_tree = parse_table_tree(truth_html)
    except Exception as exc:
        return {
            "teds": None,
            "teds_s": None,
            "status": "parse_error",
            "error": str(exc),
        }

    prediction_count = _tree_size(prediction_tree)
    truth_count = _tree_size(truth_tree)
    denominator = max(prediction_count, truth_count)
    distance_teds = tree_edit_distance(
        prediction_tree,
        truth_tree,
        include_text=True,
    )
    distance_structure = tree_edit_distance(
        prediction_tree,
        truth_tree,
        include_text=False,
    )
    return {
        "teds": round(1 - distance_teds / denominator, 6),
        "teds_s": round(1 - distance_structure / denominator, 6),
        "status": "available",
        "error": "",
    }
