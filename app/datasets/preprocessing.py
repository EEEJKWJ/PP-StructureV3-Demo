"""FinTabNet.c-Structure XML 预处理。"""

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from app.evaluation.runner import atomic_write_json


class StructureAnnotationError(ValueError):
    """结构 XML 无法规范化为单元格网格。"""


def _bbox_from_element(element):
    box = element.find("bndbox")
    if box is None:
        raise StructureAnnotationError("对象缺少 bndbox")
    return [
        float(box.findtext("xmin")),
        float(box.findtext("ymin")),
        float(box.findtext("xmax")),
        float(box.findtext("ymax")),
    ]


def _intersection_length(first_start, first_end, second_start, second_end):
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _intersection_area(first, second):
    width = _intersection_length(first[0], first[2], second[0], second[2])
    height = _intersection_length(first[1], first[3], second[1], second[3])
    return width * height


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _contains_center(box, point_x, point_y):
    return box[0] <= point_x <= box[2] and box[1] <= point_y <= box[3]


def _span_indices(span_box, boxes, horizontal):
    indices = []
    span_start = span_box[0] if horizontal else span_box[1]
    span_end = span_box[2] if horizontal else span_box[3]
    for index, box in enumerate(boxes):
        box_start = box[0] if horizontal else box[1]
        box_end = box[2] if horizontal else box[3]
        overlap = _intersection_length(
            span_start,
            span_end,
            box_start,
            box_end,
        )
        minimum = min(span_end - span_start, box_end - box_start)
        if minimum > 0 and overlap / minimum >= 0.45:
            indices.append(index)
    return indices


def _contiguous(indices):
    return bool(indices) and indices == list(
        range(indices[0], indices[-1] + 1)
    )


def _load_words(words_path):
    if not Path(words_path).is_file():
        return []
    data = json.loads(Path(words_path).read_text(encoding="utf-8"))
    return [
        item
        for item in data
        if item.get("text") and item.get("bbox")
    ]


def _assign_words(cells, words):
    assignments = [[] for _ in cells]
    for word in words:
        box = word["bbox"]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        containing = [
            index
            for index, cell in enumerate(cells)
            if _contains_center(cell["bbox"], center_x, center_y)
        ]
        if containing:
            cell_index = min(
                containing,
                key=lambda index: _area(cells[index]["bbox"]),
            )
        else:
            overlaps = [
                (_intersection_area(box, cell["bbox"]), index)
                for index, cell in enumerate(cells)
            ]
            overlap, cell_index = max(overlaps, default=(0, -1))
            if overlap <= 0:
                continue
        assignments[cell_index].append(word)
    return assignments


def _word_lines(words):
    ordered = sorted(
        words,
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        ),
    )
    lines = []
    current = []
    current_center = None
    for word in ordered:
        center = (word["bbox"][1] + word["bbox"][3]) / 2
        height = max(1.0, word["bbox"][3] - word["bbox"][1])
        if current and abs(center - current_center) > height * 0.6:
            lines.append(current)
            current = []
            current_center = None
        current.append(word)
        centers = [
            (item["bbox"][1] + item["bbox"][3]) / 2
            for item in current
        ]
        current_center = sum(centers) / len(centers)
    if current:
        lines.append(current)

    result = []
    for line in lines:
        line.sort(key=lambda item: item["bbox"][0])
        result.append(" ".join(item["text"] for item in line))
    return "\n".join(result)


def cells_to_structured_html(cells):
    if not cells:
        return ""
    max_row = max(cell["row"] + cell["rowspan"] - 1 for cell in cells)
    rows = []
    by_row = {}
    for cell in cells:
        by_row.setdefault(cell["row"], []).append(cell)
    for row_index in range(max_row + 1):
        row_cells = sorted(
            by_row.get(row_index, []),
            key=lambda cell: cell["col"],
        )
        cells_html = []
        for cell in row_cells:
            attrs = []
            if cell["rowspan"] > 1:
                attrs.append(f' rowspan="{cell["rowspan"]}"')
            if cell["colspan"] > 1:
                attrs.append(f' colspan="{cell["colspan"]}"')
            text = html.escape(cell.get("text") or "").replace(
                "\n",
                "<br>",
            )
            cells_html.append(
                f"<td{''.join(attrs)}>{text}</td>"
            )
        if cells_html:
            rows.append("<tr>" + "".join(cells_html) + "</tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def parse_structure_xml(xml_path, words_path=None):
    """把 FinTabNet.c XML 规范化为单元格和 HTML。"""
    root = ET.parse(xml_path).getroot()
    objects = []
    for element in root.findall("object"):
        name = (element.findtext("name") or "").strip()
        objects.append({
            "name": name,
            "bbox": _bbox_from_element(element),
        })

    tables = [item["bbox"] for item in objects if item["name"] == "table"]
    if not tables:
        raise StructureAnnotationError("缺少 table 对象")
    table_bbox = tables[0]

    rows = sorted(
        [
            item["bbox"]
            for item in objects
            if item["name"] == "table row"
        ],
        key=lambda box: (box[1], box[0]),
    )
    columns = sorted(
        [
            item["bbox"]
            for item in objects
            if item["name"] == "table column"
        ],
        key=lambda box: (box[0], box[1]),
    )
    if not rows or not columns:
        raise StructureAnnotationError("缺少 table row 或 table column")

    warnings = []
    spans = []
    for item in objects:
        if item["name"] != "table spanning cell":
            continue
        row_indices = _span_indices(item["bbox"], rows, horizontal=False)
        col_indices = _span_indices(item["bbox"], columns, horizontal=True)
        if not _contiguous(row_indices) or not _contiguous(col_indices):
            warnings.append(
                f"跨单元格无法对齐行列: {item['bbox']}"
            )
            continue
        spans.append({
            "bbox": item["bbox"],
            "row": row_indices[0],
            "col": col_indices[0],
            "rowspan": len(row_indices),
            "colspan": len(col_indices),
        })
    spans.sort(key=lambda item: -_area(item["bbox"]))

    occupied = {}
    cells = []
    for span in spans:
        covered = [
            (row, col)
            for row in range(span["row"], span["row"] + span["rowspan"])
            for col in range(
                span["col"],
                span["col"] + span["colspan"],
            )
        ]
        if any(position in occupied for position in covered):
            warnings.append(
                f"重叠跨单元格已跳过: {span['bbox']}"
            )
            continue
        for position in covered:
            occupied[position] = True
        cells.append({**span, "text": ""})

    for row_index, row_box in enumerate(rows):
        for col_index, col_box in enumerate(columns):
            if (row_index, col_index) in occupied:
                continue
            cell_bbox = [
                col_box[0],
                row_box[1],
                col_box[2],
                row_box[3],
            ]
            if _area(cell_bbox) <= 0:
                warnings.append(
                    f"空几何单元格 row={row_index}, col={col_index}"
                )
                continue
            cells.append({
                "row": row_index,
                "col": col_index,
                "rowspan": 1,
                "colspan": 1,
                "bbox": cell_bbox,
                "text": "",
            })

    words = _load_words(words_path) if words_path else []
    assignments = _assign_words(cells, words)
    for cell, cell_words in zip(cells, assignments):
        cell["text"] = _word_lines(cell_words)
    cells.sort(key=lambda cell: (cell["row"], cell["col"]))
    structure_html = cells_to_structured_html(cells)

    object_counts = {}
    for item in objects:
        object_counts[item["name"]] = object_counts.get(
            item["name"],
            0,
        ) + 1
    return {
        "width": int(root.findtext("./size/width") or 0),
        "height": int(root.findtext("./size/height") or 0),
        "table_bbox": table_bbox,
        "row_count": len(rows),
        "col_count": len(columns),
        "spanning_cell_count": len(spans),
        "word_count": len(words),
        "object_counts": object_counts,
        "cells": cells,
        "html": structure_html,
        "warnings": warnings,
    }


def preprocess_sample(root, split, xml_path):
    root = Path(root)
    xml_path = Path(xml_path)
    stem = xml_path.stem
    image_path = root / "images" / f"{stem}.jpg"
    words_path = root / "words" / f"{stem}_words.json"
    if not image_path.is_file():
        raise FileNotFoundError(f"样本图像不存在: {image_path}")
    structure = parse_structure_xml(xml_path, words_path)
    return {
        "sample_id": f"{split}:{stem}",
        "split": split,
        "stem": stem,
        "image_path": str(image_path.relative_to(root)),
        "xml_path": str(xml_path.relative_to(root)),
        "words_path": str(words_path.relative_to(root)),
        **structure,
    }


def prepare_split(
    root,
    split,
    output_path,
    limit=None,
    overwrite=False,
    progress=None,
):
    """预处理一个划分并写入 JSONL 缓存。"""
    root = Path(root).resolve()
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"划分目录不存在: {split_dir}")
    output_path = Path(output_path).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() and not overwrite:
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"缓存已存在但缺少元数据: {output_path}"
        )

    xml_files = sorted(split_dir.glob("*.xml"))
    if limit is not None:
        xml_files = xml_files[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    failures = []
    written = 0
    warnings = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for index, xml_path in enumerate(xml_files, 1):
            try:
                record = preprocess_sample(root, split, xml_path)
            except Exception as exc:
                failures.append({
                    "file": str(xml_path),
                    "error": str(exc),
                })
                continue
            warnings += len(record["warnings"])
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            written += 1
            if progress and (
                index == 1 or index % 100 == 0 or index == len(xml_files)
            ):
                progress(index, len(xml_files), written, len(failures))
    temporary.replace(output_path)
    metadata = {
        "format_version": 1,
        "source": str(root),
        "split": split,
        "xml_files": len(xml_files),
        "written": written,
        "failures": failures,
        "warnings": warnings,
        "output_path": str(output_path),
    }
    atomic_write_json(meta_path, metadata)
    return metadata
