"""把整表 OCR 文字按几何关系回填到官方表格单元格。"""

from app.datasets.preprocessing import cells_to_structured_html
from app.services.table_preprocess import page_bbox_to_processed


def _intersection_area(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _contains_center(box, line_bbox):
    center_x = (line_bbox[0] + line_bbox[2]) / 2
    center_y = (line_bbox[1] + line_bbox[3]) / 2
    return box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]


def fuse_table_text(table_block, ocr_lines, crop_info, transform):
    """保留官方单元格结构，只替换单元格文字。"""
    cells = list(table_block.get("cells") or [])
    if not cells:
        return None
    processed_boxes = []
    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox or list(bbox) == [0, 0, 0, 0]:
            processed_boxes.append(None)
        else:
            processed_boxes.append(
                page_bbox_to_processed(
                    bbox,
                    crop_info,
                    transform,
                )
            )

    assignments = [[] for _ in cells]
    unassigned = []
    for line in ocr_lines or []:
        line_bbox = line.get("bbox")
        if not line_bbox:
            unassigned.append(line)
            continue
        containing = [
            index
            for index, box in enumerate(processed_boxes)
            if box is not None and _contains_center(box, line_bbox)
        ]
        if containing:
            cell_index = min(
                containing,
                key=lambda index: (
                    (processed_boxes[index][2] - processed_boxes[index][0])
                    * (processed_boxes[index][3] - processed_boxes[index][1])
                ),
            )
        else:
            overlaps = [
                (
                    _intersection_area(box, line_bbox),
                    index,
                )
                for index, box in enumerate(processed_boxes)
                if box is not None
            ]
            overlap, cell_index = max(
                overlaps,
                default=(0, -1),
            )
            if overlap <= 0:
                unassigned.append(line)
                continue
        assignments[cell_index].append(line)

    updated = []
    assigned_count = 0
    for cell, cell_lines in zip(cells, assignments):
        if not cell_lines:
            new_cell = dict(cell)
            new_cell["text"] = ""
            updated.append(new_cell)
            continue
        assigned_count += 1
        cell_lines.sort(
            key=lambda line: (
                (line["bbox"][1] + line["bbox"][3]) / 2,
                line["bbox"][0],
            )
        )
        new_cell = dict(cell)
        new_cell["text"] = "\n".join(
            str(line.get("text") or "")
            for line in cell_lines
            if line.get("text")
        )
        updated.append(new_cell)
    return updated, {
        "assigned_cell_count": assigned_count,
        "assigned_line_count": (
            len(ocr_lines or []) - len(unassigned)
        ),
        "unassigned_line_count": len(unassigned),
    }


def build_fused_table(table_block, ocr_lines, crop_info, transform):
    """执行整表 OCR 文字融合并返回完整表格结果。"""
    fused = fuse_table_text(
        table_block,
        ocr_lines,
        crop_info,
        transform,
    )
    if fused is None:
        return None
    cells, assignment = fused
    return {
        "cells": cells,
        "html": cells_to_structured_html(cells),
        "assignment": assignment,
        "table_model": table_block.get("table_model"),
        "official_table": {
            "html": table_block.get("table_html") or "",
            "model": table_block.get("table_model"),
            "cell_bbox_available": bool(
                table_block.get("cell_bbox_available")
            ),
        },
    }
