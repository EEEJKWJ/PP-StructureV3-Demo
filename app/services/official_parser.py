"""官方 PP-StructureV3 整页解析与 Demo 结果转换。"""

import threading

from PIL import Image

from app.engines.base import parse_html_table, restore_cell_newlines
from app.services.structure import StructureService


TITLE_LABELS = {
    "doc_title",
    "paragraph_title",
}

TEXT_LABELS = {
    "abstract",
    "algorithm",
    "content",
    "figure_title",
    "footer",
    "header",
    "reference",
    "reference_content",
    "table_title",
    "text",
}

FORMULA_LABELS = {
    "formula",
    "formula_number",
}


def _as_bbox(value):
    """把官方 bbox 规范为整数列表。"""
    if value is None:
        return [0, 0, 0, 0]
    return [round(float(item)) for item in value[:4]]


def _table_cells(table_result):
    """把官方表格 HTML 和单元格框转换为 Demo cells。"""
    table_html = str(table_result.get("pred_html") or "")
    raw_boxes = table_result.get("cell_box_list") or []
    quads = []
    for box in raw_boxes:
        x1, y1, x2, y2 = [float(item) for item in box[:4]]
        quads.append([
            x1,
            y1,
            x2,
            y1,
            x2,
            y2,
            x1,
            y2,
        ])
    cells = parse_html_table(
        table_html,
        [quads] if quads else None,
        0,
        0,
    )
    restore_cell_newlines(
        cells,
        raw_boxes,
        table_result.get("table_ocr_pred") or {},
    )
    has_bbox = any(
        cell.get("bbox") and list(cell["bbox"]) != [0, 0, 0, 0]
        for cell in cells
    )
    if not has_bbox:
        for cell in cells:
            cell.pop("bbox", None)
    return table_html, cells, has_bbox


def _structured_item(block):
    """把单个官方 block 转换为当前 structured.content 项。"""
    label = str(block.get("label") or "")
    content = block.get("ocr_text") or ""
    if label == "table":
        cells = block.get("cells") or []
        html = block.get("table_html") or ""
        if cells:
            headers, rows = StructureService.table_structure(cells)
        else:
            headers, rows = [], []
        item = {
            "type": "table",
            "headers": headers,
            "rows": rows,
            "html": html,
            "engine": "official_v3",
            "table_source": block.get("table_source", "official"),
        }
        item["capabilities"] = {
            "cell_bbox": bool(block.get("cell_bbox_available")),
        }
        return item
    if label in TITLE_LABELS and content:
        return {"type": "title", "text": content}
    if label in FORMULA_LABELS and content:
        return {"type": "formula", "text": content}
    if label in TEXT_LABELS and content:
        return {"type": "text", "text": content}
    return None


def _build_structured_content(blocks):
    content = []
    for block in blocks:
        item = _structured_item(block)
        if item is not None:
            content.append(item)
    return content


def normalize_official_result(
    official_result,
    page_num=None,
    is_pdf=False,
):
    """把官方 PP-StructureV3 结果转换为 Demo 页面响应。"""
    if not isinstance(official_result, dict):
        raise ValueError("官方 PP-StructureV3 结果必须是对象")
    data = official_result.get("res", official_result)
    if not isinstance(data, dict):
        raise ValueError("官方 PP-StructureV3 结果缺少 res 对象")

    parsing_blocks = data.get("parsing_res_list") or []
    table_results = data.get("table_res_list") or []
    table_index = 0
    blocks = []

    for source_block in parsing_blocks:
        label = str(source_block.get("block_label") or "text")
        block_content = source_block.get("block_content") or ""
        block = {
            "label": label,
            "bbox": _as_bbox(source_block.get("block_bbox")),
            "score": 1.0,
            "order": int(source_block.get("block_order") or 0),
            "official_block_id": source_block.get("block_id"),
            "ocr_text": block_content,
        }

        table_result = None
        if label == "table":
            if table_index < len(table_results):
                table_result = table_results[table_index]
                table_index += 1
            table_html, cells, has_bbox = _table_cells(
                table_result or {
                    "pred_html": block_content,
                    "cell_box_list": [],
                }
            )
            block["table_html"] = table_html
            block["cells"] = cells if cells else None
            block["cell_bbox_available"] = has_bbox
            block["table_model"] = {
                "id": "official-pp-structurev3",
                "name": "Official PP-StructureV3 table recognition",
            }
            block["table_source"] = "official"
            block["ocr_text"] = ""

        blocks.append(block)

    blocks.sort(key=lambda item: item["order"])
    content = _build_structured_content(blocks)
    page_index = data.get("page_index")
    resolved_page = page_num
    if resolved_page is None:
        resolved_page = (
            int(page_index) + 1
            if page_index is not None
            else 1
        )

    layout_result = data.get("layout_det_res") or {}
    ocr_result = data.get("overall_ocr_res") or {}
    return {
        "page": int(resolved_page),
        "width": int(data.get("width") or 0),
        "height": int(data.get("height") or 0),
        "blocks": blocks,
        "is_pdf": bool(is_pdf),
        "text_source": "official_v3",
        "structured": {
            "page": int(resolved_page),
            "content": content,
        },
        "engine": {
            "id": "official_v3",
            "name": "PaddleOCR PP-StructureV3",
            "page_count": data.get("page_count"),
        },
        "official_context": {
            "layout_block_count": len(layout_result.get("boxes") or []),
            "ocr_text_count": len(ocr_result.get("rec_texts") or []),
            "table_count": len(table_results),
            "formula_count": len(data.get("formula_res_list") or []),
        },
        "official_result": data,
    }


def _add_fallback_table_blocks(page_result, fallback_blocks):
    """把回退版面模型找到的表格按阅读位置加入页面。"""
    table_blocks = [
        block
        for block in fallback_blocks or []
        if block.get("label") == "table" and block.get("bbox")
    ]
    if not table_blocks:
        return 0
    blocks = page_result.setdefault("blocks", [])
    for index, source in enumerate(table_blocks):
        bbox = _as_bbox(source.get("bbox"))
        later_orders = [
            block["order"]
            for block in blocks
            if block.get("bbox")
            and block["bbox"][1] > bbox[1]
        ]
        order = min(later_orders) if later_orders else (
            max(
                (block.get("order", 0) for block in blocks),
                default=-1,
            )
            + 1
        )
        for block in blocks:
            if block.get("order", 0) >= order:
                block["order"] += 1
        blocks.append({
            "label": "table",
            "bbox": bbox,
            "score": float(source.get("score", 1.0)),
            "order": order,
            "official_block_id": f"fallback-table-{index}",
            "ocr_text": "",
            "cells": None,
            "table_html": "",
            "cell_bbox_available": False,
            "table_source": "pending",
            "table_model": None,
            "table_fallback_detection": True,
        })
    blocks.sort(key=lambda item: item["order"])
    return len(table_blocks)


def apply_custom_table_results(
    page_result,
    image_path,
    recognizer,
    fallback_table_detector=None,
    allow_full_image_table=False,
):
    """用当前注册表格模型替换官方 table block 内容。"""
    blocks = page_result.get("blocks") or []
    fallback_count = 0
    if not any(block.get("label") == "table" for block in blocks):
        if fallback_table_detector is not None:
            fallback_blocks = fallback_table_detector(image_path) or []
            fallback_count = _add_fallback_table_blocks(
                page_result,
                fallback_blocks,
            )
            blocks = page_result.get("blocks") or []
            if (
                fallback_count == 0
                and allow_full_image_table
            ):
                with Image.open(image_path) as image:
                    width, height = image.size
                fallback_count = _add_fallback_table_blocks(
                    page_result,
                    [{
                        "label": "table",
                        "bbox": [0, 0, width, height],
                        "score": 0.0,
                        "full_image_fallback": True,
                    }],
                )
                blocks = page_result.get("blocks") or []
                for block in blocks:
                    if block.get("label") == "table":
                        block["full_image_fallback"] = True
    success_count = 0
    failures = []
    for block in blocks:
        if block.get("label") != "table":
            continue
        table_bbox = block.get("bbox") or []
        official_table = {
            "html": block.get("table_html") or "",
            "model": block.get("table_model"),
            "cell_bbox_available": bool(
                block.get("cell_bbox_available")
            ),
        }
        try:
            result = recognizer(image_path, table_bbox)
            if not result or (
                not result.get("cells") and not result.get("html")
            ):
                raise RuntimeError("自定义表格模型未返回结果")
            cells = result.get("cells") or []
            has_bbox = any(
                cell.get("bbox")
                and list(cell["bbox"]) != [0, 0, 0, 0]
                for cell in cells
            )
            if not has_bbox:
                for cell in cells:
                    cell.pop("bbox", None)
            block["official_table"] = official_table
            block["cells"] = cells or None
            block["table_html"] = result.get("html") or ""
            block["table_model"] = result.get("model")
            block["table_elapsed_ms"] = result.get("elapsed_ms")
            block["cell_bbox_available"] = has_bbox
            block["table_source"] = "custom"
            success_count += 1
        except Exception as exc:
            block["table_source"] = "official"
            block["table_error"] = str(exc)
            failures.append({
                "block_id": block.get("official_block_id"),
                "bbox": table_bbox,
                "error": str(exc),
            })

    page_result["structured"]["content"] = _build_structured_content(
        blocks
    )
    context = page_result.setdefault("official_context", {})
    context["custom_table_count"] = success_count
    context["custom_table_failures"] = failures
    context["fallback_table_count"] = fallback_count
    context["full_image_fallback_count"] = sum(
        bool(block.get("full_image_fallback"))
        for block in blocks
        if block.get("label") == "table"
    )
    return page_result


class OfficialPPStructureParser:
    """延迟加载并封装官方 PP-StructureV3。"""

    def __init__(
        self,
        logger,
        engine_factory=None,
        pipeline_kwargs=None,
    ):
        self.log = logger
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self.engine_factory = engine_factory or self._create_engine
        self.engine = None
        self.load_error = ""
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _create_engine(self):
        from paddleocr import PPStructureV3

        return PPStructureV3(**self.pipeline_kwargs)

    def ensure_loaded(self):
        if self.engine is not None:
            return self.engine
        with self._load_lock:
            if self.engine is not None:
                return self.engine
            try:
                self.engine = self.engine_factory()
                self.load_error = ""
                self.log.info("官方 PP-StructureV3 已加载")
            except Exception as exc:
                self.load_error = str(exc)
                self.log.error("官方 PP-StructureV3 加载失败: %s", exc)
                raise
        return self.engine

    def parse_page_result(self, image_path, predict_options=None):
        """执行官方整页解析并返回官方原始 JSON。"""
        payload, _ = self.parse_page_context(
            image_path,
            predict_options=predict_options,
        )
        return payload

    def parse_page_context(self, image_path, predict_options=None):
        """执行官方解析并返回 JSON 与预处理后的页面图像。"""
        engine = self.ensure_loaded()
        options = dict(predict_options or {})
        with self._inference_lock:
            results = list(engine.predict(image_path, **options))
        if not results:
            raise RuntimeError("官方 PP-StructureV3 未返回结果")
        result = results[0]
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result["error"]))
        payload = result.json if hasattr(result, "json") else result
        if not isinstance(payload, dict):
            raise ValueError("官方 PP-StructureV3 返回格式无效")
        page_image = None
        if hasattr(result, "get"):
            preprocess_result = result.get("doc_preprocessor_res")
            if isinstance(preprocess_result, dict):
                page_image = preprocess_result.get("output_img")
        return payload, page_image

    def parse_page(
        self,
        image_path,
        page_num=None,
        is_pdf=False,
        predict_options=None,
    ):
        payload = self.parse_page_result(
            image_path,
            predict_options=predict_options,
        )
        return normalize_official_result(
            payload,
            page_num=page_num,
            is_pdf=is_pdf,
        )

    def parse_pages(
        self,
        image_paths,
        is_pdf=False,
        predict_options=None,
    ):
        """按顺序解析多页图片并返回 Demo 页面结果。"""
        return [
            self.parse_page(
                image_path,
                page_num=index,
                is_pdf=is_pdf,
                predict_options=predict_options,
            )
            for index, image_path in enumerate(image_paths, 1)
        ]

    def parse_page_with_image(
        self,
        image_path,
        page_num=None,
        is_pdf=False,
        predict_options=None,
    ):
        """解析单页并同时返回预处理后的页面图像。"""
        payload, page_image = self.parse_page_context(
            image_path,
            predict_options=predict_options,
        )
        result = normalize_official_result(
            payload,
            page_num=page_num,
            is_pdf=is_pdf,
        )
        return result, page_image
