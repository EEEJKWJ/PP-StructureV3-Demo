"""官方 PP-StructureV3 整页解析与 Demo 结果转换。"""

import threading

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


def _structured_item(block, table):
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
        }
        if table is not None:
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
    content = []

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
            block["ocr_text"] = ""

        item = _structured_item(block, table_result)
        if item is not None:
            content.append(item)
        blocks.append(block)

    blocks.sort(key=lambda item: item["order"])
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


class OfficialPPStructureParser:
    """延迟加载并封装官方 PP-StructureV3。"""

    def __init__(self, logger, engine_factory=None):
        self.log = logger
        self.engine_factory = engine_factory or self._create_engine
        self.engine = None
        self.load_error = ""
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @staticmethod
    def _create_engine():
        from paddleocr import PPStructureV3

        return PPStructureV3()

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
        return payload

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

    def parse_pages(self, image_paths, is_pdf=False):
        """按顺序解析多页图片并返回 Demo 页面结果。"""
        return [
            self.parse_page(
                image_path,
                page_num=index,
                is_pdf=is_pdf,
            )
            for index, image_path in enumerate(image_paths, 1)
        ]
