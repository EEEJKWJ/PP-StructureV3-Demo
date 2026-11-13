"""表格识别服务。

集中处理表格引擎初始化、PP-TableMagic/RapidTable 识别和 HTML 生成。
"""

import html as html_lib
import os
import threading

import cv2
import numpy as np
import torch  # 必须先于 Paddle 导入，避免 Windows CUDA DLL 冲突


class TableService:
    """封装当前表格识别引擎及其输出转换。"""

    def __init__(self, logger, ocr_provider=None):
        self.log = logger
        self.ocr_provider = ocr_provider
        self.engine = None
        self.engine_type = ""
        self._lock = threading.Lock()

    def get_engine(self):
        """惰性加载表格引擎，失败时降级到 RapidTable。"""
        with self._lock:
            if self.engine is not None:
                return self.engine

            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
            try:
                from paddleocr import TableRecognitionPipelineV2

                self.engine = TableRecognitionPipelineV2(
                    device="gpu",
                    use_layout_detection=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
                self.engine_type = "paddle"
                self.log.info(
                    "表格识别引擎: PP-TableMagic V2 (gpu) — "
                    "表格分类+单元格检测+结构识别+OCR"
                )
                return self.engine
            except Exception as exc:
                self.log.warning(
                    "PP-TableMagic V2 初始化失败: %s，尝试 RapidTable",
                    exc,
                )

            try:
                from rapid_table import ModelType, RapidTable, RapidTableInput

                args = RapidTableInput(model_type=ModelType.SLANETPLUS)
                self.engine = RapidTable(args)
                self.engine_type = "rapid"
                self.log.info("表格识别引擎: RapidTable (SLANet_plus ONNX)")
            except Exception as exc:
                self.log.error("表格识别引擎加载失败: %s", exc)

            return self.engine

    def recognize(self, image_path, table_bbox):
        """识别一个表格区域，返回单元格和原始/生成 HTML。"""
        engine = self.get_engine()
        if engine is None:
            return None
        if self.engine_type == "paddle":
            return self._recognize_paddle(engine, image_path, table_bbox)
        return self._recognize_rapid(engine, image_path, table_bbox)

    def _crop(self, image_path, table_bbox):
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        x1, y1, x2, y2 = map(int, table_bbox)
        pad = 5
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(image.shape[1], x2 + pad)
        cy2 = min(image.shape[0], y2 + pad)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, None
        return crop, (cx1, cy1)

    def _recognize_paddle(self, engine, image_path, table_bbox):
        crop, offset = self._crop(image_path, table_bbox)
        if crop is None:
            return None

        try:
            results = list(engine.predict(crop))
            if not results:
                return None

            result = results[0]
            data = result.json if hasattr(result, "json") else {}
            inner = data.get("res", data) if isinstance(data, dict) else {}
            table_rows = inner.get("table_res_list", [])

            pred_html = ""
            cell_boxes = []
            ocr_data = {}
            if table_rows:
                pred_html = table_rows[0].get("pred_html", "")
                cell_boxes = table_rows[0].get("cell_box_list", [])
                ocr_data = table_rows[0].get("table_ocr_pred", {})

            if not pred_html and hasattr(result, "html"):
                if isinstance(result.html, dict):
                    pred_html = next(iter(result.html.values()), "")
            if not pred_html:
                self.log.info("PP-TableMagic V2: 未检测到表格")
                return None

            cell_quads = None
            if cell_boxes:
                quads = []
                for box in cell_boxes:
                    x1, y1, x2, y2 = box[:4]
                    quads.append([
                        x1, y1, x2, y1,
                        x2, y2, x1, y2,
                    ])
                cell_quads = [quads]

            cells = self.parse_html_table(
                pred_html,
                cell_quads,
                offset[0],
                offset[1],
            )
            self._restore_cell_newlines(cells, cell_boxes, ocr_data)
            if not cells:
                return None

            self.log.info(
                "PP-TableMagic V2 表格识别完成: %s 个单元格",
                len(cells),
            )
            return {"cells": cells, "html": pred_html}
        except Exception as exc:
            self.log.error("PP-TableMagic V2 表格识别失败: %s", exc)
            return None

    def _recognize_rapid(self, engine, image_path, table_bbox):
        crop, offset = self._crop(image_path, table_bbox)
        if crop is None:
            return None

        ocr_results = None
        if self.ocr_provider is not None:
            ocr = self.ocr_provider()
            if ocr is not None:
                ocr.run(crop)
                if ocr.results:
                    boxes = np.array([item["rec_pos"] for item in ocr.results])
                    texts = tuple(item["text"] for item in ocr.results)
                    scores = tuple(0.9 for _ in ocr.results)
                    ocr_results = [(boxes, texts, scores)]

        try:
            result = engine(crop, ocr_results=ocr_results)
            table_html = result.pred_htmls[0] if result.pred_htmls else ""
            if not table_html:
                return None
            cells = self.parse_html_table(
                table_html,
                result.cell_bboxes,
                offset[0],
                offset[1],
            )
            self.log.info("RapidTable 表格识别完成: %s 个单元格", len(cells))
            return {"cells": cells, "html": table_html}
        except Exception as exc:
            self.log.error("RapidTable 表格识别失败: %s", exc)
            return None

    @staticmethod
    def _restore_cell_newlines(cells, cell_boxes, ocr_data):
        texts = ocr_data.get("rec_texts", []) if isinstance(ocr_data, dict) else []
        boxes = ocr_data.get("rec_boxes", []) if isinstance(ocr_data, dict) else []
        if not cell_boxes or not texts or not boxes:
            return

        text_by_cell = {}
        for index, text in enumerate(texts):
            if index >= len(boxes):
                break
            box = boxes[index]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            best_cell = -1
            for cell_index, cell_box in enumerate(cell_boxes):
                if (
                    cell_box[0] <= center_x <= cell_box[2]
                    and cell_box[1] <= center_y <= cell_box[3]
                ):
                    best_cell = cell_index
                    break

            if best_cell == -1:
                min_distance = float("inf")
                for cell_index, cell_box in enumerate(cell_boxes):
                    cell_x = (cell_box[0] + cell_box[2]) / 2
                    cell_y = (cell_box[1] + cell_box[3]) / 2
                    distance = (center_x - cell_x) ** 2 + (center_y - cell_y) ** 2
                    if distance < min_distance:
                        min_distance = distance
                        best_cell = cell_index

            if best_cell >= 0:
                text_by_cell.setdefault(best_cell, []).append((center_y, text))

        for cell_index, lines in text_by_cell.items():
            if cell_index < len(cells):
                lines.sort(key=lambda item: item[0])
                cells[cell_index]["text"] = "\n".join(text for _, text in lines)

    @staticmethod
    def parse_html_table(table_html, cell_boxes_batch, offset_x, offset_y):
        """把表格 HTML 转换为带行列和坐标的单元格列表。"""
        from html.parser import HTMLParser

        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.cells = []
                self.row = -1
                self.col = 0
                self.in_cell = False
                self.current_text = ""
                self.occupied = {}
                self.colspan = 1
                self.rowspan = 1

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.row += 1
                    self.col = 0
                    while (self.row, self.col) in self.occupied:
                        self.col += 1
                elif tag == "td":
                    self.in_cell = True
                    self.current_text = ""
                    while (self.row, self.col) in self.occupied:
                        self.col += 1
                    attrs = dict(attrs)
                    self.colspan = int(attrs.get("colspan", 1))
                    self.rowspan = int(attrs.get("rowspan", 1))

            def handle_data(self, data):
                if self.in_cell:
                    self.current_text += data

            def handle_endtag(self, tag):
                if tag != "td" or not self.in_cell:
                    return
                self.in_cell = False
                self.cells.append({
                    "row": self.row,
                    "col": self.col,
                    "text": self.current_text.strip(),
                })
                for row_delta in range(self.rowspan):
                    for col_delta in range(self.colspan):
                        if row_delta or col_delta:
                            self.occupied[
                                (self.row + row_delta, self.col + col_delta)
                            ] = True
                self.col += self.colspan

        parser = TableParser()
        parser.feed(table_html)

        boxes = []
        if cell_boxes_batch:
            for box in cell_boxes_batch[0]:
                xs = [float(box[index]) for index in range(0, len(box), 2)]
                ys = [float(box[index]) for index in range(1, len(box), 2)]
                boxes.append([
                    int(min(xs)) + offset_x,
                    int(min(ys)) + offset_y,
                    int(max(xs)) + offset_x,
                    int(max(ys)) + offset_y,
                ])

        for index, cell in enumerate(parser.cells):
            cell["bbox"] = boxes[index] if index < len(boxes) else [0, 0, 0, 0]
        return parser.cells

    @staticmethod
    def cells_to_html(cells):
        """从单元格坐标生成基础无合并单元格的 HTML 表格。"""
        if not cells:
            return ""
        max_row = max(cell["row"] for cell in cells)
        max_col = max(cell["col"] for cell in cells)
        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            grid[cell["row"]][cell["col"]] = cell.get("text", "")

        rows = []
        for row in grid:
            columns = []
            for value in row:
                text = html_lib.escape(str(value)).replace("\n", "<br>")
                columns.append(f"<td>{text}</td>")
            rows.append("<tr>" + "".join(columns) + "</tr>")
        return "<table><tbody>" + "".join(rows) + "</tbody></table>"
