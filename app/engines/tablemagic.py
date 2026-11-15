"""PP-TableMagic V2 模型适配器。"""

import os

from app.engines.base import (
    TableAdapter,
    TableOutput,
    crop_table,
    parse_html_table,
    restore_cell_newlines,
)


class TableMagicAdapter(TableAdapter):
    """包装 PaddleOCR TableRecognitionPipelineV2。"""

    backend = "paddle"

    def load(self):
        if self.loaded:
            return self.engine

        os.environ.setdefault(
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
            "True",
        )
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
        try:
            from paddleocr import TableRecognitionPipelineV2

            options = self.config.get("options", {})
            pipeline_kwargs = {
                "use_layout_detection": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            }
            pipeline_kwargs.update(options.get("pipeline_kwargs", {}))
            pipeline_kwargs.setdefault(
                "device",
                self.config.get("device", "gpu"),
            )
            self.engine = TableRecognitionPipelineV2(**pipeline_kwargs)
            self.loaded = True
            self.load_error = ""
            self.log.info(
                "表格识别引擎: %s (%s)",
                self.config["name"],
                self.config.get("device", "unknown"),
            )
            return self.engine
        except Exception as exc:
            self.load_error = str(exc)
            self.log.warning(
                "PP-TableMagic V2 初始化失败: %s",
                exc,
            )
            raise

    def predict(self, image_path, table_bbox, options=None):
        engine = self.ensure_loaded()
        adapter_options = self.config.get("options", {})
        crop, offset = crop_table(
            image_path,
            table_bbox,
            padding=adapter_options.get("crop_padding", 5),
        )
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

            table_html = ""
            cell_boxes = []
            ocr_data = {}
            if table_rows:
                table_html = table_rows[0].get("pred_html", "")
                cell_boxes = table_rows[0].get("cell_box_list", [])
                ocr_data = table_rows[0].get("table_ocr_pred", {})

            if not table_html and hasattr(result, "html"):
                if isinstance(result.html, dict):
                    table_html = next(iter(result.html.values()), "")
            if not table_html:
                self.log.info("PP-TableMagic V2: 未检测到表格")
                return None

            cell_quads = None
            if cell_boxes:
                quads = []
                for box in cell_boxes:
                    x1, y1, x2, y2 = box[:4]
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
                cell_quads = [quads]

            cells = parse_html_table(
                table_html,
                cell_quads,
                offset[0],
                offset[1],
            )
            restore_cell_newlines(cells, cell_boxes, ocr_data)
            if not cells:
                return None

            self.log.info(
                "PP-TableMagic V2 表格识别完成: %s 个单元格",
                len(cells),
            )
            raw = None
            if adapter_options.get("return_raw"):
                raw = {
                    "cell_boxes": cell_boxes,
                    "ocr_text_count": len(ocr_data.get("rec_texts", [])),
                }
            return TableOutput(cells=cells, html=table_html, raw=raw)
        except Exception as exc:
            self.log.error("PP-TableMagic V2 表格识别失败: %s", exc)
            return None
