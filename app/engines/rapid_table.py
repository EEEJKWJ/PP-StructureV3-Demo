"""RapidTable 模型适配器。"""

import numpy as np

from app.engines.base import (
    TableAdapter,
    TableOutput,
    crop_table,
    parse_html_table,
)


class RapidTableAdapter(TableAdapter):
    """包装 RapidTable/SLANet_plus ONNX 产线。"""

    backend = "rapid"

    def load(self):
        if self.loaded:
            return self.engine

        try:
            from rapid_table import ModelType, RapidTable, RapidTableInput

            options = self.config.get("options", {})
            model_type_name = str(
                options.get("model_type", "SLANETPLUS")
            ).upper()
            try:
                model_type = getattr(ModelType, model_type_name)
            except AttributeError as exc:
                raise ValueError(
                    f"RapidTable 模型类型无效: {model_type_name}"
                ) from exc

            input_kwargs = {"model_type": model_type}
            for key in (
                "model_dir_or_path",
                "engine_type",
                "engine_cfg",
                "use_ocr",
                "ocr_params",
            ):
                if key in options and options[key] is not None:
                    input_kwargs[key] = options[key]

            self.engine = RapidTable(RapidTableInput(**input_kwargs))
            self.loaded = True
            self.load_error = ""
            self.log.info(
                "表格识别引擎: RapidTable (%s)",
                options.get("model_type", "SLANETPLUS"),
            )
            return self.engine
        except Exception as exc:
            self.load_error = str(exc)
            self.log.warning("RapidTable 初始化失败: %s", exc)
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

        ocr_results = None
        if (
            adapter_options.get("use_external_ocr", True)
            and self.ocr_provider is not None
        ):
            ocr = self.ocr_provider()
            if ocr is not None:
                ocr.run(crop)
                if ocr.results:
                    boxes = np.array(
                        [item["rec_pos"] for item in ocr.results]
                    )
                    texts = tuple(
                        item["text"] for item in ocr.results
                    )
                    scores = tuple(0.9 for _ in ocr.results)
                    ocr_results = [(boxes, texts, scores)]

        try:
            result = engine(crop, ocr_results=ocr_results)
            table_html = (
                result.pred_htmls[0] if result.pred_htmls else ""
            )
            if not table_html:
                return None
            cells = parse_html_table(
                table_html,
                result.cell_bboxes,
                offset[0],
                offset[1],
            )
            self.log.info(
                "RapidTable 表格识别完成: %s 个单元格",
                len(cells),
            )
            raw = None
            if adapter_options.get("return_raw"):
                raw = {"cell_bbox_count": len(result.cell_bboxes or [])}
            return TableOutput(cells=cells, html=table_html, raw=raw)
        except Exception as exc:
            self.log.error("RapidTable 表格识别失败: %s", exc)
            return None
