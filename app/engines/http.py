"""HTTP 远程表格模型适配器。"""

import base64
import json
import urllib.request

import cv2

from app.engines.base import (
    TableAdapter,
    TableOutput,
    cells_to_html,
    crop_table,
    parse_html_table,
)


class HttpTableAdapter(TableAdapter):
    """使用 JSON + Base64 图片调用标准远程表格接口。"""

    backend = "http"

    def load(self):
        options = self.config.get("options", {})
        endpoint = options.get("endpoint") or self.config.get("source")
        if not endpoint or not str(endpoint).startswith(("http://", "https://")):
            self.load_error = "HTTP 适配器需要有效的 endpoint"
            raise ValueError(self.load_error)
        self.loaded = True
        self.load_error = ""
        return endpoint

    def predict(self, image_path, table_bbox, options=None):
        self.ensure_loaded()
        adapter_options = self.config.get("options", {})
        crop, offset = crop_table(
            image_path,
            table_bbox,
            padding=adapter_options.get("crop_padding", 5),
        )
        if crop is None:
            return None

        success, encoded = cv2.imencode(
            ".jpg",
            crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not success:
            self.log.error("HTTP 表格识别失败: 无法编码裁剪图")
            return None

        endpoint = adapter_options.get("endpoint") or self.config["source"]
        payload = {
            "model_id": self.model_id,
            "image_base64": base64.b64encode(encoded).decode("ascii"),
            "image_format": "jpeg",
            "bbox": list(table_bbox),
            "options": options or {},
        }

        try:
            response = self._post_json(endpoint, payload)
            inner = (
                response.get("result", response)
                if isinstance(response, dict)
                else {}
            )
            table_html = (
                inner.get("html")
                or inner.get("pred_html")
                or inner.get("table_html")
                or ""
            )
            cells = inner.get("cells")
            cells_relative = adapter_options.get(
                "cells_relative",
                True,
            )
            if cells:
                cells = [dict(cell) for cell in cells]
                for cell in cells:
                    cell.setdefault("text", "")
                    cell.setdefault("bbox", [0, 0, 0, 0])
                    if cells_relative:
                        cell["bbox"] = [
                            cell["bbox"][0] + offset[0],
                            cell["bbox"][1] + offset[1],
                            cell["bbox"][2] + offset[0],
                            cell["bbox"][3] + offset[1],
                        ]
            elif table_html:
                cells = parse_html_table(
                    table_html,
                    None,
                    offset[0],
                    offset[1],
                )
            else:
                return None

            if not table_html:
                table_html = cells_to_html(cells)
            raw = response if adapter_options.get("return_raw") else None
            return TableOutput(
                cells=cells,
                html=table_html,
                raw=raw,
            )
        except Exception as exc:
            self.log.error("HTTP 表格识别失败: %s", exc)
            return None

    def _post_json(self, endpoint, payload):
        options = self.config.get("options", {})
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **options.get("headers", {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=options.get("timeout_seconds", 60),
        ) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
