"""Isolated PP-TableMagic V2 GPU worker.

PyTorch and PaddlePaddle ship different CUDA/cuDNN DLL versions on Windows.
Loading both frameworks in one Python process can make either import fail, so
this worker owns the Paddle runtime while the FastAPI process owns PyTorch.
"""

import json
import os
import sys
import types

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")

# Keep the JSON-lines protocol on a dedicated duplicate of the original stdout.
# Native Paddle/PaddleX logs are redirected to stderr so arbitrary byte output
# cannot corrupt the parent process's UTF-8 protocol stream.
_protocol_fd = os.dup(sys.stdout.fileno())
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

# ModelScope is only used by PaddleX for model downloads. Its logger imports
# PyTorch eagerly, which we must avoid in this Paddle-only process.
modelscope_torch_utils = types.ModuleType("modelscope.utils.torch_utils")
modelscope_torch_utils.is_dist = lambda: False
modelscope_torch_utils.is_master = lambda: True
sys.modules["modelscope.utils.torch_utils"] = modelscope_torch_utils

import cv2
import paddle
from paddleocr import TableRecognitionPipelineV2


def _reply(prefix, payload, output):
    output.write(prefix + json.dumps(payload, ensure_ascii=False) + "\n")
    output.flush()


def main():
    output = os.fdopen(_protocol_fd, "w", encoding="utf-8", buffering=1)
    try:
        pipeline = TableRecognitionPipelineV2(
            device="gpu",
            use_layout_detection=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        _reply(
            "READY:",
            {
                "paddle": paddle.__version__,
                "cuda": paddle.version.cuda(),
                "device": paddle.device.get_device(),
            },
            output,
        )
    except Exception as exc:
        _reply("INIT_ERROR:", {"error": str(exc)}, output)
        return 1

    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            image = cv2.imread(request["image_path"])
            if image is None:
                raise ValueError("无法读取表格图片")

            x1, y1, x2, y2 = map(int, request["bbox"])
            pad = 5
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(image.shape[1], x2 + pad)
            cy2 = min(image.shape[0], y2 + pad)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                raise ValueError("表格裁剪区域为空")

            results = list(pipeline.predict(crop))
            data = {}
            if results and hasattr(results[0], "json"):
                data = results[0].json
            inner = data.get("res", data) if isinstance(data, dict) else {}
            table_rows = inner.get("table_res_list", [])

            pred_html = ""
            cell_boxes = []
            ocr_data = {}
            if table_rows:
                pred_html = table_rows[0].get("pred_html", "")
                cell_boxes = table_rows[0].get("cell_box_list", [])
                ocr_data = table_rows[0].get("table_ocr_pred", {})

            if not pred_html and results and hasattr(results[0], "html"):
                html = results[0].html
                if isinstance(html, dict):
                    pred_html = next(iter(html.values()), "")

            _reply(
                "RESULT:",
                {
                    "id": request_id,
                    "result": {
                        "pred_html": pred_html,
                        "cell_box_list": cell_boxes,
                        "ocr_texts": ocr_data.get("rec_texts", []),
                        "ocr_boxes": ocr_data.get("rec_boxes", []),
                        "offset": [cx1, cy1],
                    },
                },
                output,
            )
        except Exception as exc:
            _reply(
                "ERROR:",
                {"id": request_id, "error": str(exc)},
                output,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
