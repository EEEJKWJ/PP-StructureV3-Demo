"""PP-OCRv5 ONNX 区域识别服务。"""

import shutil
import threading
from pathlib import Path

import torch


class OCRService:
    """封装 OCR 模型下载、加载和按版面区域识别。"""

    MODEL_FILES = {
        "ppocr5_server_det.onnx": (
            "marsena/paddleocr-onnx-models",
            "PP-OCRv5_server_det_infer.onnx",
        ),
        "ppocr5_server_rec.onnx": (
            "marsena/paddleocr-onnx-models",
            "PP-OCRv5_server_rec_infer.onnx",
        ),
        "ppocr5_cls.onnx": (
            "SWHL/RapidOCR",
            "PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        ),
        "ppocr5_dict.txt": (
            "AXERA-TECH/PPOCR_v5",
            "ppocrv5_dict.txt",
        ),
    }

    def __init__(self, logger, model_dir="model/ocr"):
        self.log = logger
        self.model_dir = Path(model_dir)
        self.engine = None
        self.model_type = ""
        self._download_lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def download_models(self):
        """下载缺失的 OCR 模型，并缓存到项目模型目录。"""
        from huggingface_hub import hf_hub_download

        with self._download_lock:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            for local_name, (repo_id, remote_name) in self.MODEL_FILES.items():
                local_path = self.model_dir / local_name
                if local_path.exists():
                    continue
                self.log.info(
                    "下载 OCR 模型: %s <- %s/%s",
                    local_name,
                    repo_id,
                    remote_name,
                )
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_name,
                )
                shutil.copy2(downloaded, str(local_path))
                self.log.info("OCR 模型已保存: %s", local_path)

    def get_engine(self):
        """惰性加载 PP-OCRv5，优先使用 server 模型。"""
        if self.engine is not None:
            return self.engine

        with self._engine_lock:
            if self.engine is not None:
                return self.engine

            if not (self.model_dir / "ppocr5_server_det.onnx").exists():
                try:
                    self.download_models()
                except Exception as exc:
                    self.log.error("OCR 模型下载失败: %s", exc)

            class_file = str(self.model_dir / "ppocr5_cls.onnx")
            dictionary_file = str(self.model_dir / "ppocr5_dict.txt")
            use_gpu = torch.cuda.is_available()

            server_detector = self.model_dir / "ppocr5_server_det.onnx"
            mobile_detector = self.model_dir / "ppocr5_mobile_det.onnx"
            if server_detector.exists():
                detector = str(server_detector)
                recognizer = str(self.model_dir / "ppocr5_server_rec.onnx")
                self.model_type = "server"
                self.log.info(
                    "使用 PP-OCRv5 server 版模型 (GPU=%s)",
                    use_gpu,
                )
            elif mobile_detector.exists():
                detector = str(mobile_detector)
                recognizer = str(self.model_dir / "ppocr5_mobile_rec.onnx")
                self.model_type = "mobile"
                self.log.info("使用 PP-OCRv5 mobile 版模型")
            else:
                self.log.warning("OCR 模型未找到，扫描件将无法识别文字")
                return None

            from ppocr5 import simple_ppocr5

            self.engine = simple_ppocr5(
                ppocr5_onnx_det=detector,
                ppocr5_onnx_cls=class_file,
                ppocr5_onnx_rec=recognizer,
                ppcor5_dict=dictionary_file,
                use_gpu=use_gpu,
            )
            self.log.info("OCR 模型加载完成 (GPU=%s)", use_gpu)
            return self.engine

    def recognize_blocks(self, image_path, blocks):
        """按版面区域裁剪 OCR，返回以 block 索引为键的识别行。"""
        import cv2

        engine = self.get_engine()
        if engine is None:
            return {}
        image = cv2.imread(image_path)
        if image is None:
            return {}

        result = {}
        for index, block in enumerate(blocks):
            label = block["label"]
            if label in ("number", "figure", "table"):
                continue

            x1, y1, x2, y2 = block["bbox"]
            padding = 5
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(image.shape[1], x2 + padding)
            crop_y2 = min(image.shape[0], y2 + padding)
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            engine.run(crop)
            if not engine.results:
                continue

            lines = []
            for item in engine.results:
                position = item["rec_pos"]
                bbox = [
                    int(position[0][0]) + crop_x1,
                    int(position[0][1]) + crop_y1,
                    int(position[2][0]) + crop_x1,
                    int(position[2][1]) + crop_y1,
                ]
                lines.append({"text": item["text"], "bbox": bbox})

            lines = self.nms_lines(lines)
            lines = self.merge_suffix_prefix_lines(lines)
            lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
            result[index] = lines
            self.log.info(
                "  区域 #%s(%s) OCR: %s 行",
                index,
                label,
                len(lines),
            )

        return result

    def recognize_block(self, image_path, block):
        """识别单个版面区域，用于文字层乱码回退。"""
        result = self.recognize_blocks(image_path, [block])
        return result.get(0, [])

    def recognize_crop(self, image):
        """识别整张表格裁图，返回裁图局部坐标。"""
        engine = self.get_engine()
        if engine is None or image is None or image.size == 0:
            return []
        with self._inference_lock:
            engine.run(image)
            results = list(engine.results or [])
        if not results:
            return []
        lines = []
        for item in results:
            position = item["rec_pos"]
            lines.append({
                "text": item["text"],
                "bbox": [
                    int(position[0][0]),
                    int(position[0][1]),
                    int(position[2][0]),
                    int(position[2][1]),
                ],
            })
        lines = self.nms_lines(lines)
        lines = self.merge_suffix_prefix_lines(lines)
        lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
        return lines

    def status(self):
        """返回与现有模型状态接口兼容的 OCR 信息。"""
        server_ready = (
            self.model_dir / "ppocr5_server_det.onnx"
        ).exists()
        mobile_ready = (
            self.model_dir / "ppocr5_mobile_det.onnx"
        ).exists()
        model_type = self.model_type
        if not model_type:
            if server_ready:
                model_type = "server"
            elif mobile_ready:
                model_type = "mobile"
        return {
            "available": server_ready or mobile_ready,
            "model": model_type,
        }

    @classmethod
    def nms_lines(
        cls,
        lines,
        iou_threshold=0.5,
        contain_threshold=0.7,
    ):
        """删除重叠或近乎包含的 OCR 行，优先保留较长文字。"""
        if len(lines) <= 1:
            return lines

        sorted_lines = sorted(
            lines,
            key=lambda line: -len(line["text"]),
        )
        kept = []
        for line in sorted_lines:
            should_drop = any(
                cls.iou(line["bbox"], previous["bbox"]) > iou_threshold
                or cls.contain_ratio(
                    line["bbox"],
                    previous["bbox"],
                ) > contain_threshold
                for previous in kept
            )
            if not should_drop:
                kept.append(line)
        return kept

    @staticmethod
    def contain_ratio(first, second):
        """计算 first 与 second 的交集占 first 面积的比例。"""
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        if first_area == 0:
            return 0
        return intersection / first_area

    @staticmethod
    def iou(first, second):
        """计算两个 OCR 框的交并比。"""
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / (first_area + second_area - intersection)

    @staticmethod
    def merge_suffix_prefix_lines(lines, max_overlap=6):
        """合并同一行相邻 OCR 框中重复的前后缀文字。"""
        if len(lines) <= 1:
            return lines

        sorted_lines = sorted(
            lines,
            key=lambda line: (line["bbox"][1], line["bbox"][0]),
        )
        merged = []
        for line in sorted_lines:
            if not merged:
                merged.append(dict(line))
                continue

            previous = merged[-1]
            previous_center = (
                previous["bbox"][1] + previous["bbox"][3]
            ) / 2
            current_center = (
                line["bbox"][1] + line["bbox"][3]
            ) / 2
            same_row = abs(previous_center - current_center) < 15
            if same_row:
                horizontal_overlap = (
                    min(previous["bbox"][2], line["bbox"][2])
                    - max(previous["bbox"][0], line["bbox"][0])
                )
                if horizontal_overlap > 0:
                    overlap_length = 0
                    previous_text = previous["text"]
                    current_text = line["text"]
                    max_length = min(
                        len(previous_text),
                        len(current_text),
                        max_overlap,
                    )
                    for length in range(max_length, 0, -1):
                        if previous_text.endswith(current_text[:length]):
                            overlap_length = length
                            break
                    if overlap_length > 0:
                        previous["text"] = (
                            previous_text + current_text[overlap_length:]
                        )
                        previous["bbox"] = [
                            min(previous["bbox"][0], line["bbox"][0]),
                            min(previous["bbox"][1], line["bbox"][1]),
                            max(previous["bbox"][2], line["bbox"][2]),
                            max(previous["bbox"][3], line["bbox"][3]),
                        ]
                        continue
            merged.append(dict(line))
        return merged
