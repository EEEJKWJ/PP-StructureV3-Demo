"""PP-DocLayoutV3 版面检测服务。"""

import threading
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor


class LayoutService:
    """封装版面模型加载、设备管理和检测逻辑。"""

    def __init__(
        self,
        logger,
        model_name="PaddlePaddle/PP-DocLayoutV3_safetensors",
        model_dir="model",
        threshold=0.35,
    ):
        self.log = logger
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.processor = None
        self.model = None
        self.status = "loading"
        self.error = ""
        self._load_lock = threading.Lock()

    def load(self):
        """加载本地模型；首次运行时下载并缓存到指定目录。"""
        with self._load_lock:
            if self.model is not None:
                return

            try:
                started_at = time.time()
                local_config = self.model_dir / "config.json"
                if local_config.exists():
                    source = str(self.model_dir)
                    self.log.info("从本地加载版面模型: %s", source)
                else:
                    source = self.model_name
                    self.log.info(
                        "首次启动，从 HuggingFace 下载 %s ...",
                        source,
                    )

                self.processor = RTDetrImageProcessor.from_pretrained(source)
                self.model = AutoModelForObjectDetection.from_pretrained(source)

                if not local_config.exists():
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    self.processor.save_pretrained(str(self.model_dir))
                    self.model.save_pretrained(str(self.model_dir))
                    self.log.info("版面模型已保存到 %s/", self.model_dir)

                if torch.cuda.is_available():
                    self.model.to("cuda")
                self.model.eval()
                self.status = "ready"
                self.error = ""
                self.log.info(
                    "版面模型加载完成: %.2fs",
                    time.time() - started_at,
                )
            except Exception as exc:
                self.status = "error"
                self.error = str(exc)
                self.log.error("版面模型加载失败: %s", exc)

    def ensure_ready(self):
        """在模型尚未加载时给出与旧接口一致的错误。"""
        if self.model is None:
            raise RuntimeError("模型尚未加载完成，请稍候")

    def detect(self, image_path):
        """检测图片版面，返回带阅读顺序的 block 列表。"""
        self.ensure_ready()
        image = Image.open(image_path).convert("RGB")
        device = next(self.model.parameters()).device

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )[0]

        labels = self.model.config.id2label
        blocks = []
        for index, (score, label_id, box) in enumerate(
            zip(results["scores"], results["labels"], results["boxes"])
        ):
            blocks.append({
                "bbox": [round(coordinate) for coordinate in box.tolist()],
                "label": labels[label_id.item()],
                "score": round(score.item(), 3),
                "order": index,
            })

        blocks = [block for block in blocks if block["label"] != "number"]
        blocks.sort(key=lambda block: -block["score"])

        kept = []
        for block in blocks:
            if any(
                self.iou(block["bbox"], previous["bbox"]) > 0.5
                for previous in kept
            ):
                continue
            kept.append(block)

        blocks = self.sort_reading_order(kept)
        for index, block in enumerate(blocks):
            block["order"] = index

        self.log.info("检测到 %s 个区域", len(blocks))
        for block in blocks:
            self.log.info(
                "  [%s] %-12s score=%.3f bbox=%s",
                block["order"],
                block["label"],
                block["score"],
                block["bbox"],
            )
        return blocks

    @staticmethod
    def sort_reading_order(blocks, y_tolerance=30):
        """按阅读顺序排序，并处理双栏页面的轻微纵向错位。"""
        if not blocks:
            return blocks

        sorted_by_y = sorted(blocks, key=lambda block: block["bbox"][1])
        rows = []
        current_row = [sorted_by_y[0]]
        row_y = sorted_by_y[0]["bbox"][1]

        for block in sorted_by_y[1:]:
            if block["bbox"][1] - row_y <= y_tolerance:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
                row_y = block["bbox"][1]
        rows.append(current_row)

        result = []
        for row in rows:
            row.sort(key=lambda block: block["bbox"][0])
            result.extend(row)
        return result

    @staticmethod
    def iou(first, second):
        """计算两个矩形框的交并比。"""
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
    def gpu_info():
        """返回当前进程可见的 CUDA 设备信息。"""
        available = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if available else ""
        return {"available": available, "name": name}
