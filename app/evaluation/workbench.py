"""单样本数据集浏览和重新识别服务。"""

import tempfile
from pathlib import Path


class SampleWorkbenchService:
    """提供不含图片大对象的样本元数据和单样本运行。"""

    def __init__(self, dataset_registry, logger):
        self.dataset_registry = dataset_registry
        self.log = logger

    def list_datasets(self):
        return self.dataset_registry.list_datasets()

    def public_sample(self, sample):
        ground_truth = sample.ground_truth or {
            "status": "not_available",
            "reason": "数据集没有提供结构真值",
        }
        ground_truth = dict(ground_truth)
        ocr_words = ground_truth.pop("ocr_words", None)
        ocr_boxes = ground_truth.pop("ocr_boxes", None)
        if ocr_words is not None:
            ground_truth["ocr_word_count"] = len(ocr_words)
        if ocr_boxes is not None:
            ground_truth["ocr_box_count"] = len(ocr_boxes)
        return {
            **sample.locating_info(),
            "ground_truth": ground_truth,
        }

    def get_sample(self, dataset_id, sample_id):
        dataset = self.dataset_registry.get_dataset(dataset_id)
        return dataset.get_sample(sample_id)

    def list_samples(self, dataset_id, offset=0, limit=20):
        if offset < 0:
            raise ValueError("offset 不能小于 0")
        if limit < 1 or limit > 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        dataset = self.dataset_registry.get_dataset(dataset_id)
        samples = [
            self.public_sample(sample)
            for sample in dataset.iter_samples(
                start=offset,
                limit=limit,
            )
        ]
        return {
            "ok": True,
            "dataset_id": dataset_id,
            "dataset": dataset.describe(),
            "offset": offset,
            "limit": limit,
            "total_samples": dataset.total_samples,
            "has_more": offset + len(samples) < dataset.total_samples,
            "samples": samples,
        }

    def get_image(self, dataset_id, sample_id):
        sample = self.get_sample(dataset_id, sample_id)
        if not sample.image_bytes:
            raise ValueError("样本图片为空")
        return sample.image_bytes

    def run_sample(self, dataset_id, sample_id, recognizer):
        sample = self.get_sample(dataset_id, sample_id)
        public_sample = self.public_sample(sample)
        if not sample.image_bytes:
            return {
                "ok": False,
                "status": "failed",
                "sample": public_sample,
                "prediction": None,
                "ground_truth": public_sample["ground_truth"],
                "error": "样本图片为空",
            }

        try:
            with tempfile.TemporaryDirectory(
                dir=Path.cwd(),
                prefix=".sample_workbench_",
            ) as temp_dir:
                image_path = Path(temp_dir) / "sample.jpg"
                image_path.write_bytes(sample.image_bytes)
                prediction = recognizer(
                    str(image_path),
                    sample.table_bbox,
                )
        except Exception as exc:
            self.log.error(
                "单样本 %s 识别失败: %s",
                sample_id,
                exc,
            )
            return {
                "ok": False,
                "status": "failed",
                "sample": public_sample,
                "prediction": None,
                "ground_truth": public_sample["ground_truth"],
                "error": str(exc),
            }

        if prediction is None:
            return {
                "ok": False,
                "status": "no_table",
                "sample": public_sample,
                "prediction": None,
                "ground_truth": public_sample["ground_truth"],
                "error": "模型未返回表格结果",
            }

        cells = prediction.get("cells") or []
        has_html = bool(prediction.get("html"))
        status = "success" if cells else "no_table"
        return {
            "ok": status == "success",
            "status": status,
            "sample": public_sample,
            "prediction": prediction,
            "ground_truth": public_sample["ground_truth"],
            "error": "" if cells else "未检出表格单元格",
            "html_present": has_html,
        }
