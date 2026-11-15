"""表格数据集基础接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TableSample:
    """模型测试使用的一条表格样本。"""

    sample_id: str
    file_name: str
    row_index: int
    global_index: int
    shard_path: str
    image_bytes: bytes
    width: int
    height: int
    table_bbox: list[float]
    table_box_count: int
    used_full_image_fallback: bool
    image_id: int | None = None
    ocr_word_count: int = 0
    ocr_box_count: int = 0
    ground_truth: dict | None = None

    def locating_info(self):
        """返回不包含图片大对象的样本定位信息。"""
        return {
            "sample_id": self.sample_id,
            "file_name": self.file_name,
            "row_index": self.row_index,
            "global_index": self.global_index,
            "shard_path": self.shard_path,
            "width": self.width,
            "height": self.height,
            "table_bbox": self.table_bbox,
            "table_box_count": self.table_box_count,
            "used_full_image_fallback": self.used_full_image_fallback,
            "image_id": self.image_id,
            "ocr_word_count": self.ocr_word_count,
            "ocr_box_count": self.ocr_box_count,
        }


class TableDataset(ABC):
    """可批量迭代的表格数据集。"""

    @abstractmethod
    def describe(self):
        """返回数据集版本、来源、数量和指纹。"""

    @abstractmethod
    def iter_samples(self, limit=None, start=0, batch_size=64):
        """按稳定顺序迭代样本。"""

    def get_sample(self, sample_id, batch_size=256):
        """按稳定 ID 获取样本；子类可覆盖为更高效实现。"""
        for sample in self.iter_samples(batch_size=batch_size):
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(f"样本不存在: {sample_id}")
