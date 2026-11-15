"""表格测试数据集注册。"""

import os
from pathlib import Path

from app.datasets.fintabnet import FinTabNetDataset


DEFAULT_FINTABNET_PATH = (
    r"D:\dataset\fintabnet-pubtables-full\test"
)


class DatasetRegistry:
    """按稳定 ID 管理可用数据集。"""

    def __init__(self, fintabnet_path=None):
        self.datasets = {
            "fintabnet-test": {
                "id": "fintabnet-test",
                "name": "FinTabNet Test",
                "path": Path(
                    fintabnet_path
                    or os.environ.get(
                        "PP_FINTABNET_PATH",
                        DEFAULT_FINTABNET_PATH,
                    )
                ),
                "type": "fintabnet",
            },
        }
        self._cache = {}

    def list_datasets(self):
        result = []
        for dataset in self.datasets.values():
            path = dataset["path"]
            available = path.exists()
            item = {
                "id": dataset["id"],
                "name": dataset["name"],
                "type": dataset["type"],
                "path": str(path),
                "available": available,
                "error": "",
                "total_samples": None,
            }
            if available:
                try:
                    item["total_samples"] = self.get_dataset(
                        dataset["id"]
                    ).total_samples
                except Exception as exc:
                    item["available"] = False
                    item["error"] = str(exc)
            else:
                item["error"] = (
                    "数据集路径不存在；可设置 PP_FINTABNET_PATH"
                )
            result.append(item)
        return result

    def get_dataset(self, dataset_id):
        if dataset_id not in self.datasets:
            raise KeyError(f"数据集不存在: {dataset_id}")
        dataset = self.datasets[dataset_id]
        path = dataset["path"].resolve()
        cache_key = (dataset_id, str(path))
        if cache_key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(
                    f"数据集路径不存在: {path}"
                )
            self._cache[cache_key] = FinTabNetDataset(path)
        return self._cache[cache_key]
