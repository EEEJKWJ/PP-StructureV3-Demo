"""FinTabNet.c-Structure XML 数据集适配器。"""

import json
import hashlib
from pathlib import Path

from app.datasets.base import TableDataset, TableSample
from app.datasets.preprocessing import prepare_split


DEFAULT_ROOT = r"D:\dataset\FinTabNet.c\FinTabNet.c-Structure"
DEFAULT_CACHE_ROOT = Path("data/processed/fintabnet-c-structure")


class FinTabNetCStructureDataset(TableDataset):
    """读取预处理后的 FinTabNet.c 结构划分。"""

    def __init__(
        self,
        root=DEFAULT_ROOT,
        split="test",
        cache_path=None,
        prepare_if_missing=True,
    ):
        self.root = Path(root).resolve()
        self.split = split
        self.cache_path = Path(
            cache_path
            or DEFAULT_CACHE_ROOT / f"{split}.jsonl"
        ).resolve()
        self.meta_path = self.cache_path.with_suffix(
            self.cache_path.suffix + ".meta.json"
        )
        if not self.cache_path.exists() and prepare_if_missing:
            prepare_split(
                self.root,
                self.split,
                self.cache_path,
            )
        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"结构预处理缓存不存在: {self.cache_path}"
            )
        self.metadata = (
            json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.meta_path.exists()
            else {}
        )
        self.total_samples = self.metadata.get("written")
        if self.total_samples is None:
            with self.cache_path.open(encoding="utf-8") as handle:
                self.total_samples = sum(1 for _ in handle)

    def describe(self):
        stat = self.cache_path.stat()
        fingerprint = hashlib.sha256(
            (
                f"{self.root}|{self.split}|"
                f"{stat.st_size}|{stat.st_mtime_ns}"
            ).encode("utf-8")
        ).hexdigest()
        return {
            "name": "FinTabNet.c Structure",
            "version": "c-structure",
            "type": "fintabnet-c-structure",
            "format": "image+xml+words-jsonl",
            "input_path": str(self.root),
            "split": self.split,
            "total_samples": self.total_samples,
            "cache_path": str(self.cache_path),
            "failures": len(self.metadata.get("failures", [])),
            "fingerprint": fingerprint,
            "fingerprint_method": "cache+path+size+mtime",
        }

    def iter_samples(self, limit=None, start=0, batch_size=64):
        if start < 0 or (limit is not None and limit < 0):
            raise ValueError("start/limit 不能小于 0")
        selected = 0
        with self.cache_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < start:
                    continue
                record = json.loads(line)
                image_path = self.root / record["image_path"]
                image_bytes = image_path.read_bytes()
                yield TableSample(
                    sample_id=record["sample_id"],
                    file_name=image_path.name,
                    row_index=index,
                    global_index=index,
                    shard_path=str(self.root / record["xml_path"]),
                    image_bytes=image_bytes,
                    width=record["width"],
                    height=record["height"],
                    table_bbox=record["table_bbox"],
                    table_box_count=1,
                    used_full_image_fallback=False,
                    ocr_word_count=record["word_count"],
                    ground_truth={
                        "status": "available",
                        "reason": "",
                        "cells": record["cells"],
                        "html": record["html"],
                    },
                )
                selected += 1
                if limit is not None and selected >= limit:
                    return

    def get_sample(self, sample_id, batch_size=256):
        with self.cache_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                record = json.loads(line)
                if record["sample_id"] != sample_id:
                    continue
                image_path = self.root / record["image_path"]
                return TableSample(
                    sample_id=record["sample_id"],
                    file_name=image_path.name,
                    row_index=index,
                    global_index=index,
                    shard_path=str(self.root / record["xml_path"]),
                    image_bytes=image_path.read_bytes(),
                    width=record["width"],
                    height=record["height"],
                    table_bbox=record["table_bbox"],
                    table_box_count=1,
                    used_full_image_fallback=False,
                    ocr_word_count=record["word_count"],
                    ground_truth={
                        "status": "available",
                        "reason": "",
                        "cells": record["cells"],
                        "html": record["html"],
                    },
                )
        raise KeyError(f"样本不存在: {sample_id}")
