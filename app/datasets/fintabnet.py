"""FinTabNet Parquet 数据集适配器。"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from app.datasets.base import TableDataset, TableSample


class FinTabNetDataset(TableDataset):
    """读取一个 Parquet 分片或包含多个分片的目录。"""

    def __init__(self, parquet_path):
        self.input_path = Path(parquet_path).resolve()
        self.shards = self._discover_shards(self.input_path)
        if not self.shards:
            raise FileNotFoundError(
                f"未找到 Parquet 分片: {self.input_path}"
            )
        self.shard_rows = {}
        for shard in self.shards:
            parquet = pq.ParquetFile(shard)
            try:
                self.shard_rows[shard] = parquet.metadata.num_rows
            finally:
                parquet.close()
        self.total_samples = sum(self.shard_rows.values())
        self._description = self._build_description()

    @staticmethod
    def _discover_shards(input_path):
        if input_path.is_file():
            if input_path.suffix.lower() != ".parquet":
                raise ValueError(f"不是 Parquet 文件: {input_path}")
            return [input_path]
        if input_path.is_dir():
            return sorted(
                path.resolve()
                for path in input_path.glob("*.parquet")
                if path.is_file()
            )
        raise FileNotFoundError(f"数据集路径不存在: {input_path}")

    def _build_description(self):
        base = (
            self.input_path.parent
            if self.input_path.is_file()
            else self.input_path
        )
        fingerprint_input = []
        shards = []
        for shard in self.shards:
            stat = shard.stat()
            fingerprint_input.append({
                "path": str(shard.relative_to(base)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "rows": self.shard_rows[shard],
            })
            shards.append({
                "path": str(shard),
                "rows": self.shard_rows[shard],
                "size": stat.st_size,
            })

        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_input,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "name": "FinTabNet",
            "version": "pubtables-full",
            "format": "parquet",
            "input_path": str(self.input_path),
            "shard_count": len(self.shards),
            "total_samples": self.total_samples,
            "shards": shards,
            "fingerprint": fingerprint,
            "fingerprint_method": "path+size+mtime",
        }

    def describe(self):
        return dict(self._description)

    def iter_samples(self, limit=None, start=0, batch_size=64):
        if start < 0:
            raise ValueError("start 不能小于 0")
        if limit is not None and limit < 0:
            raise ValueError("limit 不能小于 0")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if limit == 0:
            return

        selected = 0
        global_index = 0
        for shard in self.shards:
            parquet = pq.ParquetFile(shard)
            row_index = 0
            try:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    for row in batch.to_pylist():
                        if global_index < start:
                            global_index += 1
                            row_index += 1
                            continue

                        file_name = str(row.get("file_name") or "")
                        categories = row.get("category_ids") or []
                        boxes = row.get("boxes") or []
                        table_boxes = [
                            box
                            for category, box in zip(categories, boxes)
                            if int(category) == 1
                        ]
                        width = int(row.get("width") or 0)
                        height = int(row.get("height") or 0)
                        used_fallback = not table_boxes
                        table_bbox = (
                            list(table_boxes[0])
                            if table_boxes
                            else [0, 0, width, height]
                        )

                        image = row.get("image") or {}
                        image_bytes = image.get("bytes") or b""
                        sample_id = (
                            f"{shard.stem}:{row_index}:{file_name}"
                        )
                        yield TableSample(
                            sample_id=sample_id,
                            file_name=file_name,
                            row_index=row_index,
                            global_index=global_index,
                            shard_path=str(shard),
                            image_bytes=image_bytes,
                            width=width,
                            height=height,
                            table_bbox=table_bbox,
                            table_box_count=len(table_boxes),
                            used_full_image_fallback=used_fallback,
                            image_id=row.get("image_id"),
                            ocr_word_count=len(
                                row.get("ocr_words") or []
                            ),
                            ocr_box_count=len(
                                row.get("ocr_boxes") or []
                            ),
                        )
                        selected += 1
                        global_index += 1
                        row_index += 1
                        if limit is not None and selected >= limit:
                            return
            finally:
                parquet.close()
