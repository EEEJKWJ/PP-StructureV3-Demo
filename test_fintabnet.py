"""Smoke-test PP-TableMagic V2 against a local FinTabNet Parquet shard."""

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

import server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default=r"D:\dataset\fintabnet-pubtables-full\test\test-00000-of-00006.parquet",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.parquet)
    batch = next(parquet.iter_batches(batch_size=args.limit))
    rows = batch.to_pylist()

    engine = server._get_table_engine()
    if server._table_engine_type != "paddle":
        raise RuntimeError(f"TableMagic unavailable: {server._table_engine_type}")

    successes = 0
    total_cells = 0
    elapsed = 0.0
    with tempfile.TemporaryDirectory(
        dir=Path.cwd(), prefix=".fintabnet_test_"
    ) as temp_dir:
        for index, row in enumerate(rows, 1):
            image_bytes = row["image"]["bytes"]
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype="uint8"),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                print(f"{index:02d} {row['file_name']}: image_decode=FAIL")
                continue

            table_boxes = [
                box
                for category, box in zip(row["category_ids"], row["boxes"])
                if category == 1
            ]
            bbox = table_boxes[0] if table_boxes else [
                0, 0, row["width"], row["height"]
            ]
            image_path = Path(temp_dir) / f"{index:02d}.jpg"
            image_path.write_bytes(image_bytes)

            started = time.perf_counter()
            result = server.recognize_table_structure(str(image_path), bbox)
            duration = time.perf_counter() - started
            elapsed += duration
            cell_count = len(result["cells"]) if result else 0
            total_cells += cell_count
            successes += int(cell_count > 0)
            print(
                f"{index:02d} {row['file_name']}: "
                f"size={row['width']}x{row['height']} "
                f"tables={int(cell_count > 0)} cells={cell_count} "
                f"time={duration:.2f}s"
            )

    count = len(rows)
    print(
        f"summary: samples={count} detected={successes} "
        f"detection_rate={successes / count if count else 0:.2%} "
        f"avg_cells={total_cells / count if count else 0:.2f} "
        f"avg_time={elapsed / count if count else 0:.2f}s"
    )


if __name__ == "__main__":
    main()
