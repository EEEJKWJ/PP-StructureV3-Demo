"""预处理 FinTabNet.c-Structure XML 为统一结构真值缓存。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.datasets.preprocessing import prepare_split


DEFAULT_ROOT = r"D:\dataset\FinTabNet.c\FinTabNet.c-Structure"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "processed" / "fintabnet-c-structure"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="预处理 FinTabNet.c-Structure XML",
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit 不能小于 0")

    output = Path(
        args.output
        or DEFAULT_OUTPUT / f"{args.split}.jsonl"
    )

    def progress(index, total, written, failures):
        print(
            f"{index}/{total} processed written={written} "
            f"failures={failures}",
            flush=True,
        )

    metadata = prepare_split(
        root=args.root,
        split=args.split,
        output_path=output,
        limit=args.limit,
        overwrite=args.overwrite,
        progress=progress,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"cache={metadata['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
