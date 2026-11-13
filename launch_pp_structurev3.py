"""Launch PP-StructureV3 from any working directory."""

import argparse
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 PP-StructureV3 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=3003, help="监听端口")
    args = parser.parse_args()

    # server.py 使用相对路径，启动前先切换到项目目录。
    os.chdir(PROJECT_DIR)
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    import uvicorn
    from server import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
