"""运行项目重构前的基线检查。"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def run(command):
    print(f"\n>>> {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(description="运行 PP-StructureV3 基线检查")
    parser.add_argument("--gpu", action="store_true", help="同时运行 CUDA 和 FinTabNet 检查")
    parser.add_argument("--limit", type=int, default=3, help="FinTabNet 样本数量")
    args = parser.parse_args()

    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.gpu:
        run([sys.executable, "verify_cuda.py"])
        run([sys.executable, "test_fintabnet.py", "--limit", str(args.limit)])


if __name__ == "__main__":
    main()
