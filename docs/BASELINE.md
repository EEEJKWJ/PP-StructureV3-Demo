# 阶段 0 基线说明

记录日期：2026-11-13

## 1. 代码基线

- 重构前可运行提交：`40f7c72`
- 当前运行模式：FastAPI 单进程应用
- PyTorch 与 Paddle 在同一进程运行，Windows 下必须先导入 PyTorch
- 表格识别使用 PP-TableMagic V2 GPU 完整产线
- 表格识别位于 `app/services/table.py`
- 文档与 PDF 操作位于 `app/services/document.py`

## 2. 运行环境

| 组件 | 基线版本 |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.8.0+cu129 |
| torchvision | 0.23.0+cu129 |
| PaddlePaddle GPU | 3.4.0 |
| PaddleOCR | 3.7.0 |
| PaddleX | 3.7.2 |
| CUDA Runtime | 12.9 |
| cuDNN | 9.10.2.21 |
| ONNX Runtime GPU | 1.23.2 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| GPU 算力 | 12.0 |

`pip check` 的已知提示：

- PaddlePaddle 元数据固定要求 cuDNN 9.9，而当前统一运行环境使用 9.10.2。
- PyMuPDF 和 RapidTable 依赖名为 `onnxruntime` 的发行包，而项目实际安装的是
  `onnxruntime-gpu`。

以上提示不影响当前 GPU 模型运行，正式验证以真实推理结果为准。

## 3. 当前 API

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 返回单页前端 |
| GET | `/api/model-status` | 返回版面、OCR 和表格模型状态 |
| POST | `/api/upload` | 上传 PDF 或图片 |
| GET | `/api/images/{doc_id}/{filename}` | 获取页面图片 |
| GET | `/api/text-source/{doc_id}/{page_num}` | 判断页面文字来源 |
| POST | `/api/detect/{doc_id}/{page_num}` | 识别单页 |
| POST | `/api/detect-all/{doc_id}` | 识别全部页面 |
| DELETE | `/api/documents/{doc_id}` | 删除上传文档 |

阶段 1A 起，表格 block 同时返回 `cells` 和 `table_html`；结构化 table 项包含
`headers`、`rows` 和 `html`。原始 `cells`、`headers`、`rows` 字段保持兼容。

## 4. 基线检查命令

仅运行纯函数回归测试：

```bash
python scripts/check_baseline.py
```

包含 CUDA 和 FinTabNet 真实样本检查：

```bash
python scripts/check_baseline.py --gpu --limit 3
```

也可以分别执行：

```bash
python -m unittest discover -s tests -v
python verify_cuda.py
python test_fintabnet.py --limit 3
```

## 5. 已记录的真实样本结果

- FinTabNet 第一条分片：1,308 条样本。
- 前 10 条烟雾测试：10/10 表格检出。
- 单元格数范围：8～70。
- 平均单元格数：36.2。
- 平均耗时：0.69 秒/张，包含首张预热。
- FastAPI 单页接口已验证能够返回表格单元格和结构化行列。

## 6. 阶段 0 验收

- 纯函数回归测试通过。
- CUDA 12.9 同进程张量运算通过。
- 至少 3 条 FinTabNet 样本完成表格识别。
- `/api/model-status` 返回 `status=ready`、`gpu=true`、`table_engine=paddle`。
- 现有 API 路径、页面布局和识别参数保持不变。
