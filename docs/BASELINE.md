# 阶段 0 基线说明

记录日期：2026-11-13

## 1. 代码基线

- 重构前可运行提交：`40f7c72`
- 当前运行模式：FastAPI 单进程应用
- PyTorch 与 Paddle 在同一进程运行，Windows 下必须先导入 PyTorch
- 表格识别使用 PP-TableMagic V2 GPU 完整产线
- 版面检测位于 `app/services/layout.py`
- OCR 模型和区域识别位于 `app/services/ocr.py`
- 表格识别位于 `app/services/table.py`
- 文档与 PDF 操作位于 `app/services/document.py`
- 目录、结构化和跨页表格合并位于 `app/services/structure.py`
- 表格模型适配器和注册表位于 `app/engines/`
- FinTabNet 数据集适配器位于 `app/datasets/fintabnet.py`
- FinTabNet.c XML 预处理位于 `app/datasets/preprocessing.py`
- 批量实验和断点续跑位于 `app/evaluation/runner.py`
- 指标重算位于 `app/evaluation/metrics.py`
- TEDS/S-TEDS 树编辑距离位于 `app/evaluation/teds.py`
- 前端样式和 ES Modules 位于 `frontend/`

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
| GET | `/api/models` | 模型清单和当前激活状态 |
| GET | `/api/models/health` | 注册表及适配器健康状态 |
| POST | `/api/models/reload` | 重载模型清单 |
| POST | `/api/models/{model_id}/activate` | 激活并加载模型 |
| GET | `/api/evaluations` | 列出有指标报告的实验 |
| GET | `/api/evaluations/latest/metrics` | 获取最新实验指标 |
| GET | `/api/evaluations/{experiment_id}/metrics` | 获取指定实验指标 |
| GET | `/api/datasets` | 数据集注册及可用状态 |
| GET | `/api/datasets/{dataset_id}/samples` | 分页浏览样本 |
| GET | `/api/datasets/{dataset_id}/samples/{sample_id}` | 获取样本元数据 |
| GET | `/api/datasets/{dataset_id}/samples/{sample_id}/image` | 获取样本图片 |
| POST | `/api/datasets/{dataset_id}/samples/{sample_id}/run` | 运行单样本 |
| GET | `/api/comparisons` | 列出模型对比报告 |
| GET | `/api/comparisons/{comparison_id}` | 获取对比详情 |
| GET | `/api/comparisons/{comparison_id}/export/{format}` | 下载 JSON/CSV/HTML |

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
- FinTabNet 完整测试目录：6 个分片，共 7,848 条样本。
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

## 7. 阶段 4 真实指标基线

实验：`experiments/stage3-smoke`

- 已检测样本：3。
- 表格检出率：1.0。
- 平均单元格数：24.0。
- 推理耗时 P50：564.09 ms。
- 推理耗时 P95：3084.80 ms。
- 单样本耗时吞吐量：0.744 samples/s，不含模型启动和数据读取。
- HTML 覆盖率：1.0。
- FinTabNet 单元格/HTML 真值覆盖：0。
- 单元格 IoU、文本召回、CER/WER、TEDS/TEDS-S：`null`，无真值。
