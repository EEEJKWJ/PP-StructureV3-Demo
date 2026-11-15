# PP-StructureV3 Demo

PDF 文档版面结构化分析工具。上传 PDF 后自动检测版面区域，提取标题、表格、正文等结构化内容，输出干净的 JSON 数据。

**智能双通道**：文本型 PDF 直接提取（快速精准），扫描件 PDF 自动走 OCR + 表格结构识别。

![screenshot](screenshot.png)

## 功能

### 版面检测
- 使用 [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) 模型，识别标题、表格、文本、公式等区域
- 首次启动自动下载模型并保存到项目 `model/` 目录，之后离线可用

### 文本提取（双通道）
- **文本型 PDF**：PyMuPDF 直接提取文字层（毫秒级）
- **扫描件 PDF**：PP-OCRv5 ONNX 自动识别（GPU 用 server 版，CPU 用 mobile 版）
- 逐页智能判断，混合 PDF 也能正确处理

### 表格结构
- **文本型 PDF**：PyMuPDF 提取单元格级结构（行、列、合并单元格、文字内容）
- **扫描件 PDF**：PP-TableMagic V2 完整产线识别表格，输出单元格和 HTML 表格
- PP-TableMagic 不可用时自动降级到 RapidTable (SLANet_plus)
- 表格支持「网格」和「树形」两种可视化视图
- 表格结果可展开 HTML 渲染和源码视图，模型 HTML 经过安全白名单清理

### 目录解析
- 自动识别目录页，拆分标题和页码
- 引导线（省略号）自动去除
- 分离的页码行自动合并回标题

### 可视化校对
- 双向联动：点击右侧结果 ↔ 图片对应区域高亮定位
- 所有区域直接显示提取到的文字内容
- 扫描件表格渲染 HTML 表格视图
- 结果缓存到 localStorage，刷新不丢失
- 右侧面板宽度可拖动调整
- 顶部可直接切换已注册表格模型
- 结果面板底部显示已有批量实验的真实指标

### 结构化 JSON 输出
- `title`：标题文字，按行拆分去重
- `table`：表头 + 行数据，合并单元格自动继承，多行值拆为数组
- `table_html`：扫描件表格 HTML 结构
- `toc`：目录条目，标题 + 页码分离
- `text`：正文段落

## 技术栈

| 组件 | 技术 |
|------|------|
| 版面检测 | [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) (transformers + PyTorch) |
| OCR 引擎 | [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) ONNX (onnxruntime) |
| 表格结构识别 | 可注册适配器；默认 PP-TableMagic V2，可切换 RapidTable |
| PDF 处理 | PyMuPDF（渲染 + 文字提取 + 表格提取）|
| 后端 | FastAPI |
| 前端 | 原生 HTML + ES Modules（无框架、无构建步骤）|

## 快速开始

```bash
# 创建虚拟环境（需要 Python 3.11）
python3.11 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动（推荐）
python launch_pp_structurev3.py
```

访问 http://localhost:3003

`launch_pp_structurev3.py` 会自动切换到项目目录，因此也可以从其他工作目录启动：

```bash
python path/to/PP-StructureV3-Demo/launch_pp_structurev3.py
```

如需修改监听地址或端口：

```bash
python launch_pp_structurev3.py --host 127.0.0.1 --port 3005
```

首次启动会自动下载模型（版面检测 ~127MB + OCR ~164MB），页面顶部显示加载状态。

### 国内用户加速下载

```bash
export HF_ENDPOINT=https://hf-mirror.com
python launch_pp_structurev3.py
```

### GPU 加速

有 NVIDIA GPU 时自动启用：
- 版面检测模型加载到 GPU
- OCR 自动切换为 server 版（精度更高）
- 安装与 CUDA 12.x / cuDNN 9 匹配的 ONNX Runtime GPU：

当前统一环境为 PyTorch `2.8.0+cu129`、torchvision `0.23.0+cu129`、
PaddlePaddle GPU `3.4.0` 和 cuDNN `9.10.2`。Windows 下 `server.py` 必须先导入
PyTorch，再加载 Paddle；不要调整这两个框架的导入顺序。

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu==1.23.2"
```

`onnxruntime` 和 `onnxruntime-gpu` 不能同时安装，否则可能加载到 CPU 动态库并在
不报错的情况下回退到 CPU。项目启动时会检查 CUDA provider；GPU 就绪时 OCR
推理由 `CUDAExecutionProvider` 执行。

`rapid_table` 和当前版本的 PyMuPDF 包元数据仍声明依赖 `onnxruntime`，因此在仅安装
`onnxruntime-gpu` 后运行 `pip check` 可能报告“onnxruntime is not installed”。
这是同模块不同发行包的元数据限制，不影响 GPU 版导入和推理；不要为此重新安装 CPU 包。

### FinTabNet Parquet 冒烟测试

可直接读取 HuggingFace Parquet 格式的 FinTabNet 分片，并按文件中的 Table
类别框裁剪后调用当前注册模型。每次运行会创建独立实验目录并保存逐样本结果：

```bash
python test_fintabnet.py --limit 10
```

指定注册模型：

```bash
python test_fintabnet.py --limit 1 --model rapidtable-slanet-plus
```

默认文件是
`D:\dataset\fintabnet-pubtables-full\test\test-00000-of-00006.parquet`，也可通过
`--parquet` 指定其他分片或整个分片目录。完整测试目录当前包含 6 个分片、
7,848 条样本；日常验证建议只运行前几条。

每个实验目录包含：

```text
experiments/{experiment_id}/
├─ experiment.json    # 数据集指纹、模型和运行参数
├─ results.jsonl      # 逐样本 cells、HTML、耗时和状态
├─ failures.jsonl     # 当前失败样本
├─ metrics.json       # 可独立重算的指标和状态
└─ summary.json       # 累计统计及 metrics 副本
```

断点续跑：

```bash
python test_fintabnet.py --resume experiments/stage3-smoke
```

续跑默认跳过已完成成功样本；加 `--retry-failed` 可重新执行失败样本：

```bash
python test_fintabnet.py --resume experiments/stage3-smoke --retry-failed
```

### 实验指标

从已有逐样本结果重新计算指标，不需要重新运行模型：

```bash
python scripts/compute_metrics.py --experiment-dir experiments/stage3-smoke
```

当前基础指标包括表格检出率、平均单元格数、成功率、HTML 覆盖率、
尝试次数、推理耗时 P50/P95，以及基于单样本耗时总和的吞吐量。
FinTabNet 没有单元格或 HTML 结构真值，因此单元格 IoU、文本召回、
CER/WER、TEDS/TEDS-S 会明确标记为无真值，不会填入推测值。

### FinTabNet.c 完整结构测试集

`FinTabNet.c-Structure` 使用图像、XML 几何结构和词框 JSON。先规范化 XML：

```bash
python scripts/prepare_fintabnet_c.py \
  --root D:\dataset\FinTabNet.c\FinTabNet.c-Structure \
  --split test
```

预处理会在 `data/processed/fintabnet-c-structure/test.jsonl` 生成被 Git 忽略的
缓存，包括单元格行列、rowspan/colspan、像素坐标、文本和规范化 HTML。

使用当前模型测试：

```bash
python scripts/run_fintabnet.py \
  --dataset fintabnet-c-structure-test \
  --limit 20
```

完整 test 共 9,289 张表，可用 `--limit 9289` 运行。全量耗时较长，支持通过
实验目录断点续跑。结构真值可用时，报告会计算单元格 IoU、文本召回、CER/WER
和结构邻接关系 Precision/Recall/F1。

### 模型注册与切换

默认清单为 `configs/models/registry.json`，包含 PP-TableMagic V2 和
RapidTable。可通过 `PP_STRUCTURE_MODEL_CONFIG` 指向其他 JSON 清单。
字段和自定义模型示例见 [configs/models/README.md](configs/models/README.md)。

```bash
# 查看模型及当前激活状态
curl http://localhost:3003/api/models

# 健康检查
curl http://localhost:3003/api/models/health

# 激活 RapidTable
curl -X POST http://localhost:3003/api/models/rapidtable-slanet-plus/activate

# 修改 JSON 后重载清单
curl -X POST http://localhost:3003/api/models/reload
```

新模型加载成功前不会替换当前模型。初始加载失败时会尝试目标模型的
`fallback` 和 `default_model`，响应中会明确返回实际生效模型。

页面顶部模型选择框也使用同一套切换接口。

### 历史实验指标

```bash
curl http://localhost:3003/api/evaluations
curl http://localhost:3003/api/evaluations/latest/metrics
curl http://localhost:3003/api/evaluations/stage3-smoke/metrics
```

页面底部指标区读取最新实验报告，显示检出率、平均单元格数、P50/P95 和
吞吐量，并标明评测模型、数据集版本和结构指标状态。没有报告时显示明确占位。

### 单样本评测工作台

页面右上视图切换中的「样本评测」可以浏览 FinTabNet 测试样本，查看原图，
使用当前模型重新识别，并查看预测 HTML/源码和无真值状态。

```bash
curl http://localhost:3003/api/datasets
curl "http://localhost:3003/api/datasets/fintabnet-test/samples?offset=0&limit=20"
```

默认数据集路径为：

```text
D:\dataset\fintabnet-pubtables-full\test
```

可通过环境变量覆盖：

```bash
export PP_FINTABNET_PATH=/path/to/fintabnet/test
```

也可直接打开 `http://localhost:3003/#workbench`。FinTabNet 当前没有单元格或
HTML 结构真值，工作台会明确显示“无结构真值”，不会生成伪对比。

### 模型对比与报告

对同一数据集范围运行多个模型：

```bash
python scripts/compare_models.py \
  --models pp-tablemagic-v2,rapidtable-slanet-plus \
  --limit 10
```

续跑已有对比：

```bash
python scripts/compare_models.py \
  --resume experiments/comparisons/stage8-real
```

对比目录包含每个模型的独立实验、逐样本对齐结果以及：

- `report.json`
- `report.csv`
- `report.html`
- `comparison_samples.jsonl`

页面「模型对比」视图（`/#comparisons`）可查看模型指标、逐样本胜者和下载报告。
没有结构真值时，质量胜者显示“不可比较”；完成胜负和延迟胜负分别统计。

### 基线检查

纯函数回归测试：

```bash
python scripts/check_baseline.py
```

包含 CUDA 和 FinTabNet 真实样本检查：

```bash
python scripts/check_baseline.py --gpu --limit 3
```

基线环境和接口说明见 [docs/BASELINE.md](docs/BASELINE.md)。

## 结构化输出示例

```json
{
  "page": 1,
  "content": [
    { "type": "title", "text": "口腔内科学考试大纲" },
    { "type": "title", "text": "基础知识" },
    {
      "type": "table",
      "headers": ["单元", "细目", "要点", "要求"],
      "rows": [
        {
          "单元": "一、牙体组织",
          "细目": "1．釉质",
          "要点": ["⑴ 理化特性", "⑵ 组织学特点", "⑶ 临床意义"],
          "要求": "掌握"
        }
      ]
    },
    {
      "type": "toc",
      "items": [
        { "title": "第一章 牙体组织", "page": 1 },
        { "title": "第二节 牙周膜", "page": 8 }
      ]
    }
  ]
}
```

## 项目结构

```
├── launch_pp_structurev3.py # Python 启动器（自动切换项目目录）
├── server.py           # FastAPI 应用、路由和兼容入口
├── app/services/       # 文档、版面、OCR、表格和结构化服务
├── app/engines/        # 表格模型适配器、注册与回退
├── app/datasets/       # FinTabNet 等表格数据集适配器
├── app/evaluation/     # 批量实验、断点续跑和后续指标
├── configs/models/     # 模型注册清单和自定义配置说明
├── scripts/            # 批量运行等命令行入口
├── index.html          # 页面结构
├── frontend/css/       # 页面样式
├── frontend/js/        # 状态、API、结果、高亮和覆盖层模块
├── ppocr5.py           # PP-OCRv5 ONNX 推理引擎
├── requirements.txt    # Python 依赖
├── model/              # 版面检测模型（首次启动自动下载）
│   └── ocr/            # OCR 模型（首次启动自动下载）
└── .gitignore
```

前端不需要 Node.js。浏览器直接通过 FastAPI 的 `/frontend/` 静态路由加载 CSS
和 ES Modules。
