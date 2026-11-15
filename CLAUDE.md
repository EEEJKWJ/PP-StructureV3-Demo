# CLAUDE.md — PP-StructureV3 Demo

## 项目概述

PDF 文档版面结构化分析工具。上传 PDF → 版面检测 → 提取标题/表格/正文 → 输出结构化 JSON。
`server.py` 保留 FastAPI 路由和兼容入口，文档、版面、OCR、表格识别已拆到
`app/services/`，目录解析、页面结构化和跨页表格合并位于结构服务；
表格模型通过 `app/engines/` 的适配器和 JSON 注册表切换；
`index.html` 只保留页面结构；CSS 与原生 ES Modules 位于 `frontend/`，
不依赖 Node.js 构建。

## 架构与模型管线

```
PDF 上传
  ├─ PyMuPDF 渲染页面为 PNG
  ├─ PP-DocLayoutV3 (PyTorch, GPU) 版面检测 → blocks [{bbox, label}]
  │
  ├─ 文本型 PDF：PyMuPDF 直接提取文字/表格（毫秒级）
  └─ 扫描件 PDF：
       ├─ PP-OCRv5 ONNX (onnxruntime) 全页 OCR → 按区域归类
       └─ 表格区域 → PP-TableMagic V2 (PaddlePaddle) 表格结构识别
                      ├─ 表格分类（有线/无线自动区分，PP-LCNet）
                      ├─ 单元格检测 (RT-DETR-L)
                      ├─ 结构识别 (SLANeXt wired/wireless)
                      └─ 文字检测+识别 (PP-OCRv4 server)
                      降级方案：RapidTable (SLANet_plus ONNX)
```

## 关键文件

- `server.py` — FastAPI 应用、API 路由和旧函数兼容入口
- `app/services/layout.py` — PP-DocLayoutV3 加载、版面检测和阅读顺序
- `app/services/ocr.py` — PP-OCRv5 模型缓存、按区域 OCR 和去重
- `app/services/table.py` — 模型注册调用和标准表格结果
- `app/services/document.py` — PDF 保存、页面渲染、文字和表格提取
- `app/services/structure.py` — 目录、页面 JSON 和跨页表格合并
- `app/engines/registry.py` — 模型清单、激活、健康检查和回退
- `app/engines/tablemagic.py` — PP-TableMagic V2 适配器
- `app/engines/rapid_table.py` — RapidTable 适配器
- `app/datasets/fintabnet.py` — FinTabNet Parquet 分片与样本
- `app/datasets/registry.py` — 数据集路径注册和可用状态
- `app/evaluation/runner.py` — 逐样本实验、结果存储和断点续跑
- `app/evaluation/workbench.py` — 单样本浏览和重新识别
- `app/evaluation/metrics.py` — 基础指标、百分位和可选真值指标
- `scripts/run_fintabnet.py` — FinTabNet 批量运行 CLI
- `scripts/compute_metrics.py` — 从逐样本结果重算指标
- `configs/models/registry.json` — 默认模型注册清单
- `index.html` — 页面结构
- `frontend/css/app.css` — 原单文件页面的样式
- `frontend/js/app.js` — 上传、检测、页面选择和生命周期
- `frontend/js/state.js` — 页面状态和 localStorage
- `frontend/js/api.js` — FastAPI 请求
- `frontend/js/results.js` — JSON、区域、表格、树形和目录渲染
- `frontend/js/table-html.js` — 表格 HTML 白名单清理
- `frontend/js/workbench.js` — 单样本评测工作台
- `frontend/js/interactions.js` — 双向高亮联动
- `frontend/js/overlay.js` — 图片覆盖层
- `app/evaluation/reports.py` — 只读实验指标报告 API
- `model/` — PP-DocLayoutV3 模型缓存（首次启动自动下载）
- `model/ocr/` — PP-OCRv5 ONNX 模型缓存
- `~/.paddlex/official_models/` — PP-TableMagic V2 子模型缓存（8 个模型）

## 运行方式

```bash
source .venv/bin/activate
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True uvicorn server:app --host 0.0.0.0 --port 3003 --reload
```

环境变量 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 跳过 PaddleX 启动时的网络连通性检查（慢）。

## 依赖版本关键约束

### PaddlePaddle 版本问题（重要教训）

- **当前环境**：`torch==2.8.0+cu129` + `torchvision==0.23.0+cu129` + `nvidia-cudnn-cu12==9.10.2.21`
- **当前环境**：`paddlepaddle-gpu==3.4.0` (CUDA 12.9) + `paddleocr[doc-parser]==3.7.0` + `paddlex==3.7.2`
- RTX 50 系（Blackwell）必须使用 Paddle 官方 CUDA 12.9 索引，不要使用 PyPI 的旧版 GPU wheel
- PyTorch 和 Paddle 均使用 CUDA 12.9，并统一使用 cuDNN 9.10.2
- Windows 下必须先在 `server.py` 导入 PyTorch，再加载 Paddle；不要调整导入顺序

**当前方案**：PyTorch 版面检测和 PP-TableMagic V2 同进程运行，已通过真实版面模型、
TableMagic 完整产线和 GPU 张量运算验证。

### PP-TableMagic V2 初始化参数

```python
TableRecognitionPipelineV2(
    device="gpu",
    use_layout_detection=False,       # 已有 PP-DocLayoutV3，关闭 V2 内置版面检测
    use_doc_orientation_classify=False, # 裁切后的表格区域不需要
    use_doc_unwarping=False,           # 裁切后的表格区域不需要
)
```

**为什么关闭 `use_layout_detection`**：我们先用 PP-DocLayoutV3 检测表格位置并裁切，再传给 V2。如果 V2 再次做版面检测，会对裁切后的图片二次检测，导致表格结构错乱（列混淆、表头丢失）。

**为什么关闭 `use_doc_orientation_classify` 和 `use_doc_unwarping`**：同理，裁切后的表格区域方向已确定，不需要文档级预处理。如果未来改为传全页图给 V2，可以考虑重新开启。

## V2 输出结构

```python
res.json = {
    "res": {
        "table_res_list": [{
            "cell_box_list": [[x1,y1,x2,y2], ...],    # RT-DETR-L 检测的单元格框
            "pred_html": "<html><body><table>...</table>",  # 含 rowspan/colspan 的 HTML
            "table_ocr_pred": {
                "rec_texts": ["单元", "细目", ...],    # 每行文字
                "rec_boxes": [[x1,y1,x2,y2], ...],    # 每行文字坐标
                "rec_scores": [0.99, ...],
                "rec_polys": [[[x,y], ...], ...],
            }
        }],
        "layout_det_res": {...},
        "overall_ocr_res": {...},
    }
}
res.html = {"table_1": "<html>...</html>"}  # 便捷访问
```

### 如何恢复单元格内换行

pred_html 不保留换行。用 `table_ocr_pred.rec_texts` + `rec_boxes` 按 Y 坐标排序，同一单元格内多行文字用 `\n` 连接。已实现。

### 合并单元格

`pred_html` 包含 `rowspan`/`colspan` 属性，`_parse_html_table` 已在内部使用（计算 row/col 位置），但**当前未输出到 cell 数据中**。如需前端渲染合并效果，需要：
1. `_parse_html_table` 输出 `rowspan`/`colspan` 字段
2. 前端 `renderGrid` 跳过被占用的格子，`<td>` 加 `rowspan`/`colspan` 属性

代码已写过一次但用户要求撤回，可随时恢复。

## 乱码检测与 OCR 回退机制

三层防御：
1. **页级**：`page_has_text()` 调用 `is_garbage_text()` 检测全页文字质量，整页乱码直接走 OCR 路径
2. **Block 级**：文字型 PDF 逐区域检查提取文字，乱码区域单独 OCR（`ocr_single_block()`），结果写入 `ocr_by_block`
3. **表格级**：PyMuPDF 提取的表格单元格超过 30% 乱码，回退 PP-TableMagic V2

`build_structured()` 的 `get_text()` 优先读 `ocr_by_block`（即使 `has_text_layer=True`），确保回退结果被使用。

`is_garbage_text()` 检测规则：U+FFFD 替换字符 >10%、PUA 字符 >15%、有意义字符 <30%。

## 开发注意事项

- 服务启动时后台线程初始化三个引擎：版面模型、OCR 模型、表格引擎
- 表格引擎优先用 PP-TableMagic V2，import 失败或初始化失败时降级为 RapidTable
- 模型切换通过 `/api/models/{model_id}/activate`；加载成功后才替换当前模型
- 前端状态栏轮询 `/api/model-status`，表格引擎就绪后显示 "表格: PP-TableMagic V2"
- `uvicorn --reload` 开发模式下文件改动自动重载
- 前端 `.cells-table td` 需要 `white-space: pre-line` 才能渲染 `\n` 换行
- HTML 中保留 `onclick` 入口，`app.js` 启动时显式绑定到 `window`；
  修改事件函数名时需要同步这两处
- 模型 HTML 必须先经过 `sanitizeTableHtml()`；禁止直接插入模型返回原文
- FinTabNet 工作台真值状态为 `not_available`，不得用 OCR 词框伪造单元格真值
- 批量实验写入 `experiments/`，该目录和数据集一样不进入 Git；续跑以
  `results.jsonl` 中每个 `sample_id` 的最后一条结果为准
- `metrics.json` 必须只从逐样本结果重算；无真值指标写 `null` 和明确状态，
  不得把平均单元格数解释为结构准确率
