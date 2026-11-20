# PP-StructureV3 Demo 下一阶段任务交接

更新日期：2026-11-20

## 1. 当前仓库状态

- 项目根目录：`C:\ResearchProject\PP-StructureV3-Demo`
- 当前 Git 基线提交：`5c22f03`
- 分支：`main`
- 远程：`https://github.com/EEEJKWJ/PP-StructureV3-Demo`
- 官方参考 Fork：`https://github.com/EEEJKWJ/PaddleOCR`
- 官方 Fork 的 `main` 与 PaddleOCR 上游提交一致：
  `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`

开始下一阶段时，工作区可能有少量用户编辑的 docstring 和需求文档改动。
这些改动属于用户，必须保留，不得用 Git reset/checkout 覆盖。

## 2. 运行环境

使用项目 Conda 环境：

```powershell
C:\anaconda3\envs\PP-StructureV3_Demo\python.exe
```

不要使用系统默认 Python。

启动：

```powershell
cd C:\ResearchProject\PP-StructureV3-Demo
& "C:\anaconda3\envs\PP-StructureV3_Demo\python.exe" `
  launch_pp_structurev3.py --host 127.0.0.1 --port 3003
```

主要版本：

- PaddleOCR 3.7.0
- PaddleX 3.7.2
- PyTorch 2.8.0+cu129
- PaddlePaddle GPU 3.4.0

## 3. 当前文档解析模式

环境变量：

```text
PP_DOCUMENT_ENGINE=legacy
PP_DOCUMENT_ENGINE=official_v3
PP_DOCUMENT_ENGINE=official_v3_custom_table
```

默认是 `legacy`。

### Legacy

当前自研流程：

- PP-DocLayoutV3
- 项目 PP-OCRv5 ONNX
- PyMuPDF
- TableMagic / RapidTable
- 自研结构化和跨页表格合并

### official_v3

官方 PPStructureV3：

- 官方整页预处理
- PP-DocLayout_plus-L
- PP-OCRv5
- 官方公式和表格
- 当前 Demo 结果转换

### official_v3_custom_table

- 官方负责整页上下文。
- 官方表格子模型关闭。
- 用官方 table block bbox 裁图。
- 当前活动表格模型接管 table block。
- 官方表格结果保留在 `official_table`。
- 官方漏检 table block 时使用当前 PP-DocLayoutV3 补框。
- 可显式启用整图表格回退：

```powershell
$env:PP_TABLE_FULL_IMAGE_FALLBACK = "true"
```

## 4. 关键文件

| 文件 | 作用 |
|---|---|
| `server.py` | FastAPI、引擎开关、官方/legacy 路由 |
| `app/services/official_parser.py` | 官方 PPStructureV3 调用和结果转换 |
| `app/services/document.py` | PDF 页面渲染、文字层检测 |
| `app/services/layout.py` | 当前 PP-DocLayoutV3 |
| `app/services/ocr.py` | 当前 PP-OCRv5 ONNX 服务 |
| `app/services/table.py` | 表格模型服务和统一输出 |
| `app/engines/tablemagic.py` | TableMagic 适配器 |
| `app/evaluation/metrics.py` | IoU、文本、CER/WER、结构指标 |
| `app/evaluation/teds.py` | TEDS/S-TEDS |
| `scripts/compare_official_table_engines.py` | 官方/自定义 100 条对比 |
| `docs/OFFICIAL_TABLE_COMPARISON.md` | 当前 100 条实验结论 |
| `项目计划.md` | 项目总计划和阶段记录 |

## 5. 当前 100 条结论

数据：FinTabNet.c Structure test 前 100 条。

完成情况：

- 官方表格成功：72/100
- 自定义表格成功：100/100
- 18 条由当前版面模型补框
- 10 条整图表图回退

双方都成功的 72 条：

| 指标 | 官方 | 自定义 |
|---|---:|---:|
| TEDS | 0.4128 | 0.3657 |
| S-TEDS | 0.8369 | 0.8260 |
| 文本召回 | 0.2692 | 0.3296 |
| 平均最佳 IoU | 0.4911 | 0.5977 |
| IoU ≥ 0.5 匹配率 | 0.4696 | 0.6620 |
| CER | 0.5258 | 0.5045 |
| WER | 0.4504 | 0.4508 |

结论：

- 官方整体结构更稳定。
- 自定义局部文字和 cell bbox 更强。
- 自定义存在明显过分割，大型表和两个异常样本可达约 250 cells。
- 当前不能切换默认模型。

## 6. 下一轮五组实验

| ID | 路线 | 页面预处理 | 表格结构 | 文字来源 |
|---|---|---|---|---|
| A | 官方 | 官方整页预处理 | 官方 | 官方全页 PP-OCRv5 |
| B | 当前自定义 | 当前基线 | TableMagic | TableMagic OCR |
| C | 方案一 | 官方整页预处理 | TableMagic | TableMagic OCR |
| D | 方案二 | 官方整页预处理 | 官方 cells | 表格裁图整表 OCR |
| E | 控制组 | 官方整页预处理 | TableMagic | TableMagic OCR，不做表格局部增强 |

C、D 对表格局部增强；E 不做局部增强。
E 与 B 用于隔离页面预处理影响。

方案 D 禁止逐 cell OCR，必须在整张表格裁图上 OCR 一次，再按几何关系分配给 cells。

## 7. 推荐实现顺序

1. 官方 Parser 返回预处理后的页面图片和 table bbox。
2. 新增表格裁切和可逆坐标变换。
3. 新增表格局部预处理模块。
4. 新增整表 OCR 到官方 cells 的文字回填模块。
5. 统一五组运行器，支持断点续跑。
6. 先跑 5 条冒烟，再跑 100 条正式实验。
7. 输出 JSON/HTML 报告并更新正式对比文档。

## 8. 必须保持的约束

- 不修改默认引擎。
- 不替换默认表格模型。
- 不修改 `site-packages`。
- 不修改官方 PaddleOCR Fork，除非已有公开接口无法满足。
- 不伪造 cell bbox。
- 不把单元格数量增加当作结构改善。
- 必须报告 raw 和 canonical 两套文字指标。
- 必须记录成功样本集合和双方都成功的配对结果。
- 不提交模型缓存、数据缓存、实验大文件。

## 9. 模型和缓存

官方 PPStructureV3 模型保存在：

```text
C:\Users\Ying Jun\.paddlex\official_models
```

包括：

- PP-DocLayout_plus-L
- PP-OCRv5 server
- SLANeXt / SLANet_plus
- RT-DETR table cells
- UVDoc
- PP-FormulaNet_plus-L

模型权重不得提交 Git。

## 10. 验证命令

完整测试：

```powershell
& "C:\anaconda3\envs\PP-StructureV3_Demo\python.exe" `
  -m unittest discover -s tests -v
```

100 条对比续跑：

```powershell
& "C:\anaconda3\envs\PP-StructureV3_Demo\python.exe" `
  scripts\compare_official_table_engines.py `
  --resume experiments\official-table-comparisons\official-vs-custom-100-20261119
```

只重试失败样本：

```powershell
& "C:\anaconda3\envs\PP-StructureV3_Demo\python.exe" `
  scripts\compare_official_table_engines.py `
  --resume experiments\official-table-comparisons\official-vs-custom-100-20261119 `
  --retry-failed
```

## 11. 已知风险

- 官方预处理后的图像和原始图像可能尺寸不同。
- 表格裁切局部增强会引入新的坐标变换。
- 强二值化可能损伤财务数字和标点。
- 整表 OCR 行可能与官方 cell 边界正好交叉。
- 官方 bbox 漏检仍需本地布局或整图回退。
- 当前 TEDS 真值含大量引导点，raw 指标偏低。
- 官方公式模型首次下载约 735 MB，初始化较重。

## 12. 完成标准

下一轮完成时必须提供：

- 五组实验脚本和可恢复结果。
- 五组 100 条指标。
- 双方都成功的配对指标。
- 过分割、漏检、补框和回退统计。
- raw/canonical TEDS、S-TEDS、CER、WER。
- 每个阶段独立 Git 提交。
- 完整测试通过。
- 明确推荐是否启用新方案；未验证前不得切换默认模型。
