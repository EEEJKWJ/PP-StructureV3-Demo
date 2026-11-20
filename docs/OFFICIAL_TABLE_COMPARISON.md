# 官方表格与自定义表格 100 条对比

日期：2026-11-19  
数据：FinTabNet.c Structure test 前 100 条  
自定义表格模型：`pp-tablemagic-v2`

## 对比方法

两条路线使用同一官方 PP-StructureV3 整页版面、PP-OCRv5 和公式上下文：

- 官方：使用 PP-StructureV3 默认表格子流程。
- 自定义：用同一页面结果中的 table block，调用当前注册的
  `pp-tablemagic-v2`；官方漏检时允许使用当前 PP-DocLayoutV3 补表格框。

结果为双方都成功样本的配对比较，避免成功样本集合不同导致偏差。

## 完成情况

| 项目 | 官方 | 自定义注入 |
|---|---:|---:|
| 成功 | 72 | 100 |
| 成功率 | 72% | 100% |
| 仅单边成功 | 0 | 28 |
| 双方失败 | 0 | 0 |
| 本地版面补框 | 0 | 28 |
| 其中整图表格回退 | 0 | 10 |

自定义路线比官方多完成 28 条。其中 18 条由当前版面模型补充表格框，
10 条在两种版面模型均未找到 table block 时，将已裁切的表图整图作为
表格处理。没有出现官方成功而自定义失败的反向情况。

## 双方都成功的 72 条

| 指标 | 官方 | 自定义 |
|---|---:|---:|
| TEDS | 0.4128 | 0.3657 |
| S-TEDS | 0.8369 | 0.8260 |
| 文本召回 | 0.2692 | 0.3296 |
| 平均最佳 IoU | 0.4911 | 0.5977 |
| IoU ≥ 0.5 匹配率 | 0.4696 | 0.6620 |
| CER | 0.5258 | 0.5045 |
| WER | 0.4504 | 0.4508 |

## 全量成功样本

| 指标 | 官方 72 条 | 自定义 100 条 |
|---|---:|---:|
| TEDS | 0.4128 | 0.3720 |
| S-TEDS | 0.8369 | 0.8129 |
| 文本召回 | 0.2692 | 0.3023 |
| 平均最佳 IoU | 0.4911 | 0.5619 |
| IoU ≥ 0.5 匹配率 | 0.4696 | 0.6035 |
| CER | 0.5258 | 0.5170 |
| WER | 0.4504 | 0.4965 |

## 指标下降原因

### TEDS 和 S-TEDS

在双方都成功的 72 条中，自定义 TEDS 降低约 0.047，S-TEDS 降低约 0.011。

- 72 条中，自定义 TEDS 劣于官方的有 42 条，优于官方的有 24 条。
- S-TEDS 劣于官方的有 28 条，优于官方的有 26 条。
- 主要下降集中在大型表、密集财务表和部分窄长两列表。
- TableMagic 更容易产生过多空单元格和过分割。
- 两个最明显样本 `ADS_2009_page_185_table_0` 和
  `ADS_2009_page_186_table_0` 的真值只有 24 和 20 个单元格附近，
  当前模型却生成约 250 个单元格，导致结构树严重失配。
- 这些错误会同时降低 TEDS 和只比较结构的 S-TEDS。

### WER

双方都成功样本的 WER 接近：

```text
官方 0.4504
自定义 0.4508
```

全量自定义 WER 上升到 `0.4965`，主要因为新增的 28 条本身是两种官方版面
都难以定位的困难样本，而不是这 28 条之外的常用样本普遍退化。

WER 还比 CER 更敏感：

- CER 按字符比较，当前模型即使合并单词，字符仍可能大部分正确。
- WER 按单词 token 比较，空格丢失、单词合并或拆分会产生额外词错误。
- 例如 `Income before income taxes` 被输出为连续单词时，CER 只增加少量
  字符差异，但 WER 可能把多个词全部记为错误。

## 结论

- 官方表格的完整结构和 TEDS 更好。
- 当前 TableMagic 的文字召回和单元格 bbox 更强。
- 当前自定义注入能通过本地版面回退处理官方漏检表格，成功率更高。
- 延迟不能直接比较：官方统计完整页面解析，自定义统计的是表格注入和
  可选补框耗时。

因此目前不应替换默认表格模型。建议保留官方表格作为默认结构结果，
将自定义表格作为实验模型和官方漏检时的候选回退。

## 后续五组实验

在后续实验中对“官方页面预处理”和“表格局部增强”进行了拆分。正式结果见
[`TABLE_PREPROCESS_EXPERIMENT.md`](TABLE_PREPROCESS_EXPERIMENT.md)。
在五组共同成功的 58 条中，当前自定义 B 的 TEDS/S-TEDS、文字召回和 cell
bbox 均为第一；官方去畸变和现有局部增强组合没有带来稳定的整体收益。

## 复现

```powershell
python scripts\compare_official_table_engines.py `
  --limit 100 `
  --comparison-id official-vs-custom-100-20261119
```

续跑：

```powershell
python scripts\compare_official_table_engines.py `
  --resume experiments\official-table-comparisons\official-vs-custom-100-20261119
```

仅重试失败样本：

```powershell
python scripts\compare_official_table_engines.py `
  --resume experiments\official-table-comparisons\official-vs-custom-100-20261119 `
  --retry-failed
```

完整结果：

```text
experiments/official-table-comparisons/official-vs-custom-100-20261119/
```
