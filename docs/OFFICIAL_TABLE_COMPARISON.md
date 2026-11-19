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
| 成功 | 72 | 90 |
| 成功率 | 72% | 90% |
| 仅单边成功 | 0 | 18 |
| 双方失败 | 10 | 10 |
| 本地版面补框 | 0 | 18 |

自定义路线比官方多完成 18 条，全部来自“官方无 table block、当前版面模型
补框后成功”的情况；没有出现官方成功而自定义失败的反向情况。

## 双方都成功的 72 条

| 指标 | 官方 | 自定义 |
|---|---:|---:|
| TEDS | 0.4128 | 0.3657 |
| S-TEDS | 0.8369 | 0.8260 |
| 文本召回 | 0.2692 | 0.3296 |
| 平均最佳 IoU | 0.4911 | 0.5977 |
| IoU ≥ 0.5 匹配率 | 0.4696 | 0.6620 |

## 全量成功样本

| 指标 | 官方 72 条 | 自定义 90 条 |
|---|---:|---:|
| TEDS | 0.4128 | 0.3762 |
| S-TEDS | 0.8369 | 0.8199 |
| 文本召回 | 0.2692 | 0.3316 |
| 平均最佳 IoU | 0.4911 | 0.5801 |
| IoU ≥ 0.5 匹配率 | 0.4696 | 0.6383 |
| CER | 0.5258 | 0.5040 |
| WER | 0.4504 | 0.4671 |

## 结论

- 官方表格的完整结构和 TEDS 更好。
- 当前 TableMagic 的文字召回和单元格 bbox 更强。
- 当前自定义注入能通过本地版面回退处理官方漏检表格，成功率更高。
- 延迟不能直接比较：官方统计完整页面解析，自定义统计的是表格注入和
  可选补框耗时。

因此目前不应替换默认表格模型。建议保留官方表格作为默认结构结果，
将自定义表格作为实验模型和官方漏检时的候选回退。

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

完整结果：

```text
experiments/official-table-comparisons/official-vs-custom-100-20261119/
```
