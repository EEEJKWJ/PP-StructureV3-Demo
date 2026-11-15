# 表格模型注册清单

默认清单为 `registry.json`。可通过环境变量
`PP_STRUCTURE_MODEL_CONFIG` 指向其他 JSON 清单。

## 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 稳定模型 ID，只能包含字母、数字、点、下划线和连字符 |
| `name` | 否 | 页面和 API 显示名称，默认使用 ID |
| `version` | 否 | 模型或产线版本 |
| `adapter` | 是 | `tablemagic`、`rapidtable`、`http`、`onnx` 或 `pytorch` |
| `backend` | 否 | 运行后端，默认与适配器类型相同 |
| `device` | 否 | `gpu`、`cpu` 或适配器自定义值 |
| `source` | 否 | 模型来源说明；HTTP 适配器在此填写 endpoint 也可 |
| `fallback` | 否 | 加载失败时使用的其他模型 ID |
| `artifact_hash` | 否 | 模型文件哈希，未知时为 `null` |
| `options` | 否 | 适配器运行参数 |

## TableMagic 自定义模型

通过 `options.pipeline_kwargs` 直接传递
`TableRecognitionPipelineV2` 构造参数。例如：

```json
{
  "adapter": "tablemagic",
  "device": "gpu",
  "options": {
    "pipeline_kwargs": {
      "use_layout_detection": false,
      "wired_table_structure_recognition_model_dir": "D:/models/custom-wired"
    }
  }
}
```

## RapidTable 自定义模型

通过 `options.model_dir_or_path` 指定本地模型目录或文件：

```json
{
  "adapter": "rapidtable",
  "options": {
    "model_type": "SLANETPLUS",
    "model_dir_or_path": "D:/models/rapid-table.onnx"
  }
}
```

## HTTP 标准接口

HTTP 适配器向 endpoint 发送 JSON：

```json
{
  "model_id": "remote-table",
  "image_base64": "<JPEG Base64>",
  "image_format": "jpeg",
  "bbox": [0, 0, 100, 80],
  "options": {}
}
```

响应可直接返回 `html`、`cells`，也可放在 `result` 对象内。默认认为
`cells[].bbox` 是裁剪图坐标，适配器会自动转换为页面坐标；如果服务已返回
页面全局坐标，设置 `options.cells_relative=false`。

`onnx` 和 `pytorch` 当前是显式扩展占位符。配置可以校验，但激活会返回
“适配器尚未实现”，并尝试清单中的回退模型。

## 切换与回退

- `GET /api/models` 查看模型清单和当前模型。
- `GET /api/models/health` 检查当前适配器。
- `POST /api/models/reload` 重新读取 JSON。
- `POST /api/models/{id}/activate` 加载并切换模型。

新模型加载成功前不会替换当前模型。若当前没有可用模型，注册表会尝试目标
模型的 `fallback`，再尝试 `default_model`。激活失败响应中同时包含
`error`、实际的 `active_model` 和 `fallback_used`。
