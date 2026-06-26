# astrbot_plugin_NewAPI

OpenAI `gpt-image-2` 最小 Demo 插件，用于验证：

- `/生图` 开启生图功能，后续直接发送提示词即可生图。
- `/结束生图` 关闭生图功能，后续普通消息不再触发生图。
- 开启后发送 `提示词` 即可生图。
- `/批量生成 3 提示词` 批量文生图。
- 普通图片消息静默缓存到当前会话。
- 当前图、引用图、缓存图的参考图优先级。
- `/重置` 清空当前会话参考图缓存。
- 后台任务完成后主动发送生成图，并尝试引用用户提示词消息。
- Demo 仅处理好友消息，群消息不缓存图片、不响应命令。

## 配置

在插件配置中新增 API 供应商分组：

- `name`: 分组名称，用户余额管理里通过该名称选择供应商。
- `remark`: 备注。
- `base_url`: API Base URL。
- `text_to_image_enabled`: 是否启用文生图。
- `text_to_image_endpoint`: 文生图接口地址。
- `image_to_image_enabled`: 是否启用图生图。
- `image_to_image_endpoint`: 图生图接口地址，官方 gpt-image-2 单图和多图参考都使用该接口。
- `multi_reference_endpoint`: 兼容旧配置保留；官方 gpt-image-2 多参考图同样使用 `/v1/images/edits`。
- `max_reference_images`: Demo 默认 `1`。
- `max_request_size_mb`: Demo 默认 `20`。

在插件配置中新增 `balance_users` 用户：

- `name`: 用户名称。
- `umo`: 允许使用生图的会话 UMO。
- `enabled`: 关闭后该 UMO 禁止生图。
- `provider_group`: 选择供应商分组，填写 API 供应商配置中的分组名称。
- `api_key`: 该用户使用的一条 API Key。
- `image_model`: 生图模型，例如 `gpt-image-2`。
- `image_quality`: 生图质量，可选 `auto` / `low` / `medium` / `high`。
- `image_size`: 图像分辨率，留空使用默认尺寸，也可填 `auto` 或 `1024x1024` 等官方尺寸。
- `cost_per_image`: 每生成 1 张图扣除的额度。
- `add_amount`: 充值额度；插件加载或检查余额时会写入真实余额并归零。
- `balance_display`: 只作展示，真实余额以插件数据目录中的 `balances.json` 为准。

模型固定为 `gpt-image-2`，Demo 不调用 AstrBot 对话模型。

## 命令

```text
/生图
一只白色猫
/批量生成 3 一只白色猫
/结束生图
/重置
```

直接发送的普通消息会整体作为提示词；分辨率使用用户配置里的 `image_size`，留空时按默认尺寸生成。
`/生图` 仅用于开启生图功能，不再接受提示词参数。`/批量生成` 需要先开启生图功能，数量最大按 4 张执行。
带参考图的 `/批量生成` 会按 1 张执行并提示用户。

## 暂不包含

- mask、涂抹、局部编辑。
- 生产级异常恢复。
