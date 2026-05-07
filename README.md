# astrbot_plugin_NewAPI

米醋 `gpt-image-2-pro` 最小 Demo 插件，用于验证：

- `/生图 尺寸: 1k 提示词` 文生图。
- 普通图片消息静默缓存到当前会话。
- 当前图、引用图、缓存图的参考图优先级。
- `/重置` 清空当前会话参考图缓存。
- 后台任务完成后主动发送生成图，并尝试引用用户提示词消息。

## 配置

在插件配置中新增 `micu_gpt_image2` 供应商：

- `base_url`: 米醋 API Base URL。
- `api_keys`: 至少填一个 API Key。
- `max_reference_images`: Demo 默认 `1`。
- `max_request_size_mb`: Demo 默认 `20`。

模型固定为 `gpt-image-2-pro`，Demo 不调用 AstrBot 对话模型。

## 命令

```text
/生图 尺寸: 1k 一只白色猫
/重置
```

Demo 只接受 `1k`。`2k` / `4k` 会直接返回提示，不请求接口。

## 暂不包含

- 余额管理和扣费。
- `/批量生成`。
- 2K / 4K 文生图。
- mask、涂抹、局部编辑。
- 生产级异常恢复。
