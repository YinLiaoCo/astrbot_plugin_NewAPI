from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.star.star_tools import StarTools
from astrbot.core.utils.io import download_image_by_url

from .adapter import MicuGPTImage2Adapter, MicuImageInput
from .core import BalanceManager, ReferenceBuffer, parse_generate_command


@dataclass(frozen=True)
class MicuProviderConfig:
    name: str
    base_url: str
    text_to_image_endpoint: str
    image_to_image_endpoint: str
    multi_reference_endpoint: str
    supports_text_to_image: bool
    supports_image_to_image: bool
    proxy: str | None
    max_reference_images: int
    max_request_size_mb: int


@dataclass(frozen=True)
class MicuRuntimeConfig:
    provider: MicuProviderConfig
    api_key: str


@register("astrbot_plugin_NewAPI", "cl", "米醋 gpt-image-2-pro Demo", "0.1.0")
class MicuImageDemoPlugin(Star):
    """Minimal demo for Micu gpt-image-2-pro generation."""

    def __init__(self, context: Context, config: Any):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self.reference_buffer = ReferenceBuffer()
        self.provider_configs = self._load_provider_configs()
        self.tasks: set[asyncio.Task] = set()

        self.data_dir = Path(StarTools.get_data_dir()) / "astrbot_plugin_NewAPI"
        self.balance_manager = BalanceManager(self.data_dir, self.config)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        if not self.provider_configs:
            logger.warning("[MicuDemo] 未配置可用 API 供应商，/生图 将不可用")
            return

        logger.info(f"[MicuDemo] 插件加载完成，已配置 {len(self.provider_configs)} 个 API 分组")

    async def terminate(self):
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def handle_message(self, event: AstrMessageEvent):
        """Handle demo commands and silently cache non-command images."""
        if not self._is_friend_message(event):
            return

        message = self._get_event_text(event)
        command = self._parse_demo_command(message)

        if command == "reset":
            self.reference_buffer.clear(event.unified_msg_origin)
            yield event.plain_result("已重置当前会话参考图")
            return

        if command == "generate":
            async for result in self._handle_generate_command(event, message):
                yield result
            return

        count = await self.reference_buffer.cache_plain_message_images(
            event,
            self._download_image,
        )
        if count:
            logger.info(
                f"[MicuDemo] 已缓存 {count} 张参考图，UMO={event.unified_msg_origin}"
            )

    async def _handle_generate_command(self, event: AstrMessageEvent, message: str):
        selection = await self.reference_buffer.select_for_command(
            event,
            self._download_image,
        )
        logger.info(
            f"[MicuDemo] 收到生图命令，UMO={event.unified_msg_origin}，"
            f"文本={message!r}，参考图来源={selection.source}，参考图数量={len(selection.images)}"
        )
        parsed = parse_generate_command(message, selection.has_images)
        if not parsed.ok or not parsed.command:
            yield event.plain_result(parsed.error or "格式错误，请使用 /生图 4:3 1k 提示词")
            return

        balance_check = self.balance_manager.precheck(event.unified_msg_origin, parsed.command.n)
        if not balance_check.ok:
            yield event.plain_result(balance_check.message or "余额不足，禁止生图")
            return

        runtime = self._runtime_config_for(event.unified_msg_origin)
        if runtime is None:
            yield event.plain_result("请先在用户余额管理中为当前用户配置 API Key 和有效的供应商分组")
            return

        provider = runtime.provider
        if selection.has_images and not provider.supports_image_to_image:
            yield event.plain_result(f"当前分组 {provider.name} 未启用图生图能力")
            return
        if not selection.has_images and not provider.supports_text_to_image:
            yield event.plain_result(f"当前分组 {provider.name} 未启用文生图能力")
            return

        images = selection.images[: provider.max_reference_images]
        if not self._request_size_ok(images, provider):
            yield event.plain_result("参考图过大，已超过当前供应商 max_request_size_mb 限制")
            return

        if parsed.command.warning:
            yield event.plain_result(parsed.command.warning)

        yield event.plain_result("正在生图中，可以继续写提示词")

        self.reference_buffer.clear_after_task_created(event.unified_msg_origin)
        task = asyncio.create_task(
            self._generate_and_send(
                prompt=parsed.command.prompt,
                unified_msg_origin=event.unified_msg_origin,
                runtime=runtime,
                images=images,
                size=parsed.command.size,
                source=selection.source,
                reply_message_id=self._get_message_id(event),
                requested_count=parsed.command.n,
            )
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def _parse_demo_command(self, message: str) -> str | None:
        text = message.strip()
        for prefix in ("/", "／"):
            if text.startswith(f"{prefix}重置"):
                return "reset"
            if text.startswith(f"{prefix}生图"):
                return "generate"
            if text.startswith(f"{prefix}批量生成"):
                return "generate"

        # Some adapters may remove the wake prefix before plugins see message_str.
        if text == "重置" or text.startswith("重置 "):
            return "reset"
        if text == "生图" or text.startswith("生图 "):
            return "generate"
        if text == "批量生成" or text.startswith("批量生成 "):
            return "generate"
        return None

    def _is_friend_message(self, event: AstrMessageEvent) -> bool:
        return ":FriendMessage:" in str(getattr(event, "unified_msg_origin", ""))

    async def _generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        runtime: MicuRuntimeConfig,
        images: list[tuple[bytes, str]],
        size: str,
        source: str,
        reply_message_id: str | None,
        requested_count: int,
    ) -> None:
        adapter = self._make_adapter(runtime)
        task_id = hashlib.md5(f"{time.time()}{unified_msg_origin}".encode()).hexdigest()[:8]
        try:
            if images:
                inputs = [
                    MicuImageInput(
                        data=data,
                        filename=f"reference_{index}{self._extension_for_mime(mime)}",
                        content_type=mime,
                    )
                    for index, (data, mime) in enumerate(images, start=1)
                ]
                generated = await adapter.image_to_image(prompt, inputs, size=size)
            else:
                generated = await adapter.text_to_image(prompt, size=size, n=requested_count)
        except Exception as exc:
            logger.error(f"[MicuDemo] 任务 {task_id} 生成失败: {exc}", exc_info=True)
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"生成失败: {exc}"),
            )
            return
        finally:
            await adapter.close()

        file_paths = self._save_generated_images(task_id, generated)
        if not file_paths:
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message("生成完成，但未能保存图片文件"),
            )
            return

        chain, used_reply = self._build_result_chain(file_paths, reply_message_id)
        logger.info(
            f"[MicuDemo] 任务 {task_id} 发送图片，参考图来源={source}，"
            f"尝试引用={bool(reply_message_id)}，引用成功={used_reply}"
        )
        await self.context.send_message(unified_msg_origin, chain)
        charge_result = self.balance_manager.charge(
            unified_msg_origin,
            len(file_paths),
        )
        await self.context.send_message(
            unified_msg_origin,
            MessageChain().message(self.balance_manager.format_usage_message(charge_result)),
        )

    def _load_provider_configs(self) -> dict[str, MicuProviderConfig]:
        providers = self.config.get("api_providers", [])
        if not isinstance(providers, list):
            return {}

        configs: dict[str, MicuProviderConfig] = {}
        for provider in providers:
            if not isinstance(provider, dict):
                continue

            base_url = self._clean_base_url(str(provider.get("base_url") or "").strip())
            if not base_url:
                continue

            name = self._unique_name(
                str(provider.get("name") or "").strip() or "API",
                configs,
            )
            supports_text_to_image = self._capability_enabled(
                provider,
                "text_to_image_enabled",
                "文生图",
            )
            supports_image_to_image = self._capability_enabled(
                provider,
                "image_to_image_enabled",
                "图生图",
            )

            configs[name] = MicuProviderConfig(
                name=name,
                base_url=base_url,
                text_to_image_endpoint=self._endpoint(
                    provider.get("text_to_image_endpoint"),
                    "/v1/images/generations",
                ),
                image_to_image_endpoint=self._endpoint(
                    provider.get("image_to_image_endpoint"),
                    "/v1/images/edits",
                ),
                multi_reference_endpoint=self._endpoint(
                    provider.get("multi_reference_endpoint")
                    or provider.get("multi_image_endpoint"),
                    "/v1/chat/completions",
                ),
                supports_text_to_image=supports_text_to_image,
                supports_image_to_image=supports_image_to_image,
                proxy=str(provider.get("proxy") or "").strip() or None,
                max_reference_images=self._positive_int(
                    provider.get("max_reference_images"),
                    1,
                ),
                max_request_size_mb=self._positive_int(
                    provider.get("max_request_size_mb"),
                    20,
                ),
            )
        return configs

    def _runtime_config_for(self, umo: str) -> MicuRuntimeConfig | None:
        user = self.balance_manager.user_config(umo)
        if user is None:
            return None

        group = str(user.get("provider_group") or "").strip() or "API"
        provider = self.provider_configs.get(group)
        if provider is None and group == "API" and self.provider_configs:
            provider = next(iter(self.provider_configs.values()))
        api_key = str(user.get("api_key") or "").strip()
        if not api_key:
            api_key = self._legacy_provider_api_key()
        if provider is None or not api_key:
            return None
        return MicuRuntimeConfig(provider=provider, api_key=api_key)

    def _legacy_provider_api_key(self) -> str:
        providers = self.config.get("api_providers", [])
        if not isinstance(providers, list):
            return ""
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            api_keys = provider.get("api_keys", [])
            if not isinstance(api_keys, list):
                continue
            for key in api_keys:
                value = str(key).strip()
                if value:
                    return value
        return ""

    def _make_adapter(self, runtime: MicuRuntimeConfig) -> MicuGPTImage2Adapter:
        provider = runtime.provider
        return MicuGPTImage2Adapter(
            base_url=provider.base_url,
            api_key=runtime.api_key,
            text_to_image_endpoint=provider.text_to_image_endpoint,
            image_to_image_endpoint=provider.image_to_image_endpoint,
            multi_image_endpoint=provider.multi_reference_endpoint,
            proxy=provider.proxy,
        )

    async def _download_image(self, url: str) -> tuple[bytes, str] | None:
        try:
            path = self._local_path_from_image_ref(url)
            if path is not None:
                if not path.exists() or not path.is_file():
                    return None
                data = path.read_bytes()
            else:
                file_name = f"ref_{hashlib.md5(url.encode()).hexdigest()[:12]}"
                target = self.cache_dir / file_name
                downloaded = await download_image_by_url(url, path=str(target))
                if not downloaded:
                    return None
                data = Path(downloaded).read_bytes()

            if not data:
                return None
            return data, self._detect_mime_type(data)
        except Exception as exc:
            logger.warning(f"[MicuDemo] 提取参考图失败: {exc}")
            return None

    def _clean_base_url(self, base_url: str) -> str:
        url = base_url.rstrip("/")
        marker = "/v1"
        index = url.find(marker)
        if index != -1:
            url = url[:index]
        return url.rstrip("/")

    def _unique_name(self, base_name: str, existing: dict[str, Any]) -> str:
        name = base_name
        index = 1
        while name in existing:
            name = f"{base_name}{index}"
            index += 1
        return name

    def _capability_enabled(
        self,
        provider: dict[str, Any],
        bool_key: str,
        legacy_name: str,
    ) -> bool:
        if bool_key in provider:
            return bool(provider.get(bool_key))

        legacy = provider.get("capability_options")
        if isinstance(legacy, list):
            return legacy_name in {str(item).strip() for item in legacy}
        return True

    def _endpoint(self, value: Any, default: str) -> str:
        endpoint = str(value or "").strip()
        return endpoint or default

    def _positive_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, parsed)

    def _local_path_from_image_ref(self, image_ref: str) -> Path | None:
        value = image_ref.strip()
        if not value:
            return None

        plain_path = Path(value)
        if plain_path.exists():
            return plain_path

        parsed = urlparse(value)
        if parsed.scheme != "file":
            return None

        path_text = unquote(parsed.path or "")
        if parsed.netloc:
            if len(parsed.netloc) == 2 and parsed.netloc[1] == ":":
                path_text = f"{parsed.netloc}{path_text}"
            else:
                path_text = f"//{parsed.netloc}{path_text}"
        if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
            path_text = path_text[1:]
        return Path(path_text)

    def _get_event_text(self, event: AstrMessageEvent) -> str:
        candidates = [
            str(getattr(event, "message_str", "") or ""),
            str(getattr(getattr(event, "message_obj", None), "message_str", "") or ""),
            self._text_from_components(getattr(getattr(event, "message_obj", None), "message", None)),
            self._text_from_raw_message(getattr(getattr(event, "message_obj", None), "raw_message", None)),
        ]
        return max((item.strip() for item in candidates), key=len, default="")

    def _text_from_components(self, components: Any) -> str:
        if not isinstance(components, list):
            return ""

        parts: list[str] = []
        for component in components:
            if component.__class__.__name__ == "Plain":
                text = getattr(component, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)

    def _text_from_raw_message(self, raw_message: Any) -> str:
        if isinstance(raw_message, str):
            return raw_message
        if not isinstance(raw_message, dict):
            return ""

        message = raw_message.get("message") or raw_message.get("raw_message")
        if isinstance(message, str):
            return message
        if not isinstance(message, list):
            return ""

        parts: list[str] = []
        for item in message:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                data = item.get("data") or {}
                text = data.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    def _request_size_ok(
        self,
        images: list[tuple[bytes, str]],
        provider: MicuProviderConfig,
    ) -> bool:
        total = sum(len(data) for data, _ in images)
        return total <= provider.max_request_size_mb * 1024 * 1024

    def _save_generated_images(self, task_id: str, images: list[bytes]) -> list[str]:
        paths: list[str] = []
        for index, image in enumerate(images, start=1):
            digest = hashlib.md5(image).hexdigest()[:8]
            path = self.cache_dir / f"gen_{task_id}_{index}_{digest}.png"
            path.write_bytes(image)
            paths.append(str(path))
        return paths

    def _build_result_chain(
        self,
        file_paths: list[str],
        reply_message_id: str | None,
    ) -> tuple[MessageChain, bool]:
        chain = MessageChain()
        used_reply = self._try_prepend_reply(chain, reply_message_id)
        for file_path in file_paths:
            chain.file_image(file_path)
        return chain, used_reply

    def _try_prepend_reply(self, chain: MessageChain, message_id: str | None) -> bool:
        if not message_id:
            return False

        for kwargs in ({"id": message_id}, {"message_id": message_id}):
            try:
                reply = Comp.Reply(**kwargs)
            except Exception as exc:
                logger.debug(f"[MicuDemo] Reply 构造失败 {kwargs}: {exc}")
                continue

            for attr in ("chain", "message_chain", "messages"):
                items = getattr(chain, attr, None)
                if isinstance(items, list):
                    items.insert(0, reply)
                    return True

            logger.info("[MicuDemo] MessageChain 未暴露可插入 Reply 的列表，退化为普通发图")
            return False

        logger.info("[MicuDemo] 当前 Reply 组件构造方式不可用，退化为普通发图")
        return False

    def _get_message_id(self, event: AstrMessageEvent) -> str | None:
        message_obj = getattr(event, "message_obj", None)
        message_id = getattr(message_obj, "message_id", None)
        return str(message_id) if message_id else None

    def _detect_mime_type(self, data: bytes) -> str:
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if data.startswith(b"GIF"):
            return "image/gif"
        if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
            return "image/webp"
        return "image/png"

    def _extension_for_mime(self, mime: str) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/png": ".png",
        }.get(mime, ".png")
