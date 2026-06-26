from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol

import astrbot.api.message_components as Comp


ImageData = tuple[bytes, str]
ImageDownloader = Callable[[str], Awaitable[ImageData | None]]
ReferenceSource = Literal["current", "reply", "buffer", "none"]

_BATCH_GENERATE_PATTERN = re.compile(r"^(\S+)\s+(.+)$", re.DOTALL)
_CQ_IMAGE_PATTERN = re.compile(r"\[CQ:image,([^\]]+)\]")


class MessageEventLike(Protocol):
    unified_msg_origin: str
    message_str: str
    message_obj: object


@dataclass(frozen=True)
class GenerateCommand:
    prompt: str
    size: str = "1024x1024"
    n: int = 1
    command_name: str = "生图"
    warning: str | None = None


@dataclass(frozen=True)
class ParseResult:
    command: GenerateCommand | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.command is not None and self.error is None


@dataclass(frozen=True)
class ReferenceSelection:
    images: list[ImageData]
    source: ReferenceSource

    @property
    def has_images(self) -> bool:
        return bool(self.images)


class ReferenceBuffer:
    """In-memory Demo reference image buffer, isolated by unified_msg_origin."""

    def __init__(self, max_images_per_session: int = 4):
        self._max_images_per_session = max_images_per_session
        self._buffers: dict[str, list[ImageData]] = {}

    def add(self, unified_msg_origin: str, images: list[ImageData]) -> None:
        if not images:
            return

        bucket = self._buffers.setdefault(unified_msg_origin, [])
        bucket.extend(images)
        if len(bucket) > self._max_images_per_session:
            del bucket[: len(bucket) - self._max_images_per_session]

    def get(self, unified_msg_origin: str) -> list[ImageData]:
        return list(self._buffers.get(unified_msg_origin, []))

    def clear(self, unified_msg_origin: str) -> None:
        self._buffers.pop(unified_msg_origin, None)

    def clear_after_task_created(self, unified_msg_origin: str) -> None:
        self.clear(unified_msg_origin)

    async def cache_plain_message_images(
        self,
        event: MessageEventLike,
        download_image: ImageDownloader,
    ) -> int:
        images = await extract_current_message_images(event, download_image)
        self.add(event.unified_msg_origin, images)
        return len(images)

    async def select_for_command(
        self,
        event: MessageEventLike,
        download_image: ImageDownloader,
    ) -> ReferenceSelection:
        current_images = await extract_current_message_images(event, download_image)
        if current_images:
            return ReferenceSelection(current_images, "current")

        reply_images = await extract_reply_message_images(event, download_image)
        if reply_images:
            return ReferenceSelection(reply_images, "reply")

        buffered_images = self.get(event.unified_msg_origin)
        if buffered_images:
            return ReferenceSelection(buffered_images, "buffer")

        return ReferenceSelection([], "none")


def parse_generate_command(message: str, has_reference_images: bool = False) -> ParseResult:
    command_name, text = _split_command(message)
    if command_name == "批量生成":
        return _parse_batch_generate_text(text, has_reference_images)
    return _parse_single_generate_text(text, has_reference_images)


def parse_prompt_message(message: str, has_reference_images: bool = False) -> ParseResult:
    command_name, text = _split_command(message)
    if command_name == "批量生成":
        return _parse_batch_generate_text(text, has_reference_images)

    text = text.strip()
    if not text:
        return ParseResult(None, "请提供提示词")

    return ParseResult(GenerateCommand(prompt=text))


def _parse_single_generate_text(text: str, has_reference_images: bool) -> ParseResult:
    prompt = text.strip()
    if not prompt:
        return ParseResult(None, "请提供提示词")

    return ParseResult(GenerateCommand(prompt=prompt))


def _parse_batch_generate_text(text: str, has_reference_images: bool) -> ParseResult:
    match = _BATCH_GENERATE_PATTERN.match(text.strip())
    if not match:
        return ParseResult(None, "格式错误，请使用 /批量生成 3 提示词")

    try:
        requested_count = int(match.group(1))
    except ValueError:
        return ParseResult(None, "数量格式错误，请使用 /批量生成 3 提示词")

    if requested_count < 1:
        return ParseResult(None, "数量必须大于 0")

    prompt = match.group(2).strip()

    if not prompt:
        return ParseResult(None, "请提供提示词")

    n = min(requested_count, 4)
    warning = None
    if has_reference_images and n > 1:
        n = 1
        warning = "参考图生成仅支持单张，本次将按 1 张执行"

    return ParseResult(
        GenerateCommand(
            prompt=prompt,
            n=n,
            command_name="批量生成",
            warning=warning,
        )
    )


def _split_command(message: str) -> tuple[str, str]:
    text = message.strip()
    for command in ("批量生成", "生图"):
        for prefix in (f"/{command}", f"／{command}", command):
            if text.startswith(prefix):
                return command, text[len(prefix) :].strip()
    return "生图", text


async def extract_current_message_images(
    event: MessageEventLike,
    download_image: ImageDownloader,
) -> list[ImageData]:
    images: list[ImageData] = []
    for component in _iter_message_components(event):
        if _is_image_component(component):
            await _append_downloaded_image(images, component, download_image)
    return _dedupe_images(images)


async def extract_reply_message_images(
    event: MessageEventLike,
    download_image: ImageDownloader,
) -> list[ImageData]:
    images: list[ImageData] = []
    for component in _iter_message_components(event):
        if _is_reply_component(component):
            await _append_images_from_components(
                images,
                _reply_chain(component),
                download_image,
            )
            if not images:
                reply_id = _reply_id(component)
                if reply_id:
                    await _append_images_from_reply_id(
                        images,
                        event,
                        reply_id,
                        download_image,
                    )
    if images:
        return images

    for reply_id in _raw_reply_ids(event):
        await _append_images_from_reply_id(images, event, reply_id, download_image)
        if images:
            return images

    await _append_images_from_components(
        images,
        _raw_reply_image_components(event),
        download_image,
    )
    return _dedupe_images(images)


def _strip_command_prefix(message: str, command: str) -> str:
    text = message.strip()
    for prefix in (f"/{command}", command):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _iter_message_components(event: MessageEventLike) -> list[object]:
    message_obj = getattr(event, "message_obj", None)
    message = getattr(message_obj, "message", None)
    return list(message or [])


def _is_image_component(component: object) -> bool:
    if isinstance(component, Comp.Image):
        return True
    if component.__class__.__name__ == "Image":
        return True
    return _component_image_url(component) is not None


def _is_reply_component(component: object) -> bool:
    if isinstance(component, dict):
        type_name = str(component.get("type") or component.get("_type") or "").lower()
        return type_name == "reply"
    return isinstance(component, Comp.Reply) or component.__class__.__name__ == "Reply"


def _reply_chain(component: object) -> list[object]:
    if isinstance(component, dict):
        values: list[object] = []
        data = component.get("data")
        for source in (component, data if isinstance(data, dict) else {}):
            for key in ("chain", "message", "raw_message", "content"):
                value = source.get(key)
                if isinstance(value, list):
                    values.extend(value)
                elif isinstance(value, dict):
                    values.append(value)
        return values

    chain = getattr(component, "chain", None)
    if isinstance(chain, list):
        return chain
    message = getattr(component, "message", None)
    if isinstance(message, list):
        return message
    return []


def _reply_id(component: object) -> str | None:
    for attr in ("id", "message_id", "seq"):
        value = getattr(component, attr, None)
        if value not in (None, ""):
            return str(value)

    if isinstance(component, dict):
        data = component.get("data")
        for source in (component, data if isinstance(data, dict) else {}):
            for key in ("id", "message_id", "seq"):
                value = source.get(key)
                if value not in (None, ""):
                    return str(value)
    return None


async def _append_images_from_reply_id(
    images: list[ImageData],
    event: MessageEventLike,
    reply_id: str,
    download_image: ImageDownloader,
) -> None:
    bot = getattr(event, "bot", None)
    call_action = getattr(bot, "call_action", None)
    if not callable(call_action):
        return

    try:
        message_id = int(reply_id)
    except (TypeError, ValueError):
        return

    try:
        payload = await call_action(
            action="get_msg",
            message_id=message_id,
            **_routing_params(event),
        )
    except Exception:
        return

    await _append_images_from_components(
        images,
        _raw_message_components(payload),
        download_image,
    )


def _routing_params(event: MessageEventLike) -> dict[str, object]:
    message_obj = getattr(event, "message_obj", None)
    self_id = getattr(message_obj, "self_id", None)
    raw_message = getattr(message_obj, "raw_message", None)
    if not self_id and isinstance(raw_message, dict):
        self_id = raw_message.get("self_id")
    if self_id:
        return {"self_id": self_id}
    return {}


def _raw_reply_ids(event: MessageEventLike) -> list[str]:
    ids: list[str] = []
    for component in _raw_message_components(_raw_event_message(event)):
        if _is_reply_component(component):
            reply_id = _reply_id(component)
            if reply_id and reply_id not in ids:
                ids.append(reply_id)
    return ids


def _raw_reply_image_components(event: MessageEventLike) -> list[object]:
    raw = _raw_event_message(event)
    components: list[object] = []
    for value in _iter_raw_reply_values(raw):
        components.extend(_raw_message_components(value))
    return components


def _raw_event_message(event: MessageEventLike) -> object:
    message_obj = getattr(event, "message_obj", None)
    return getattr(message_obj, "raw_message", None)


def _iter_raw_reply_values(value: object, depth: int = 0) -> list[object]:
    if depth > 6:
        return []

    values: list[object] = []
    if isinstance(value, dict):
        for key in ("reply", "quote", "source", "quoted_message"):
            nested = value.get(key)
            if nested is not None:
                values.append(nested)
                values.extend(_iter_raw_reply_values(nested, depth + 1))

        data = value.get("data")
        if isinstance(data, dict):
            values.extend(_iter_raw_reply_values(data, depth + 1))

        for key in ("message", "raw_message"):
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                values.extend(_iter_raw_reply_values(nested, depth + 1))
    elif isinstance(value, list):
        for item in value:
            values.extend(_iter_raw_reply_values(item, depth + 1))
    return values


def _raw_message_components(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return _cq_image_components(value)
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, list):
            return message
        if isinstance(message, str):
            return _cq_image_components(message)

        raw_message = value.get("raw_message")
        if isinstance(raw_message, list):
            return raw_message
        if isinstance(raw_message, str):
            return _cq_image_components(raw_message)

        return [value]
    return [value]


def _cq_image_components(text: str) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for match in _CQ_IMAGE_PATTERN.finditer(text):
        data: dict[str, str] = {}
        for part in match.group(1).split(","):
            key, separator, value = part.partition("=")
            if separator:
                data[key.strip()] = html.unescape(value.strip())
        if data:
            components.append({"type": "image", "data": data})
    return components


async def _append_images_from_components(
    images: list[ImageData],
    components: list[object],
    download_image: ImageDownloader,
) -> None:
    for component in _iter_nested_components(components):
        if _is_image_component(component):
            await _append_downloaded_image(images, component, download_image)


def _dedupe_images(images: list[ImageData]) -> list[ImageData]:
    deduped: list[ImageData] = []
    seen: set[bytes] = set()
    for data, mime in images:
        if data in seen:
            continue
        seen.add(data)
        deduped.append((data, mime))
    return deduped


def _iter_nested_components(value: object, depth: int = 0) -> list[object]:
    if value is None or depth > 6:
        return []

    components: list[object] = []
    if isinstance(value, list):
        for item in value:
            components.extend(_iter_nested_components(item, depth + 1))
        return components

    components.append(value)
    nested_values: list[object] = []
    if isinstance(value, dict):
        data = value.get("data")
        for source in (value, data if isinstance(data, dict) else {}):
            for key in (
                "chain",
                "message",
                "raw_message",
                "content",
                "messages",
                "nodes",
            ):
                nested = source.get(key)
                if nested is not None:
                    nested_values.append(nested)
    else:
        for attr in ("chain", "message", "content", "messages", "nodes"):
            nested = getattr(value, attr, None)
            if nested is not None:
                nested_values.append(nested)

    for nested in nested_values:
        components.extend(_iter_nested_components(nested, depth + 1))
    return components


def _component_image_url(component: object) -> str | None:
    if isinstance(component, dict):
        type_name = str(component.get("type") or component.get("_type") or "").lower()
        data = component.get("data")
        source = data if isinstance(data, dict) else component
        if type_name and type_name not in {"image", "mface"}:
            return None
        for key in ("url", "file", "path", "file_path"):
            value = source.get(key)
            if value:
                return str(value)
        return None

    for attr in ("url", "file", "path"):
        value = getattr(component, attr, None)
        if value:
            return str(value)

    data = getattr(component, "data", None)
    if isinstance(data, dict):
        for key in ("url", "file", "path"):
            value = data.get(key)
            if value:
                return str(value)

    return None


async def _append_downloaded_image(
    images: list[ImageData],
    component: object,
    download_image: ImageDownloader,
) -> None:
    url = _component_image_url(component)
    if not url:
        return

    data = await download_image(url)
    if data:
        images.append(data)
        return

    converter = getattr(component, "convert_to_file_path", None)
    if not callable(converter):
        return

    try:
        converted = converter()
        if hasattr(converted, "__await__"):
            converted = await converted
    except Exception:
        return

    if converted:
        data = await download_image(str(converted))
        if data:
            images.append(data)
