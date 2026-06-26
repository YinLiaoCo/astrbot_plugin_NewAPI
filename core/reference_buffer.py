from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol

import astrbot.api.message_components as Comp


ImageData = tuple[bytes, str]
ImageDownloader = Callable[[str], Awaitable[ImageData | None]]
ReferenceSource = Literal["current", "reply", "buffer", "none"]

_BATCH_GENERATE_PATTERN = re.compile(r"^(\S+)\s+(.+)$", re.DOTALL)


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
    return images


async def extract_reply_message_images(
    event: MessageEventLike,
    download_image: ImageDownloader,
) -> list[ImageData]:
    images: list[ImageData] = []
    for component in _iter_message_components(event):
        if _is_reply_component(component):
            for sub_component in _reply_chain(component):
                if _is_image_component(sub_component):
                    await _append_downloaded_image(images, sub_component, download_image)
    return images


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
    return isinstance(component, Comp.Reply) or component.__class__.__name__ == "Reply"


def _reply_chain(component: object) -> list[object]:
    chain = getattr(component, "chain", None)
    if isinstance(chain, list):
        return chain
    message = getattr(component, "message", None)
    if isinstance(message, list):
        return message
    return []


def _component_image_url(component: object) -> str | None:
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
