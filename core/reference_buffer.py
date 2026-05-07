from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol

import astrbot.api.message_components as Comp


ImageData = tuple[bytes, str]
ImageDownloader = Callable[[str], Awaitable[ImageData | None]]
ReferenceSource = Literal["current", "reply", "buffer", "none"]

_SIZE_PATTERN = re.compile(r"(?:^|\s)尺寸\s*[:：]\s*(\S+)", re.IGNORECASE)


class MessageEventLike(Protocol):
    unified_msg_origin: str
    message_str: str
    message_obj: object


@dataclass(frozen=True)
class GenerateCommand:
    prompt: str
    size: str = "1024x1024"
    requested_size: str = "1k"


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
    text = _strip_command_prefix(message, "生图").strip()
    match = _SIZE_PATTERN.search(text)
    if not match:
        return ParseResult(None, "格式错误，请使用 /生图 尺寸: 1k 提示词")

    requested_size = "1k"
    requested_size = match.group(1).lower()
    prompt = (text[: match.start()] + text[match.end() :]).strip()

    if requested_size != "1k":
        if requested_size in {"2k", "4k"} and has_reference_images:
            return ParseResult(None, "图生图仅支持 1K")
        return ParseResult(None, "Demo 只测 1K")

    if not prompt:
        return ParseResult(None, "请提供提示词")

    return ParseResult(GenerateCommand(prompt=prompt, requested_size=requested_size))


async def extract_current_message_images(
    event: MessageEventLike,
    download_image: ImageDownloader,
) -> list[ImageData]:
    images: list[ImageData] = []
    for component in _iter_message_components(event):
        if isinstance(component, Comp.Image):
            await _append_downloaded_image(images, component, download_image)
    return images


async def extract_reply_message_images(
    event: MessageEventLike,
    download_image: ImageDownloader,
) -> list[ImageData]:
    images: list[ImageData] = []
    for component in _iter_message_components(event):
        if isinstance(component, Comp.Reply) and component.chain:
            for sub_component in component.chain:
                if isinstance(sub_component, Comp.Image):
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


async def _append_downloaded_image(
    images: list[ImageData],
    component: Comp.Image,
    download_image: ImageDownloader,
) -> None:
    url = component.url or component.file
    if not url:
        return

    data = await download_image(url)
    if data:
        images.append(data)
