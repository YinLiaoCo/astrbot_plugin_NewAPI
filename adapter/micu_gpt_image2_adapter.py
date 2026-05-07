from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    import aiohttp


MODEL = "gpt-image-2-pro"
SIZE = "1024x1024"
RESPONSE_FORMAT = "b64_json"


@dataclass(frozen=True)
class MicuImageInput:
    data: bytes
    filename: str = "image.png"
    content_type: str = "image/png"


class MicuGPTImage2Adapter:
    """Demo adapter for Micu gpt-image-2-pro image API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        proxy: str | None = None,
        timeout: int = 180,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.proxy = proxy
        self.timeout = timeout
        self._session = session
        self._owns_session = session is None

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def text_to_image(self, prompt: str) -> list[bytes]:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "n": 1,
            "size": SIZE,
            "response_format": RESPONSE_FORMAT,
        }
        return await self._post_json("/v1/images/generations", payload)

    async def image_to_image(
        self,
        prompt: str,
        images: list[MicuImageInput] | list[bytes],
    ) -> list[bytes]:
        aiohttp = self._aiohttp()
        form = aiohttp.FormData()
        form.add_field("model", MODEL)
        form.add_field("prompt", prompt)
        form.add_field("n", "1")
        form.add_field("size", SIZE)
        form.add_field("response_format", RESPONSE_FORMAT)

        for index, image in enumerate(images):
            image_input = self._normalize_image_input(image, index)
            form.add_field(
                "image[]",
                image_input.data,
                filename=image_input.filename,
                content_type=image_input.content_type,
            )

        return await self._post_form("/v1/images/edits", form)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> list[bytes]:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        aiohttp = self._aiohttp()
        async with self._session_or_new().post(
            f"{self.base_url}{path}",
            headers=headers,
            json=payload,
            proxy=self.proxy,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            return await self._handle_response(response)

    async def _post_form(self, path: str, form: aiohttp.FormData) -> list[bytes]:
        aiohttp = self._aiohttp()
        async with self._session_or_new().post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            data=form,
            proxy=self.proxy,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            return await self._handle_response(response)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> list[bytes]:
        raw_text = await response.text()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(
                f"Micu API error {response.status}: {self._response_preview(raw_text)}"
            )

        try:
            images = await self.parse_response(raw_text)
        except Exception as exc:
            raise RuntimeError(
                "Micu API response image parsing failed. "
                f"Reason: {exc}. Raw response preview: {self._response_preview(raw_text)}"
            ) from exc
        if not images:
            raise RuntimeError(
                "Micu API response did not contain an image. "
                f"Raw response preview: {self._response_preview(raw_text)}"
            )
        return images

    async def parse_response(self, raw_text: str) -> list[bytes]:
        images: list[bytes] = []
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            data = None

        if data is not None:
            images.extend(await self._extract_from_json(data))

        if not images:
            images.extend(await self._extract_from_text(raw_text))

        return images

    async def _extract_from_json(self, value: Any) -> list[bytes]:
        images: list[bytes] = []

        for item in self._iter_data_items(value):
            if not isinstance(item, dict):
                continue

            image_value = item.get("b64_json")
            if isinstance(image_value, str):
                image = self._decode_base64_like(image_value)
                if image:
                    images.append(image)
                    continue

            url = item.get("url")
            if isinstance(url, str):
                images.append(await self._load_image_reference(url))

        text_values = self._iter_text_values(value)
        for text in text_values:
            images.extend(await self._extract_from_text(text))

        return images

    async def _extract_from_text(self, text: str) -> list[bytes]:
        images: list[bytes] = []
        seen: set[str] = set()

        for value in self._find_image_references(text):
            if value in seen:
                continue
            seen.add(value)

            if value.startswith(("http://", "https://")):
                images.append(await self._download_image(value))
                continue

            image = self._decode_base64_like(value)
            if image:
                images.append(image)

        return images

    def _find_image_references(self, text: str) -> list[str]:
        results: list[str] = []
        results.extend(re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', text, re.I))
        results.extend(re.findall(r"!\[[^\]]*]\(([^)\s]+)\)", text, re.I))
        results.extend(
            re.findall(
                r"data:image/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
                text,
                re.I,
            )
        )
        results.extend(
            re.findall(
                r"https?://[^\s\"'<>)]+\.(?:png|jpe?g|gif|webp)(?:\?[^\s\"'<>)]*)?",
                text,
                re.I,
            )
        )

        if not results:
            stripped = re.sub(r"\s+", "", text)
            if stripped and self._decode_base64_like(stripped):
                results.append(stripped)

        if not results:
            results.extend(re.findall(r"(?:[A-Za-z0-9+/]{4}){50,}(?:==|=)?", text))

        return results

    def _decode_base64_like(self, value: str) -> bytes | None:
        candidate = value.strip()
        if candidate.startswith("data:"):
            _, _, candidate = candidate.partition(",")
        candidate = re.sub(r"\s+", "", candidate)
        if not candidate:
            return None

        try:
            return base64.b64decode(candidate, validate=True)
        except (binascii.Error, ValueError):
            return None

    async def _load_image_reference(self, value: str) -> bytes:
        if value.startswith("data:"):
            image = self._decode_base64_like(value)
            if image:
                return image
            raise ValueError("Invalid data URL image in Micu API response")

        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            return await self._download_image(value)

        image = self._decode_base64_like(value)
        if image:
            return image
        raise ValueError(f"Unsupported image reference: {value[:80]}")

    async def _download_image(self, url: str) -> bytes:
        aiohttp = self._aiohttp()
        async with self._session_or_new().get(
            url,
            proxy=self.proxy,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"Image download failed {response.status}: {url}")
            return await response.read()

    def _iter_data_items(self, value: Any) -> list[Any]:
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            return value["data"]
        return []

    def _iter_text_values(self, value: Any) -> list[str]:
        values: list[str] = []
        if isinstance(value, dict):
            for key in ("content", "text", "message"):
                item = value.get(key)
                if isinstance(item, str):
                    values.append(item)
            for item in value.values():
                values.extend(self._iter_text_values(item))
        elif isinstance(value, list):
            for item in value:
                values.extend(self._iter_text_values(item))
        return values

    def _normalize_image_input(
        self, image: MicuImageInput | bytes, index: int
    ) -> MicuImageInput:
        if isinstance(image, MicuImageInput):
            return image
        return MicuImageInput(data=image, filename=f"image_{index}.png")

    def _session_or_new(self) -> aiohttp.ClientSession:
        aiohttp = self._aiohttp()
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _response_preview(self, raw_text: str) -> str:
        return raw_text[:300]

    def _aiohttp(self) -> Any:
        try:
            import aiohttp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "aiohttp is required to call the Micu image API"
            ) from exc
        return aiohttp
