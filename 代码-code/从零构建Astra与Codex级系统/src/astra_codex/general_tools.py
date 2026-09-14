from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from .structured import ToolSpec
from .tools import ToolResult


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            stripped = " ".join(data.split())
            if stripped:
                self.parts.append(stripped)


@dataclass(frozen=True, slots=True)
class HTTPPolicy:
    timeout: float = 15.0
    max_bytes: int = 2_000_000
    user_agent: str = "astra-codex-from-scratch/0.1"


class HTTPBrowserTool:
    """Very small HTTP GET browser for the general-agent path.

    It intentionally does not pretend to be browser automation.  JavaScript,
    cookies, authentication, forms, screenshots, and computer use belong to a
    later browser/computer adapter with stronger isolation.
    """

    spec = ToolSpec(
        name="http_get",
        description="Fetch a public http(s) URL and return readable text.",
        parameters={
            "type": "object",
            "required": ["url"],
            "additionalProperties": False,
            "properties": {"url": {"type": "string"}},
        },
    )

    def __init__(self, policy: HTTPPolicy | None = None) -> None:
        self.policy = policy or HTTPPolicy()

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        url = arguments["url"]
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError("only http(s) URLs are allowed")
        request = urllib.request.Request(url, headers={"User-Agent": self.policy.user_agent})
        with urllib.request.urlopen(request, timeout=self.policy.timeout) as response:
            raw = response.read(self.policy.max_bytes + 1)
            truncated = len(raw) > self.policy.max_bytes
            raw = raw[: self.policy.max_bytes]
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"

        text = raw.decode(charset, errors="replace")
        if content_type == "text/html":
            parser = _TextExtractor()
            parser.feed(text)
            text = "\n".join(parser.parts)
        return ToolResult(True, text, {"content_type": content_type, "truncated": truncated})
