#!/usr/bin/env python3
"""Generate eight latent document dimensions and inject them into index.html.

Usage:
  OPENAI_API_KEY=... python3 generate_embeddings.py index.html

The script uses only Python's standard library. It fetches every `.project` link,
extracts readable text, requests OpenAI embeddings, reduces them to eight PCA
components, and writes `data-embedding="..."` attributes back into the HTML.
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_BASE_URL = "https://meghabyte.github.io/blog/index.html"
PROJECT_TAG = re.compile(r'<a\b(?=[^>]*\bclass=["\'][^"\']*\bproject\b[^"\']*["\'])[^>]*>', re.I)
HREF = re.compile(r'\bhref=["\']([^"\']+)["\']', re.I)
OLD_EMBEDDING = re.compile(r'\s+data-embedding=["\'][^"\']*["\']', re.I)


class ReadableText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "nav", "footer", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "nav", "footer", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "meghabytes-embedding-builder/1.0 (+personal site generator)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read(2_000_000)
    if content_type not in {"text/html", "text/plain"}:
        raise ValueError(f"unsupported content type: {content_type}")
    decoded = body.decode(charset, errors="replace")
    if content_type == "text/plain":
        return " ".join(decoded.split())
    parser = ReadableText()
    parser.feed(decoded)
    return " ".join(parser.parts)


def request_embeddings(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    # Truncating by characters keeps unusually large pages comfortably below the
    # embedding model's input limit while retaining a substantial document sample.
    inputs = [text[:20_000] for text in texts]
    payload = json.dumps({"model": model, "input": inputs, "encoding_format": "float"}).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Embeddings API returned HTTP {error.code}: {detail}") from error
    return [item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def pca_reduce(rows: list[list[float]], components: int) -> list[list[float]]:
    """PCA via the small sample-space Gram matrix; efficient for ~14 documents."""
    count = len(rows)
    dimensions = len(rows[0])
    means = [sum(row[j] for row in rows) / count for j in range(dimensions)]
    centered = [[value - means[j] for j, value in enumerate(row)] for row in rows]
    gram = [[dot(centered[i], centered[j]) for j in range(count)] for i in range(count)]
    eigenvectors: list[list[float]] = []
    coordinates = [[0.0] * components for _ in range(count)]

    for component in range(min(components, count - 1)):
        vector = [math.sin((i + 1) * (component + 1) * 1.618) for i in range(count)]
        for _ in range(300):
            candidate = [sum(gram[i][j] * vector[j] for j in range(count)) for i in range(count)]
            for previous in eigenvectors:
                projection = dot(candidate, previous)
                candidate = [value - projection * previous[i] for i, value in enumerate(candidate)]
            norm = math.sqrt(dot(candidate, candidate))
            if norm < 1e-12:
                break
            candidate = [value / norm for value in candidate]
            if sum(abs(candidate[i] - vector[i]) for i in range(count)) < 1e-10:
                vector = candidate
                break
            vector = candidate
        eigenvalue = max(0.0, dot(vector, [sum(gram[i][j] * vector[j] for j in range(count)) for i in range(count)]))
        eigenvectors.append(vector)
        scale = math.sqrt(eigenvalue)
        for i in range(count):
            coordinates[i][component] = vector[i] * scale

    # Standardize every component so distances are comparable between pairs.
    for component in range(components):
        values = [row[component] for row in coordinates]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / max(1, count - 1)
        deviation = math.sqrt(variance) or 1.0
        for row in coordinates:
            row[component] = (row[component] - mean) / deviation
    return coordinates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="?", default="index.html", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is not set")

    source = args.html.read_text(encoding="utf-8")
    matches = list(PROJECT_TAG.finditer(source))
    if len(matches) < 3:
        parser.error("could not find project links in the HTML")

    documents: list[str] = []
    for index, match in enumerate(matches, start=1):
        tag = match.group(0)
        href_match = HREF.search(tag)
        if not href_match:
            raise RuntimeError(f"project {index} has no href")
        url = urllib.parse.urljoin(args.base_url, html_module.unescape(href_match.group(1)))
        print(f"[{index}/{len(matches)}] Fetching {url}", file=sys.stderr)
        try:
            text = fetch_text(url, args.timeout)
        except Exception as error:
            # A title still gives the document a deterministic fallback rather than
            # aborting a whole rebuild because one external page is temporarily down.
            title = re.sub(r"<[^>]+>", " ", source[match.end(): source.find("</a>", match.end())])
            text = f"{title} {url}"
            print(f"  warning: {error}; embedding title and URL instead", file=sys.stderr)
        documents.append(text)

    print(f"Requesting {len(documents)} embeddings with {args.model}…", file=sys.stderr)
    vectors = request_embeddings(documents, args.model, api_key)
    reduced = pca_reduce(vectors, 8)

    chunks: list[str] = []
    cursor = 0
    for match, values in zip(matches, reduced):
        chunks.append(source[cursor:match.start()])
        tag = OLD_EMBEDDING.sub("", match.group(0))
        encoded = ",".join(f"{value:.6f}" for value in values)
        chunks.append(tag[:-1] + f' data-embedding="{encoded}">')
        cursor = match.end()
    chunks.append(source[cursor:])

    temporary = args.html.with_suffix(args.html.suffix + ".tmp")
    temporary.write_text("".join(chunks), encoding="utf-8")
    temporary.replace(args.html)
    print(f"Updated {args.html} with eight PCA dimensions.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
