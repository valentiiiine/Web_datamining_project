from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura


LOGGER = logging.getLogger("wikipedia_crawler")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TennisGrandSlamKG/1.0; "
        "+https://example.org/academic-project)"
    )
}

NOISE_PATTERNS = [
    r"\bcopyright\b",
    r"\bprivacy policy\b",
    r"\bterms of use\b",
    r"\bsubscribe\b",
    r"\bcookie\b",
    r"\ball rights reserved\b",
]


@dataclass
class CrawlRecord:
    url: str
    title: str
    text: str
    word_count: int

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "word_count": self.word_count,
        }


def configure_logging(level: int = logging.INFO) -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def build_robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def load_robot_parser(robots_url: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = requests.get(robots_url, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        lines = response.text.splitlines()
        if any(line.lower().startswith("user-agent:") for line in lines):
            parser.parse(lines)
        else:
            LOGGER.info(
                "robots.txt at %s does not expose standard user-agent rules; allowing crawl with caution",
                robots_url,
            )
            parser.allow_all = True
    except requests.RequestException as exc:
        LOGGER.warning("Could not read robots.txt at %s: %s", robots_url, exc)
        parser.allow_all = True
    return parser


def can_fetch_url(
    url: str,
    robot_parsers: dict[str, RobotFileParser],
    user_agent: str = DEFAULT_HEADERS["User-Agent"],
) -> bool:
    # Ethical crawling requires checking robots.txt before requesting a page.
    robots_url = build_robots_url(url)
    parser = robot_parsers.get(robots_url)
    if parser is None:
        parser = load_robot_parser(robots_url)
        robot_parsers[robots_url] = parser
    return parser.can_fetch(user_agent, url)


def fetch_html(url: str, timeout: int = 30, delay_seconds: float = 1.0) -> str | None:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        time.sleep(delay_seconds)
        return response.text
    except requests.RequestException as exc:
        LOGGER.warning("Request failed for %s: %s", url, exc)
        return None


def extract_title_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return slug.replace("_", " ")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_light_boilerplate(text: str) -> str:
    cleaned = text
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def clean_main_text(html: str) -> str:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        no_fallback=False,
    )
    if not extracted:
        return ""
    return remove_light_boilerplate(extracted)


def count_words(text: str) -> int:
    return len(text.split())


def is_useful_page(text: str, min_words: int = 500) -> bool:
    return count_words(text) >= min_words


def crawl_urls(urls: Iterable[str], min_words: int = 500) -> tuple[list[CrawlRecord], dict]:
    useful_records: list[CrawlRecord] = []
    robot_parsers: dict[str, RobotFileParser] = {}
    stats = {
        "fetched": 0,
        "useful": 0,
        "missing_text": 0,
        "filtered_short": 0,
        "request_failed": 0,
        "blocked_by_robots": 0,
    }

    for url in urls:
        if not can_fetch_url(url, robot_parsers):
            LOGGER.info("Skipping %s because robots.txt disallows it", url)
            stats["blocked_by_robots"] += 1
            continue

        LOGGER.info("Fetching %s", url)
        html = fetch_html(url)
        if html is None:
            stats["request_failed"] += 1
            continue

        stats["fetched"] += 1
        text = clean_main_text(html)
        if not text:
            stats["missing_text"] += 1
            continue

        if not is_useful_page(text, min_words=min_words):
            stats["filtered_short"] += 1
            continue

        record = CrawlRecord(
            url=url,
            title=extract_title_from_url(url),
            text=text,
            word_count=count_words(text),
        )
        useful_records.append(record)
        stats["useful"] += 1

    return useful_records, stats


def save_jsonl(records: Iterable[CrawlRecord], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows
