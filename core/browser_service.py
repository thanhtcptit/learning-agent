from __future__ import annotations

import json
import logging
import re
import webbrowser
from urllib.parse import quote_plus

import httpx

log = logging.getLogger(__name__)

_ACTION_PATTERN = re.compile(
    r"<<<BROWSER_ACTION>>>\s*(\{.*?\})\s*<<<END_ACTION>>>",
    re.DOTALL,
)

_ACTION_CLEANUP_PATTERN = re.compile(
    r"<<<BROWSER_ACTION>>>.*?<<<END_ACTION>>>",
    re.DOTALL,
)

SUPPORTED_SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia": "https://en.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "linkedin": "https://www.linkedin.com",
    "gmail": "https://mail.google.com",
    "maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
    "chatgpt": "https://chatgpt.com",
}


def _build_search_url(site: str, query: str) -> str:
    site_lower = site.strip().lower()
    encoded_query = quote_plus(query)

    if site_lower == "youtube":
        return f"https://www.youtube.com/results?search_query={encoded_query}"
    if site_lower == "google":
        return f"https://www.google.com/search?q={encoded_query}"
    if site_lower == "github":
        return f"https://github.com/search?q={encoded_query}"
    if site_lower == "stackoverflow":
        return f"https://stackoverflow.com/search?q={encoded_query}"
    if site_lower == "wikipedia":
        return f"https://en.wikipedia.org/w/index.php?search={encoded_query}"
    if site_lower == "reddit":
        return f"https://www.reddit.com/search/?q={encoded_query}"
    if site_lower == "maps":
        return f"https://www.google.com/maps/search/{encoded_query}"

    return f"https://www.google.com/search?q={encoded_query}+site:{site_lower}"


def _is_safe_url(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")


_YT_VIDEO_ID_PATTERN = re.compile(r'/watch\?v=([a-zA-Z0-9_-]{11})')


def _fetch_first_youtube_video_url(query: str) -> str | None:
    """Fetch YouTube search results and extract the first video URL."""
    search_url = _build_search_url("youtube", query)
    try:
        resp = httpx.get(
            search_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("Failed to fetch YouTube search results: %s", exc)
        return None

    match = _YT_VIDEO_ID_PATTERN.search(resp.text)
    if not match:
        log.warning("No video ID found in YouTube search results")
        return None

    video_id = match.group(1)
    return f"https://www.youtube.com/watch?v={video_id}"


def execute_browser_action(action: dict) -> str:
    """Execute a single browser action dict and return a status message."""
    action_type = action.get("action", "").strip().lower()

    if action_type == "open_url":
        url = action.get("url", "").strip()
        if not url:
            return "No URL provided."
        if not _is_safe_url(url):
            return f"Blocked unsafe URL scheme: {url}"
        webbrowser.open(url)
        log.info("Opened URL: %s", url)
        return f"Opened {url}"

    if action_type == "search":
        site = action.get("site", "google").strip()
        query = action.get("query", "").strip()
        if not query:
            return "No search query provided."
        url = _build_search_url(site, query)
        webbrowser.open(url)
        log.info("Searched %s for: %s", site, query)
        return f"Searched {site} for '{query}'"

    if action_type == "search_and_play":
        query = action.get("query", "").strip()
        if not query:
            return "No search query provided."
        video_url = _fetch_first_youtube_video_url(query)
        if video_url:
            webbrowser.open(video_url)
            log.info("Playing first YouTube result for: %s -> %s", query, video_url)
            return f"Playing first YouTube result for '{query}'"
        fallback_url = _build_search_url("youtube", query)
        webbrowser.open(fallback_url)
        log.info("Could not find video, falling back to search: %s", query)
        return f"Searched YouTube for '{query}' (could not auto-play)"

    if action_type == "open_site":
        site = action.get("site", "").strip().lower()
        url = SUPPORTED_SITES.get(site, "")
        if not url:
            url = f"https://www.{site}.com" if site else ""
        if not url:
            return "No site specified."
        if not _is_safe_url(url):
            return f"Blocked unsafe URL: {url}"
        webbrowser.open(url)
        log.info("Opened site: %s", url)
        return f"Opened {site}"

    return f"Unknown browser action: {action_type}"


def extract_and_execute_actions(text: str) -> tuple[str, list[str]]:
    """Parse action tags from LLM output, execute them, return cleaned text and results."""
    results: list[str] = []

    has_tags = _ACTION_CLEANUP_PATTERN.search(text) is not None
    if not has_tags:
        return text, results

    for match in _ACTION_PATTERN.finditer(text):
        raw_json = match.group(1)
        try:
            action = json.loads(raw_json)
        except json.JSONDecodeError:
            log.warning("Malformed browser action JSON: %s", raw_json)
            continue
        result = execute_browser_action(action)
        results.append(result)

    # Always strip action tags from the displayed text
    cleaned = _ACTION_CLEANUP_PATTERN.sub("", text).strip()
    return cleaned, results
