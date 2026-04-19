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


def _build_youtube_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


_YT_VIDEO_ID_PATTERN = re.compile(r'/watch\?v=([a-zA-Z0-9_-]{11})')


def _fetch_first_youtube_video_url(query: str) -> str | None:
    """Fetch YouTube search results and extract the first video URL."""
    search_url = _build_youtube_search_url(query)
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

    if action_type == "search_and_play":
        query = action.get("query", "").strip()
        if not query:
            return "No search query provided."
        video_url = _fetch_first_youtube_video_url(query)
        if video_url:
            webbrowser.open(video_url)
            log.info("Playing first YouTube result for: %s -> %s", query, video_url)
            return f"Playing first YouTube result for '{query}'"
        fallback_url = _build_youtube_search_url(query)
        webbrowser.open(fallback_url)
        log.info("Could not find video, falling back to search: %s", query)
        return f"Searched YouTube for '{query}' (could not auto-play)"

    return f"Unknown browser action: {action_type}"


def extract_and_execute_actions(text: str) -> tuple[str, list[str], bool]:
    """Parse action tags from LLM output, execute them, and return cleanup state."""
    results: list[str] = []
    suppress_tts = False

    has_tags = _ACTION_CLEANUP_PATTERN.search(text) is not None
    if not has_tags:
        return text, results, suppress_tts

    for match in _ACTION_PATTERN.finditer(text):
        raw_json = match.group(1)
        try:
            action = json.loads(raw_json)
        except json.JSONDecodeError:
            log.warning("Malformed browser action JSON: %s", raw_json)
            continue
        action_type = action.get("action", "").strip().lower()
        if action_type == "search_and_play":
            suppress_tts = True
        result = execute_browser_action(action)
        results.append(result)

    # Always strip action tags from the displayed text
    cleaned = _ACTION_CLEANUP_PATTERN.sub("", text).strip()
    return cleaned, results, suppress_tts
