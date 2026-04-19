from __future__ import annotations

from unittest.mock import patch

from core.browser_service import (
    _build_search_url,
    _is_safe_url,
    execute_browser_action,
    extract_and_execute_actions,
    SUPPORTED_SITES,
)


class TestBuildSearchUrl:
    def test_youtube(self):
        url = _build_search_url("youtube", "piano music")
        assert url == "https://www.youtube.com/results?search_query=piano+music"

    def test_google(self):
        url = _build_search_url("google", "Python tutorials")
        assert url == "https://www.google.com/search?q=Python+tutorials"

    def test_github(self):
        url = _build_search_url("github", "fastapi")
        assert url == "https://github.com/search?q=fastapi"

    def test_stackoverflow(self):
        url = _build_search_url("stackoverflow", "how to sort a list")
        assert url == "https://stackoverflow.com/search?q=how+to+sort+a+list"

    def test_wikipedia(self):
        url = _build_search_url("wikipedia", "quantum computing")
        assert url == "https://en.wikipedia.org/w/index.php?search=quantum+computing"

    def test_reddit(self):
        url = _build_search_url("reddit", "best laptops")
        assert url == "https://www.reddit.com/search/?q=best+laptops"

    def test_maps(self):
        url = _build_search_url("maps", "coffee shops")
        assert url == "https://www.google.com/maps/search/coffee+shops"

    def test_unknown_site_falls_back_to_google(self):
        url = _build_search_url("bing", "test query")
        assert url == "https://www.google.com/search?q=test+query+site:bing"

    def test_case_insensitive(self):
        url = _build_search_url("YouTube", "cats")
        assert url == "https://www.youtube.com/results?search_query=cats"


class TestIsSafeUrl:
    def test_https(self):
        assert _is_safe_url("https://example.com") is True

    def test_http(self):
        assert _is_safe_url("http://example.com") is True

    def test_file_scheme_blocked(self):
        assert _is_safe_url("file:///etc/passwd") is False

    def test_javascript_blocked(self):
        assert _is_safe_url("javascript:alert(1)") is False

    def test_empty_string(self):
        assert _is_safe_url("") is False


class TestExecuteBrowserAction:
    @patch("core.browser_service.webbrowser.open")
    def test_open_url(self, mock_open):
        result = execute_browser_action({"action": "open_url", "url": "https://example.com"})
        mock_open.assert_called_once_with("https://example.com")
        assert "Opened" in result

    @patch("core.browser_service.webbrowser.open")
    def test_open_url_no_url(self, mock_open):
        result = execute_browser_action({"action": "open_url", "url": ""})
        mock_open.assert_not_called()
        assert "No URL" in result

    @patch("core.browser_service.webbrowser.open")
    def test_open_url_unsafe_scheme(self, mock_open):
        result = execute_browser_action({"action": "open_url", "url": "file:///etc/passwd"})
        mock_open.assert_not_called()
        assert "Blocked" in result

    @patch("core.browser_service.webbrowser.open")
    def test_search_youtube(self, mock_open):
        result = execute_browser_action({"action": "search", "site": "youtube", "query": "piano music"})
        mock_open.assert_called_once_with("https://www.youtube.com/results?search_query=piano+music")
        assert "piano music" in result

    @patch("core.browser_service.webbrowser.open")
    def test_search_no_query(self, mock_open):
        result = execute_browser_action({"action": "search", "site": "youtube", "query": ""})
        mock_open.assert_not_called()
        assert "No search query" in result

    @patch("core.browser_service.webbrowser.open")
    def test_search_defaults_to_google(self, mock_open):
        result = execute_browser_action({"action": "search", "query": "test"})
        mock_open.assert_called_once()
        call_url = mock_open.call_args[0][0]
        assert "google.com" in call_url

    @patch("core.browser_service.webbrowser.open")
    def test_open_site_known(self, mock_open):
        result = execute_browser_action({"action": "open_site", "site": "youtube"})
        mock_open.assert_called_once_with("https://www.youtube.com")
        assert "youtube" in result

    @patch("core.browser_service.webbrowser.open")
    def test_open_site_unknown_constructs_url(self, mock_open):
        result = execute_browser_action({"action": "open_site", "site": "spotify"})
        mock_open.assert_called_once_with("https://www.spotify.com")
        assert "spotify" in result

    @patch("core.browser_service.webbrowser.open")
    def test_open_site_empty(self, mock_open):
        result = execute_browser_action({"action": "open_site", "site": ""})
        mock_open.assert_not_called()
        assert "No site" in result

    def test_unknown_action(self):
        result = execute_browser_action({"action": "delete_files"})
        assert "Unknown" in result


class TestExtractAndExecuteActions:
    @patch("core.browser_service.webbrowser.open")
    def test_extracts_and_executes(self, mock_open):
        text = (
            'I\'ll open YouTube for you!\n'
            '<<<BROWSER_ACTION>>>{"action": "search", "site": "youtube", "query": "piano music"}<<<END_ACTION>>>'
        )
        cleaned, results = extract_and_execute_actions(text)
        assert cleaned == "I'll open YouTube for you!"
        assert len(results) == 1
        assert "piano music" in results[0]
        mock_open.assert_called_once()

    def test_no_action_tags(self):
        text = "Just a normal reply."
        cleaned, results = extract_and_execute_actions(text)
        assert cleaned == text
        assert results == []

    @patch("core.browser_service.webbrowser.open")
    def test_multiple_actions(self, mock_open):
        text = (
            'Here you go!\n'
            '<<<BROWSER_ACTION>>>{"action": "open_site", "site": "youtube"}<<<END_ACTION>>>\n'
            '<<<BROWSER_ACTION>>>{"action": "search", "site": "google", "query": "test"}<<<END_ACTION>>>'
        )
        cleaned, results = extract_and_execute_actions(text)
        assert "<<<BROWSER_ACTION>>>" not in cleaned
        assert len(results) == 2
        assert mock_open.call_count == 2

    def test_malformed_json_skipped(self):
        text = "Hello!\n<<<BROWSER_ACTION>>>{bad json<<<END_ACTION>>>"
        cleaned, results = extract_and_execute_actions(text)
        assert results == []
        assert "<<<BROWSER_ACTION>>>" not in cleaned


class TestSupportedSites:
    def test_all_urls_are_https(self):
        for site, url in SUPPORTED_SITES.items():
            assert url.startswith("https://"), f"{site}: {url} does not start with https://"
