from __future__ import annotations

from unittest.mock import patch

import httpx

from core.browser_service import (
    _fetch_first_youtube_video_url,
    execute_browser_action,
    extract_and_execute_actions,
)


class TestExecuteBrowserAction:
    def test_unknown_action(self):
        result = execute_browser_action({"action": "delete_files"})
        assert "Unknown" in result

    @patch("core.browser_service._fetch_first_youtube_video_url", return_value="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    @patch("core.browser_service.webbrowser.open")
    def test_search_and_play_opens_first_video(self, mock_open, mock_fetch):
        result = execute_browser_action({"action": "search_and_play", "query": "piano music"})
        mock_fetch.assert_called_once_with("piano music")
        mock_open.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert "Playing" in result
        assert "piano music" in result

    @patch("core.browser_service._fetch_first_youtube_video_url", return_value=None)
    @patch("core.browser_service.webbrowser.open")
    def test_search_and_play_falls_back_to_search(self, mock_open, mock_fetch):
        result = execute_browser_action({"action": "search_and_play", "query": "piano music"})
        mock_open.assert_called_once()
        call_url = mock_open.call_args[0][0]
        assert "youtube.com/results" in call_url
        assert "could not auto-play" in result

    @patch("core.browser_service.webbrowser.open")
    def test_search_and_play_no_query(self, mock_open):
        result = execute_browser_action({"action": "search_and_play", "query": ""})
        mock_open.assert_not_called()
        assert "No search query" in result


class TestFetchFirstYoutubeVideoUrl:
    @patch("core.browser_service.httpx.get")
    def test_extracts_video_id_from_html(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = 'some html /watch?v=abc12345678 more html'
        mock_get.return_value.raise_for_status = lambda: None
        url = _fetch_first_youtube_video_url("test")
        assert url == "https://www.youtube.com/watch?v=abc12345678"

    @patch("core.browser_service.httpx.get")
    def test_returns_none_when_no_video_found(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = 'no video ids here'
        mock_get.return_value.raise_for_status = lambda: None
        url = _fetch_first_youtube_video_url("test")
        assert url is None

    @patch("core.browser_service.httpx.get", side_effect=httpx.ConnectError("network error"))
    def test_returns_none_on_network_error(self, mock_get):
        url = _fetch_first_youtube_video_url("test")
        assert url is None


class TestExtractAndExecuteActions:
    @patch("core.browser_service._fetch_first_youtube_video_url", return_value="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    @patch("core.browser_service.webbrowser.open")
    def test_extracts_and_executes(self, mock_open, mock_fetch):
        text = (
            'I\'ll open YouTube for you!\n'
            '<<<BROWSER_ACTION>>>{"action": "search_and_play", "query": "piano music"}<<<END_ACTION>>>'
        )
        cleaned, results, suppress_tts = extract_and_execute_actions(text)
        assert cleaned == "I'll open YouTube for you!"
        assert len(results) == 1
        assert "piano music" in results[0]
        assert suppress_tts is True
        mock_fetch.assert_called_once_with("piano music")
        mock_open.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_no_action_tags(self):
        text = "Just a normal reply."
        cleaned, results, suppress_tts = extract_and_execute_actions(text)
        assert cleaned == text
        assert results == []
        assert suppress_tts is False

    @patch("core.browser_service._fetch_first_youtube_video_url", side_effect=["https://www.youtube.com/watch?v=video1111111", "https://www.youtube.com/watch?v=video2222222"])
    @patch("core.browser_service.webbrowser.open")
    def test_multiple_actions(self, mock_open, mock_fetch):
        text = (
            'Here you go!\n'
            '<<<BROWSER_ACTION>>>{"action": "search_and_play", "query": "piano music"}<<<END_ACTION>>>\n'
            '<<<BROWSER_ACTION>>>{"action": "search_and_play", "query": "guitar music"}<<<END_ACTION>>>'
        )
        cleaned, results, suppress_tts = extract_and_execute_actions(text)
        assert "<<<BROWSER_ACTION>>>" not in cleaned
        assert len(results) == 2
        assert mock_open.call_count == 2
        assert mock_fetch.call_count == 2
        assert suppress_tts is True

    def test_malformed_json_skipped(self):
        text = "Hello!\n<<<BROWSER_ACTION>>>{bad json<<<END_ACTION>>>"
        cleaned, results, suppress_tts = extract_and_execute_actions(text)
        assert results == []
        assert "<<<BROWSER_ACTION>>>" not in cleaned
        assert suppress_tts is False
