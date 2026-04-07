from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Iterator, Sequence

import httpx

from core.config import ProviderConfig
from llm.base import LLMMessage


@dataclass
class _ActiveRequest:
	response: httpx.Response
	cancel_event: threading.Event | None


class OpenAIProvider:
	def __init__(
		self,
		*,
		model: str,
		api_key: str,
		base_url: str,
		temperature: float = 0.2,
		reasoning_effort: str | None = None,
		web_search_enabled: bool = False,
		web_search_external_web_access: bool = True,
		web_search_allowed_domains: Sequence[str] = (),
		max_output_tokens: int | None = None,
		timeout: float = 60.0,
	) -> None:
		self.model = model
		self._api_key = api_key
		self._base_url = base_url.rstrip("/")
		self._temperature = temperature
		self._reasoning_effort = self._clean_text(reasoning_effort)
		self._web_search_enabled = bool(web_search_enabled)
		self._web_search_external_web_access = bool(web_search_external_web_access)
		cleaned_domains: list[str] = []
		for domain in web_search_allowed_domains:
			cleaned_domain = str(domain).strip()
			if cleaned_domain:
				cleaned_domains.append(cleaned_domain)

		self._web_search_allowed_domains = tuple(cleaned_domains)
		self._max_output_tokens = max_output_tokens
		self._timeout = timeout
		self._active_request_lock = threading.Lock()
		self._active_request: _ActiveRequest | None = None

	@classmethod
	def from_config(cls, config: ProviderConfig) -> "OpenAIProvider":
		api_key = os.getenv(config.api_key_env, "").strip()
		if not api_key:
			raise RuntimeError(f"Missing API key. Set the {config.api_key_env} environment variable.")

		return cls(
			model=config.model,
			api_key=api_key,
			base_url=config.base_url,
			temperature=config.temperature,
			reasoning_effort=config.reasoning_effort,
			web_search_enabled=config.web_search_enabled,
			web_search_external_web_access=config.web_search_external_web_access,
			web_search_allowed_domains=config.web_search_allowed_domains,
			max_output_tokens=config.max_output_tokens,
		)

	def cancel_current_request(self) -> None:
		with self._active_request_lock:
			active_request = self._active_request

		if active_request is None:
			return

		if active_request.cancel_event is not None:
			active_request.cancel_event.set()

		active_request.response.close()

	def stream_chat(
		self,
		messages: Sequence[LLMMessage],
		*,
		temperature: float | None = None,
		cancel_event: threading.Event | None = None,
	) -> Iterator[str]:
		if cancel_event is not None and cancel_event.is_set():
			return

		payload: dict[str, object] = {
			"model": self.model,
			"input": [{"role": message.role, "content": message.content} for message in messages],
			"stream": True,
			"store": False,
			"temperature": self._temperature if temperature is None else temperature,
			"tool_choice": "auto",
		}

		if self._max_output_tokens is not None:
			payload["max_output_tokens"] = self._max_output_tokens

		if self._reasoning_effort:
			payload["reasoning"] = {"effort": self._reasoning_effort}

		tools = self._build_tools()
		if tools:
			payload["tools"] = tools
			payload["include"] = ["web_search_call.action.sources"]

		headers = {
			"Authorization": f"Bearer {self._api_key}",
			"Content-Type": "application/json",
		}

		url = f"{self._base_url}/responses"

		try:
			with httpx.Client(timeout=self._timeout) as client:
				with client.stream("POST", url, json=payload, headers=headers) as response:
					with self._active_request_lock:
						self._active_request = _ActiveRequest(response=response, cancel_event=cancel_event)

					try:
						response.raise_for_status()
						yield from self._stream_response(response, cancel_event)
					finally:
						with self._active_request_lock:
							if self._active_request and self._active_request.response is response:
								self._active_request = None
		except httpx.HTTPError as exc:
			if cancel_event is not None and cancel_event.is_set():
				return
			raise RuntimeError(f"OpenAI request failed: {exc}") from exc

	def _build_tools(self) -> list[dict[str, object]]:
		if not self._web_search_enabled:
			return []

		tool: dict[str, object] = {
			"type": "web_search",
			"external_web_access": self._web_search_external_web_access,
		}
		if self._web_search_allowed_domains:
			tool["filters"] = {"allowed_domains": list(self._web_search_allowed_domains)}
		return [tool]

	def _stream_response(
		self,
		response: httpx.Response,
		cancel_event: threading.Event | None,
	) -> Iterator[str]:
		saw_output_text_delta = False
		pending_lines: list[str] = []
		completed_response: dict[str, object] | None = None

		for raw_line in response.iter_lines():
			if cancel_event is not None and cancel_event.is_set():
				break

			if raw_line is None:
				continue

			line = raw_line.strip()
			if not line:
				if not pending_lines:
					continue
				event_payload = self._parse_event_payload(pending_lines)
				pending_lines = []
				if event_payload is None:
					continue

				event_type = str(event_payload.get("type") or "")
				if event_type == "response.output_text.delta":
					delta = str(event_payload.get("delta") or "")
					if delta:
						saw_output_text_delta = True
						yield delta
					continue

				if event_type == "response.output_text.done":
					if not saw_output_text_delta:
						text = str(event_payload.get("text") or "")
						if text:
							yield text
							saw_output_text_delta = True
					continue

				if event_type == "response.completed":
					completed_response = event_payload.get("response") if isinstance(event_payload.get("response"), dict) else None
					break

				if event_type in {"response.failed", "error"}:
					message = self._extract_error_message(event_payload)
					raise RuntimeError(f"OpenAI request failed: {message}")

				continue

			if line.startswith("data:"):
				pending_lines.append(line.removeprefix("data:").strip())
				continue

			if line.startswith("event:"):
				continue

			pending_lines.append(line)

		if pending_lines:
			event_payload = self._parse_event_payload(pending_lines)
			if event_payload is not None:
				event_type = str(event_payload.get("type") or "")
				if event_type == "response.output_text.delta":
					delta = str(event_payload.get("delta") or "")
					if delta:
						saw_output_text_delta = True
						yield delta
				elif event_type == "response.output_text.done" and not saw_output_text_delta:
					text = str(event_payload.get("text") or "")
					if text:
						yield text
				elif event_type == "response.completed":
					completed_response = event_payload.get("response") if isinstance(event_payload.get("response"), dict) else None
				elif event_type in {"response.failed", "error"}:
					message = self._extract_error_message(event_payload)
					raise RuntimeError(f"OpenAI request failed: {message}")

		if not saw_output_text_delta and completed_response is not None:
			final_text = self._extract_completed_text(completed_response)
			if final_text:
				yield final_text

	def _parse_event_payload(self, lines: list[str]) -> dict[str, object] | None:
		data = "\n".join(part for part in lines if part).strip()
		if not data:
			return None

		try:
			payload = json.loads(data)
		except json.JSONDecodeError:
			return None

		if isinstance(payload, dict):
			return payload
		return None

	def _extract_completed_text(self, response_payload: dict[str, object]) -> str:
		output = response_payload.get("output")
		if not isinstance(output, list):
			return ""

		chunks: list[str] = []
		for item in output:
			if not isinstance(item, dict):
				continue
			if item.get("type") != "message" or item.get("role") != "assistant":
				continue

			content = item.get("content")
			if not isinstance(content, list):
				continue

			for part in content:
				if not isinstance(part, dict):
					continue
				if part.get("type") == "output_text":
					text = str(part.get("text") or "")
					if text:
						chunks.append(text)

		return "".join(chunks).strip()

	def _extract_error_message(self, payload: dict[str, object]) -> str:
		error = payload.get("error")
		if isinstance(error, dict):
			message = str(error.get("message") or "").strip()
			if message:
				return message

		message = str(payload.get("message") or "").strip()
		if message:
			return message

		return "Unknown OpenAI API error"

	def _clean_text(self, value: str | None) -> str | None:
		if not value:
			return None

		cleaned = value.strip()
		return cleaned or None


OpenAICompatibleProvider = OpenAIProvider
