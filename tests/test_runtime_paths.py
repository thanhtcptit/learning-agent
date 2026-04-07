from __future__ import annotations

from core import runtime_paths


def test_get_bundle_data_root_uses_meipass_when_frozen(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runtime_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_paths.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert runtime_paths.get_bundle_data_root() == tmp_path


def test_get_runtime_file_path_uses_executable_directory_when_frozen(monkeypatch, tmp_path) -> None:
    exe_path = tmp_path / "learning-agent.exe"
    monkeypatch.setattr(runtime_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_paths.sys, "executable", str(exe_path), raising=False)

    assert runtime_paths.get_runtime_file_path(".env") == tmp_path / ".env"
