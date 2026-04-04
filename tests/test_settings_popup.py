from __future__ import annotations

import ui.settings_popup as settings_popup_module


def test_settings_popup_current_language_status_overwrites_previous_text() -> None:
    class DummyLabel:
        def __init__(self) -> None:
            self.text = "Error"

        def setText(self, value: str) -> None:
            self.text = value

    class DummyPopup:
        def __init__(self) -> None:
            self.status_label = DummyLabel()

    popup = DummyPopup()

    settings_popup_module.SettingsPopup._sync_current_language_status(popup, "Vietnamese")

    assert popup.status_label.text == "Current language: Vietnamese"