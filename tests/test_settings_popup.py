from __future__ import annotations

from core.config import LLMModelEntry, ProviderConfig
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


def test_settings_popup_screen_ocr_toggle_overwrites_previous_state() -> None:
    class DummyCheckbox:
        def __init__(self) -> None:
            self.checked = False

        def blockSignals(self, _enabled: bool) -> None:
            return None

        def setChecked(self, value: bool) -> None:
            self.checked = value

    class DummyPopup:
        def __init__(self) -> None:
            self.screen_ocr_checkbox = DummyCheckbox()
            self._updating = False

    popup = DummyPopup()

    settings_popup_module.SettingsPopup._sync_screen_ocr_toggle(popup, True)

    assert popup.screen_ocr_checkbox.checked is True


def test_settings_popup_model_selection_includes_provider_and_applies_selected_config() -> None:
    openai_config = ProviderConfig(provider="openai", model="gpt-4.1", family="gpt", name="gpt-4.1")
    openrouter_config = ProviderConfig(
        provider="openrouter",
        model="openai/gpt-4.1",
        family="gpt",
        name="gpt-4.1",
    )

    class DummyCombo:
        def __init__(self) -> None:
            self.items: list[tuple[str, object]] = []
            self.enabled = True
            self.current_index = -1

        def blockSignals(self, _enabled: bool) -> None:
            return None

        def clear(self) -> None:
            self.items.clear()
            self.current_index = -1

        def addItem(self, text: str, data: object) -> None:
            self.items.append((text, data))

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

        def findData(self, data: object) -> int:
            for index, (_, item_data) in enumerate(self.items):
                if item_data == data:
                    return index
            return -1

        def setCurrentIndex(self, index: int) -> None:
            self.current_index = index

        def currentData(self) -> object | None:
            if 0 <= self.current_index < len(self.items):
                return self.items[self.current_index][1]
            return None

    class DummyController:
        def __init__(self) -> None:
            self.provider_config = openrouter_config
            self.set_provider_calls: list[ProviderConfig] = []

        def set_provider(self, provider_config: ProviderConfig) -> None:
            self.set_provider_calls.append(provider_config)
            self.provider_config = provider_config

    class DummyPopup:
        def __init__(self) -> None:
            self._controller = DummyController()
            self._llm_entries = [
                LLMModelEntry(
                    display_name="gpt-4.1",
                    family="gpt",
                    name="gpt-4.1",
                    providers=(openai_config, openrouter_config),
                )
            ]
            self._updating = False
            self.model_combo = DummyCombo()
            self.status_messages: list[str] = []

        def _refresh_catalog(self) -> None:
            return None

        def _model_options(self) -> list[tuple[str, ProviderConfig]]:
            return settings_popup_module.SettingsPopup._model_options(self)

        def _set_status(self, text: str) -> None:
            self.status_messages.append(text)

        def _selected_provider_config(self) -> ProviderConfig | None:
            provider_config = self.model_combo.currentData()
            if isinstance(provider_config, ProviderConfig):
                return provider_config
            return None

        def _apply_selected_provider(self) -> None:
            return settings_popup_module.SettingsPopup._apply_selected_provider(self)

    popup = DummyPopup()

    settings_popup_module.SettingsPopup._sync_llm_selection(popup, popup._controller.provider_config)

    assert popup.model_combo.items == [
        ("gpt-4.1 (openai)", openai_config),
        ("gpt-4.1 (openrouter)", openrouter_config),
    ]
    assert popup.model_combo.currentData() == openrouter_config

    popup.model_combo.setCurrentIndex(0)
    settings_popup_module.SettingsPopup._on_model_selected(popup, 0)

    assert popup._controller.set_provider_calls == [openai_config]