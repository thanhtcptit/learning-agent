from __future__ import annotations

from core.config import LLMModelEntry, ProviderConfig
from core.voice_catalog import (
    DEFAULT_VIETNAMESE_STT_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    PHOWHISPER_MEDIUM_STT_MODEL_ID,
    VIENEU_TTS_03B_MODEL_ID,
    VIENEU_TTS_03B_Q4_GGUF_MODEL_ID,
    VIETNAMESE_STT_MODEL_CHOICES,
    VIETNAMESE_TTS_VOICE_CHOICES,
    VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS,
    VoicePresetChoice,
)
import ui.settings_popup as settings_popup_module


def test_settings_popup_current_language_status_overwrites_previous_text() -> None:
    class DummyLabel:
        def __init__(self) -> None:
            self.text = "Error"
            self.tool_tip = ""

        def setText(self, value: str) -> None:
            self.text = value

        def setToolTip(self, value: str) -> None:
            self.tool_tip = value

    class DummyPopup:
        def __init__(self) -> None:
            self.status_label = DummyLabel()

    popup = DummyPopup()

    settings_popup_module.SettingsPopup._sync_current_language_status(popup, "Vietnamese")

    assert popup.status_label.text == "Current language: Vietnamese"
    assert popup.status_label.tool_tip == "Current language: Vietnamese"


def test_settings_popup_status_tooltip_tracks_full_text() -> None:
    class DummyLabel:
        def __init__(self) -> None:
            self.text = ""
            self.tool_tip = ""

        def setText(self, value: str) -> None:
            self.text = value

        def setToolTip(self, value: str) -> None:
            self.tool_tip = value

    class DummyPopup:
        def __init__(self) -> None:
            self.status_label = DummyLabel()

    popup = DummyPopup()

    settings_popup_module.SettingsPopup._set_status(popup, "Speech ready")

    assert popup.status_label.text == "Speech ready"
    assert popup.status_label.tool_tip == "Speech ready"


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


def test_settings_popup_voice_model_selection_includes_choices_and_applies_selected_model() -> None:
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
            self.voice_stt_model_id = DEFAULT_VIETNAMESE_STT_MODEL_ID
            self.voice_tts_model_id = DEFAULT_VIETNAMESE_TTS_MODEL_ID
            self.voice_tts_voice_name = DEFAULT_VIETNAMESE_TTS_VOICE_NAME
            self.voice_stt_calls: list[str] = []
            self.voice_tts_calls: list[str] = []
            self.voice_tts_voice_calls: list[str] = []

        def set_voice_stt_model_id(self, model_id: str) -> None:
            self.voice_stt_calls.append(model_id)
            self.voice_stt_model_id = model_id

        def set_voice_tts_model_id(self, model_id: str) -> None:
            self.voice_tts_calls.append(model_id)
            self.voice_tts_model_id = model_id

        def set_voice_tts_voice_name(self, voice_name: str) -> None:
            self.voice_tts_voice_calls.append(voice_name)
            self.voice_tts_voice_name = voice_name

    class DummyPopup:
        def __init__(self) -> None:
            self._controller = DummyController()
            self._updating = False
            self.voice_stt_combo = DummyCombo()
            self.voice_tts_combo = DummyCombo()
            self.voice_tts_voice_combo = DummyCombo()
            self.status_messages: list[str] = []

        def _set_status(self, text: str) -> None:
            self.status_messages.append(text)

        def _selected_voice_stt_model_id(self) -> str | None:
            model_id = self.voice_stt_combo.currentData()
            if isinstance(model_id, str) and model_id:
                return model_id
            return None

        def _selected_voice_tts_model_id(self) -> str | None:
            model_id = self.voice_tts_combo.currentData()
            if isinstance(model_id, str) and model_id:
                return model_id
            return None

        def _apply_selected_voice_stt_model(self) -> None:
            model_id = self._selected_voice_stt_model_id()
            if model_id is None or model_id == self._controller.voice_stt_model_id:
                return
            self._controller.set_voice_stt_model_id(model_id)

        def _apply_selected_voice_tts_model(self) -> None:
            model_id = self._selected_voice_tts_model_id()
            if model_id is None or model_id == self._controller.voice_tts_model_id:
                return
            self._controller.set_voice_tts_model_id(model_id)

        def _selected_voice_tts_voice_name(self) -> str | None:
            voice_name = self.voice_tts_voice_combo.currentData()
            if isinstance(voice_name, str) and voice_name:
                return voice_name
            return None

        def _apply_selected_voice_tts_voice_name(self) -> None:
            voice_name = self._selected_voice_tts_voice_name()
            if voice_name is None or voice_name == self._controller.voice_tts_voice_name:
                return
            self._controller.set_voice_tts_voice_name(voice_name)

    popup = DummyPopup()

    settings_popup_module.SettingsPopup._sync_voice_stt_selection(popup, popup._controller.voice_stt_model_id)
    settings_popup_module.SettingsPopup._sync_voice_tts_selection(popup, popup._controller.voice_tts_model_id)
    settings_popup_module.SettingsPopup._sync_voice_tts_voice_selection(popup, popup._controller.voice_tts_voice_name)

    assert popup.voice_stt_combo.items == [
        (VIETNAMESE_STT_MODEL_CHOICES[0].label, DEFAULT_VIETNAMESE_STT_MODEL_ID),
        (VIETNAMESE_STT_MODEL_CHOICES[1].label, PHOWHISPER_MEDIUM_STT_MODEL_ID),
    ]
    assert popup.voice_tts_combo.items == [
        (VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS[0].label, DEFAULT_VIETNAMESE_TTS_MODEL_ID),
        (VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS[1].label, VIENEU_TTS_03B_Q4_GGUF_MODEL_ID),
        (VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS[2].label, VIENEU_TTS_03B_MODEL_ID),
    ]
    assert popup.voice_tts_voice_combo.items == [
        (VIETNAMESE_TTS_VOICE_CHOICES[0].label, VIETNAMESE_TTS_VOICE_CHOICES[0].voice_name),
        (VIETNAMESE_TTS_VOICE_CHOICES[1].label, VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name),
        (VIETNAMESE_TTS_VOICE_CHOICES[2].label, VIETNAMESE_TTS_VOICE_CHOICES[2].voice_name),
        (VIETNAMESE_TTS_VOICE_CHOICES[3].label, VIETNAMESE_TTS_VOICE_CHOICES[3].voice_name),
        (VIETNAMESE_TTS_VOICE_CHOICES[4].label, VIETNAMESE_TTS_VOICE_CHOICES[4].voice_name),
        (VIETNAMESE_TTS_VOICE_CHOICES[5].label, VIETNAMESE_TTS_VOICE_CHOICES[5].voice_name),
    ]

    popup.voice_stt_combo.setCurrentIndex(1)
    popup.voice_tts_combo.setCurrentIndex(2)
    popup.voice_tts_voice_combo.setCurrentIndex(1)
    settings_popup_module.SettingsPopup._on_voice_stt_selected(popup, 1)
    settings_popup_module.SettingsPopup._on_voice_tts_selected(popup, 2)
    settings_popup_module.SettingsPopup._on_voice_tts_voice_selected(popup, 1)

    assert popup._controller.voice_stt_calls == [PHOWHISPER_MEDIUM_STT_MODEL_ID]
    assert popup._controller.voice_tts_calls == [VIENEU_TTS_03B_MODEL_ID]
    assert popup._controller.voice_tts_voice_calls == [VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name]


def test_settings_popup_refreshes_voice_preset_list_when_tts_model_changes() -> None:
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
            self.voice_tts_model_id = DEFAULT_VIETNAMESE_TTS_MODEL_ID
            self.voice_tts_voice_name = DEFAULT_VIETNAMESE_TTS_VOICE_NAME
            self.voice_tts_model_calls: list[str] = []

        def set_voice_tts_model_id(self, model_id: str) -> None:
            self.voice_tts_model_calls.append(model_id)
            self.voice_tts_model_id = model_id

    class DummyPopup:
        def __init__(self) -> None:
            self._controller = DummyController()
            self._updating = False
            self.voice_tts_combo = DummyCombo()
            self.voice_tts_voice_combo = DummyCombo()

        def _selected_voice_tts_model_id(self) -> str | None:
            model_id = self.voice_tts_combo.currentData()
            if isinstance(model_id, str) and model_id:
                return model_id
            return None

        def _apply_selected_voice_tts_model(self) -> None:
            model_id = self._selected_voice_tts_model_id()
            if model_id is None or model_id == self._controller.voice_tts_model_id:
                return
            self._controller.set_voice_tts_model_id(model_id)
            settings_popup_module.SettingsPopup._sync_voice_tts_voice_selection_from_model(self, self._controller.voice_tts_model_id)

    model_one_choices = (
        VoicePresetChoice("Vinh", "Vĩnh (nam miền Nam)"),
        VoicePresetChoice("Binh", "Bình (nam miền Bắc)"),
    )
    model_two_choices = (
        VoicePresetChoice("Ly", "Ly (nữ miền Bắc)"),
        VoicePresetChoice("Ngoc", "Ngọc (nữ miền Bắc)"),
    )

    def fake_voice_choices_for_model(model_id: str | None) -> tuple[VoicePresetChoice, ...]:
        if model_id == DEFAULT_VIETNAMESE_TTS_MODEL_ID:
            return model_one_choices
        if model_id == VIENEU_TTS_03B_MODEL_ID:
            return model_two_choices
        return model_one_choices

    popup = DummyPopup()
    original_helper = settings_popup_module.vietnamese_tts_voice_choices_for_model
    settings_popup_module.vietnamese_tts_voice_choices_for_model = fake_voice_choices_for_model
    try:
        settings_popup_module.SettingsPopup._sync_voice_tts_voice_selection_from_model(popup, popup._controller.voice_tts_model_id)

        assert popup.voice_tts_voice_combo.items == [
            ("Vĩnh (nam miền Nam)", "Vinh"),
            ("Bình (nam miền Bắc)", "Binh"),
        ]

        popup.voice_tts_combo.items = [("VieNeu TTS 0.3B", VIENEU_TTS_03B_MODEL_ID)]
        popup.voice_tts_combo.current_index = 0
        popup._apply_selected_voice_tts_model()

        assert popup._controller.voice_tts_model_calls == [VIENEU_TTS_03B_MODEL_ID]
        assert popup.voice_tts_voice_combo.items == [
            ("Ly (nữ miền Bắc)", "Ly"),
            ("Ngọc (nữ miền Bắc)", "Ngoc"),
        ]
    finally:
        settings_popup_module.vietnamese_tts_voice_choices_for_model = original_helper