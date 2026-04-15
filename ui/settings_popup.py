from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.config import LLMModelEntry, ProviderConfig, discover_llm_catalog
from core.hotkey import GlobalHotkeyListener
from core.orchestrator import AppController
from core.voice_catalog import (
    DEFAULT_VIETNAMESE_STT_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    VIETNAMESE_STT_MODEL_CHOICES,
    VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS,
    resolve_voice_model_id,
    resolve_vietnamese_tts_voice_name,
    vietnamese_tts_voice_choices_for_model,
)
from prompts.templates import DEFAULT_TARGET_LANGUAGE


class SettingsPopup(QDialog):
    def __init__(self, controller: AppController, hotkey_listener: GlobalHotkeyListener, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._hotkey_listener = hotkey_listener
        self._updating = False
        self._llm_entries: list[LLMModelEntry] = []

        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("SettingsPopup")
        self.setFixedWidth(360)

        self._build_ui()
        self._connect_signals()
        self._apply_state()

    def show_near(self, anchor_widget: QWidget) -> None:
        self._apply_state()
        self.adjustSize()

        anchor_position = anchor_widget.mapToGlobal(QPoint(0, anchor_widget.height()))
        x_position = max(12, anchor_position.x() - self.width() + anchor_widget.width())
        y_position = anchor_position.y() + 8
        self.move(x_position, y_position)
        self.show()
        self.raise_()
        self.activateWindow()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("PopupCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(12)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        title_label = QLabel("Settings")
        title_label.setObjectName("PopupTitle")
        close_button = QPushButton("×")
        close_button.setObjectName("PopupCloseButton")
        close_button.setFixedSize(28, 28)
        close_button.clicked.connect(self.close)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(close_button)

        section_title = QLabel("LLM")
        section_title.setObjectName("PopupSectionTitle")

        self.model_label = QLabel("Model")
        self.model_label.setObjectName("PopupFieldLabel")
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("PopupModelCombo")

        llm_layout = QVBoxLayout()
        llm_layout.setSpacing(6)
        llm_layout.addWidget(self.model_label)
        llm_layout.addWidget(self.model_combo)

        voice_section_title = QLabel("Voice")
        voice_section_title.setObjectName("PopupSectionTitle")

        self.voice_stt_label = QLabel("Vietnamese STT")
        self.voice_stt_label.setObjectName("PopupFieldLabel")
        self.voice_stt_combo = QComboBox()
        self.voice_stt_combo.setObjectName("PopupVoiceSttCombo")

        self.voice_tts_label = QLabel("Vietnamese TTS")
        self.voice_tts_label.setObjectName("PopupFieldLabel")
        self.voice_tts_combo = QComboBox()
        self.voice_tts_combo.setObjectName("PopupVoiceTtsCombo")

        self.voice_tts_voice_label = QLabel("Voice preset")
        self.voice_tts_voice_label.setObjectName("PopupFieldLabel")
        self.voice_tts_voice_combo = QComboBox()
        self.voice_tts_voice_combo.setObjectName("PopupVoiceNameCombo")

        voice_stt_layout = QVBoxLayout()
        voice_stt_layout.setSpacing(6)
        voice_stt_layout.addWidget(self.voice_stt_label)
        voice_stt_layout.addWidget(self.voice_stt_combo)

        voice_tts_layout = QVBoxLayout()
        voice_tts_layout.setSpacing(6)
        voice_tts_layout.addWidget(self.voice_tts_label)
        voice_tts_layout.addWidget(self.voice_tts_combo)

        voice_tts_voice_layout = QVBoxLayout()
        voice_tts_voice_layout.setSpacing(6)
        voice_tts_voice_layout.addWidget(self.voice_tts_voice_label)
        voice_tts_voice_layout.addWidget(self.voice_tts_voice_combo)

        voice_layout = QVBoxLayout()
        voice_layout.setSpacing(6)
        voice_layout.addLayout(voice_stt_layout)
        voice_layout.addLayout(voice_tts_layout)
        voice_layout.addLayout(voice_tts_voice_layout)

        self.hotkey_checkbox = QCheckBox("Enable global hotkeys")
        self.hotkey_checkbox.setObjectName("PopupHotkeyToggle")

        self.screen_ocr_checkbox = QCheckBox("Enable screen OCR context")
        self.screen_ocr_checkbox.setObjectName("PopupScreenOcrToggle")
        self.screen_ocr_checkbox.setToolTip(
            "Capture the current monitor and use its text as transient context when the Definition hotkey is pressed."
        )

        self.screen_ocr_hint = QLabel(
            "Useful for Definition mode when the selected text depends on nearby labels, code, tables, or other surrounding content."
        )
        self.screen_ocr_hint.setObjectName("PopupHint")
        self.screen_ocr_hint.setWordWrap(True)

        self.session_combo = QComboBox()
        self.new_session_button = QPushButton("New")
        self.new_session_button.setObjectName("PopupSecondaryButton")
        self.new_session_button.setFixedWidth(56)

        self.delete_session_button = QPushButton("Delete")
        self.delete_session_button.setObjectName("PopupDangerButton")
        self.delete_session_button.setFixedWidth(64)

        session_row = QHBoxLayout()
        session_row.setSpacing(8)
        session_row.addWidget(self.session_combo, 1)
        session_row.addWidget(self.new_session_button)
        session_row.addWidget(self.delete_session_button)

        self.language_input = QLineEdit()
        self.language_input.setPlaceholderText(DEFAULT_TARGET_LANGUAGE)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("PopupStatus")
        self.status_label.setWordWrap(True)

        self.done_button = QPushButton("Done")
        self.done_button.setObjectName("PopupPrimaryButton")
        self.done_button.clicked.connect(self.close)

        card_layout.addLayout(header_layout)
        card_layout.addWidget(section_title)
        card_layout.addLayout(llm_layout)
        card_layout.addWidget(self.hotkey_checkbox)
        card_layout.addWidget(self.screen_ocr_checkbox)
        card_layout.addWidget(self.screen_ocr_hint)
        card_layout.addWidget(voice_section_title)
        card_layout.addLayout(voice_layout)
        card_layout.addLayout(session_row)
        card_layout.addWidget(self.language_input)

        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.status_label, 1)
        footer_layout.addWidget(self.done_button)
        card_layout.addLayout(footer_layout)

        root_layout.addWidget(card)

        self.setStyleSheet(
            """
            QDialog#SettingsPopup {
                background: transparent;
            }
            QFrame#PopupCard {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 16px;
            }
            QLabel#PopupTitle {
                color: #0f172a;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#PopupSectionTitle {
                color: #0f172a;
                font-size: 13px;
                font-weight: 700;
                margin-top: 2px;
            }
            QLabel#PopupFieldLabel {
                color: #334155;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#PopupStatus {
                color: #64748b;
                font-size: 12px;
                min-height: 32px;
            }
            QLabel#PopupHint {
                color: #64748b;
                font-size: 11px;
            }
            QComboBox#PopupModelCombo {
                min-width: 0px;
            }
            QCheckBox#PopupHotkeyToggle, QCheckBox#PopupScreenOcrToggle {
                color: #0f172a;
                font-weight: 600;
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 8px 10px;
                font-weight: 600;
            }
            QPushButton#PopupPrimaryButton {
                background: #2563eb;
                color: white;
            }
            QPushButton#PopupSecondaryButton {
                background: #f1f5f9;
                color: #0f172a;
                border: 1px solid #dbe3ee;
            }
            QPushButton#PopupDangerButton {
                background: #fef2f2;
                color: #b91c1c;
                border: 1px solid #fecaca;
            }
            QPushButton#PopupCloseButton {
                background: #f8fafc;
                color: #334155;
                border: 1px solid #dbe3ee;
                padding: 0;
                font-size: 18px;
            }
            QLineEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 10px;
                color: #0f172a;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #60a5fa;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
                outline: none;
            }
            """
        )

    def _connect_signals(self) -> None:
        self._controller.sessions_changed.connect(self._sync_sessions)
        self._controller.current_session_changed.connect(self._sync_sessions)
        self._controller.current_session_changed.connect(self._sync_llm_selection)
        self._controller.provider_config_changed.connect(self._sync_llm_selection)
        self._controller.preferred_language_changed.connect(self._sync_language_input)
        self._controller.screen_ocr_enabled_changed.connect(self._sync_screen_ocr_toggle)
        self._controller.current_language_changed.connect(self._sync_current_language_status)
        self._controller.voice_stt_model_changed.connect(self._sync_voice_stt_selection)
        self._controller.voice_tts_model_changed.connect(self._sync_voice_tts_selection)
        self._controller.voice_tts_model_changed.connect(self._sync_voice_tts_voice_selection_from_model)
        self._controller.voice_tts_voice_name_changed.connect(self._sync_voice_tts_voice_selection)
        self._controller.status_changed.connect(self._set_status)
        self._hotkey_listener.status_changed.connect(self._set_status)

        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        self.voice_stt_combo.currentIndexChanged.connect(self._on_voice_stt_selected)
        self.voice_tts_combo.currentIndexChanged.connect(self._on_voice_tts_selected)
        self.voice_tts_voice_combo.currentIndexChanged.connect(self._on_voice_tts_voice_selected)
        self.hotkey_checkbox.toggled.connect(self._on_hotkey_toggled)
        self.screen_ocr_checkbox.toggled.connect(self._on_screen_ocr_toggled)
        self.session_combo.currentIndexChanged.connect(self._on_session_selected)
        self.new_session_button.clicked.connect(self._on_new_session)
        self.delete_session_button.clicked.connect(self._on_delete_session)
        self.language_input.editingFinished.connect(self._on_language_changed)

    def _apply_state(self) -> None:
        self._sync_sessions(self._controller.sessions)
        self.language_input.setText(self._controller.preferred_language)
        self._sync_current_language_status(self._controller.target_language)
        self.hotkey_checkbox.blockSignals(True)
        self.hotkey_checkbox.setChecked(self._hotkey_listener.is_running)
        self.hotkey_checkbox.blockSignals(False)
        self._sync_screen_ocr_toggle(self._controller.screen_ocr_enabled)
        self._sync_llm_selection(self._controller.provider_config)
        self._sync_voice_stt_selection(self._controller.voice_stt_model_id)
        self._sync_voice_tts_selection(self._controller.voice_tts_model_id)
        self._sync_voice_tts_voice_selection(self._controller.voice_tts_voice_name)

    def _sync_sessions(self, _value: object) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            current_session_id = self._controller.current_session.id
            self.session_combo.blockSignals(True)
            self.session_combo.clear()
            for session in self._controller.sessions:
                self.session_combo.addItem(session.title, session.id)
            current_index = self.session_combo.findData(current_session_id)
            if current_index >= 0:
                self.session_combo.setCurrentIndex(current_index)
            self.session_combo.blockSignals(False)

            self.language_input.setText(self._controller.preferred_language)
        finally:
            self._updating = False

    def _sync_language_input(self, language: str) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            self.language_input.setText(language)
        finally:
            self._updating = False

    def _sync_current_language_status(self, language: str) -> None:
        text = f"Current language: {language}"
        self.status_label.setText(text)
        self.status_label.setToolTip(text)

    def _sync_screen_ocr_toggle(self, enabled: bool) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            self.screen_ocr_checkbox.blockSignals(True)
            self.screen_ocr_checkbox.setChecked(enabled)
            self.screen_ocr_checkbox.blockSignals(False)
        finally:
            self._updating = False

    def _refresh_catalog(self) -> None:
        self._llm_entries = discover_llm_catalog()

    def _model_options(self) -> list[tuple[str, ProviderConfig]]:
        options: list[tuple[str, ProviderConfig]] = []
        for entry in self._llm_entries:
            for provider_config in entry.providers:
                options.append((f"{entry.display_name} ({provider_config.provider})", provider_config))
        return options

    def _sync_llm_selection(self, selected_config: object | None = None) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            self._refresh_catalog()
            if not self._llm_entries:
                self.model_combo.blockSignals(True)
                self.model_combo.clear()
                self.model_combo.addItem("No models found", None)
                self.model_combo.setEnabled(False)
                self.model_combo.blockSignals(False)
                self._set_status("No LLM configs found")
                return

            model_options = self._model_options()
            if not model_options:
                self.model_combo.blockSignals(True)
                self.model_combo.clear()
                self.model_combo.addItem("No providers found", None)
                self.model_combo.setEnabled(False)
                self.model_combo.blockSignals(False)
                self._set_status("Selected LLMs have no providers")
                return

            resolved_provider_config = None
            if isinstance(selected_config, ProviderConfig):
                resolved_provider_config = next(
                    (provider_config for _, provider_config in model_options if provider_config == selected_config),
                    None,
                )
            if resolved_provider_config is None:
                current_provider_config = self._selected_provider_config()
                if current_provider_config is not None:
                    resolved_provider_config = next(
                        (provider_config for _, provider_config in model_options if provider_config == current_provider_config),
                        None,
                    )
            if resolved_provider_config is None:
                resolved_provider_config = model_options[0][1]

            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            for label, provider_config in model_options:
                self.model_combo.addItem(label, provider_config)
            self.model_combo.setEnabled(True)
            model_index = self.model_combo.findData(resolved_provider_config)
            if model_index < 0:
                model_index = 0
            self.model_combo.setCurrentIndex(model_index)
            self.model_combo.blockSignals(False)
        finally:
            self._updating = False

    def _sync_voice_stt_selection(self, selected_model_id: object | None = None) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            resolved_model_id = selected_model_id if isinstance(selected_model_id, str) else self._controller.voice_stt_model_id
            resolved_model_id = resolve_voice_model_id(
                resolved_model_id,
                VIETNAMESE_STT_MODEL_CHOICES,
                DEFAULT_VIETNAMESE_STT_MODEL_ID,
            )

            self.voice_stt_combo.blockSignals(True)
            self.voice_stt_combo.clear()
            for choice in VIETNAMESE_STT_MODEL_CHOICES:
                self.voice_stt_combo.addItem(choice.label, choice.model_id)
            self.voice_stt_combo.setEnabled(True)
            current_index = self.voice_stt_combo.findData(resolved_model_id)
            if current_index < 0:
                current_index = 0
            self.voice_stt_combo.setCurrentIndex(current_index)
            self.voice_stt_combo.blockSignals(False)
        finally:
            self._updating = False

    def _sync_voice_tts_selection(self, selected_model_id: object | None = None) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            resolved_model_id = selected_model_id if isinstance(selected_model_id, str) else self._controller.voice_tts_model_id
            resolved_model_id = resolve_voice_model_id(
                resolved_model_id,
                VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS,
                DEFAULT_VIETNAMESE_TTS_MODEL_ID,
            )

            self.voice_tts_combo.blockSignals(True)
            self.voice_tts_combo.clear()
            for choice in VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS:
                self.voice_tts_combo.addItem(choice.label, choice.model_id)
            self.voice_tts_combo.setEnabled(True)
            current_index = self.voice_tts_combo.findData(resolved_model_id)
            if current_index < 0:
                current_index = 0
            self.voice_tts_combo.setCurrentIndex(current_index)
            self.voice_tts_combo.blockSignals(False)
        finally:
            self._updating = False

    def _sync_voice_tts_voice_selection(self, selected_voice_name: object | None = None) -> None:
        if self._updating:
            return

        self._updating = True
        try:
            resolved_voice_name = selected_voice_name if isinstance(selected_voice_name, str) else self._controller.voice_tts_voice_name
            choices = vietnamese_tts_voice_choices_for_model(self._controller.voice_tts_model_id)
            default_voice_name = choices[0].voice_name if choices else DEFAULT_VIETNAMESE_TTS_VOICE_NAME
            resolved_voice_name = resolve_vietnamese_tts_voice_name(
                resolved_voice_name,
                choices,
                default_voice_name,
            )

            self.voice_tts_voice_combo.blockSignals(True)
            self.voice_tts_voice_combo.clear()
            for choice in choices:
                self.voice_tts_voice_combo.addItem(choice.label, choice.voice_name)
            self.voice_tts_voice_combo.setEnabled(True)
            current_index = self.voice_tts_voice_combo.findData(resolved_voice_name)
            if current_index < 0:
                current_index = 0
            self.voice_tts_voice_combo.setCurrentIndex(current_index)
            self.voice_tts_voice_combo.blockSignals(False)
        finally:
            self._updating = False

    def _sync_voice_tts_voice_selection_from_model(self, _selected_model_id: object | None = None) -> None:
        if self._updating:
            return

        SettingsPopup._sync_voice_tts_voice_selection(self, self._controller.voice_tts_voice_name)

    def _selected_language(self) -> str:
        language = self.language_input.text().strip()
        return language or DEFAULT_TARGET_LANGUAGE

    def _selected_provider_config(self) -> ProviderConfig | None:
        provider_config = self.model_combo.currentData()
        if isinstance(provider_config, ProviderConfig):
            return provider_config
        return None

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

    def _selected_voice_tts_voice_name(self) -> str | None:
        voice_name = self.voice_tts_voice_combo.currentData()
        if isinstance(voice_name, str) and voice_name:
            return voice_name
        return None

    def _apply_selected_provider(self) -> None:
        provider_config = self._selected_provider_config()
        if provider_config is None or provider_config == self._controller.provider_config:
            return

        previous_provider_config = self._controller.provider_config
        try:
            self._controller.set_provider(provider_config)
        except Exception as exc:  # noqa: BLE001 - keep the popup responsive on configuration failures
            self._set_status(f"LLM update failed: {exc}")
            self._sync_llm_selection(self._controller.provider_config)
            return

        if self._controller.provider_config != provider_config:
            self._sync_llm_selection(previous_provider_config)

    def _on_model_selected(self, _index: int) -> None:
        if self._updating:
            return

        self._apply_selected_provider()

    def _apply_selected_voice_stt_model(self) -> None:
        model_id = self._selected_voice_stt_model_id()
        if model_id is None or model_id == self._controller.voice_stt_model_id:
            return

        previous_model_id = self._controller.voice_stt_model_id
        try:
            self._controller.set_voice_stt_model_id(model_id)
        except Exception as exc:  # noqa: BLE001 - keep the popup responsive on configuration failures
            self._set_status(f"Vietnamese STT update failed: {exc}")
            self._sync_voice_stt_selection(self._controller.voice_stt_model_id)
            return

        if self._controller.voice_stt_model_id != model_id:
            self._sync_voice_stt_selection(previous_model_id)

    def _apply_selected_voice_tts_model(self) -> None:
        model_id = self._selected_voice_tts_model_id()
        if model_id is None or model_id == self._controller.voice_tts_model_id:
            return

        previous_model_id = self._controller.voice_tts_model_id
        try:
            self._controller.set_voice_tts_model_id(model_id)
        except Exception as exc:  # noqa: BLE001 - keep the popup responsive on configuration failures
            self._set_status(f"Vietnamese TTS update failed: {exc}")
            self._sync_voice_tts_selection(self._controller.voice_tts_model_id)
            return

        if self._controller.voice_tts_model_id != model_id:
            self._sync_voice_tts_selection(previous_model_id)
            return

        self._sync_voice_tts_voice_selection_from_model(self._controller.voice_tts_model_id)

    def _apply_selected_voice_tts_voice_name(self) -> None:
        voice_name = self._selected_voice_tts_voice_name()
        if voice_name is None or voice_name == self._controller.voice_tts_voice_name:
            return

        previous_voice_name = self._controller.voice_tts_voice_name
        try:
            self._controller.set_voice_tts_voice_name(voice_name)
        except Exception as exc:  # noqa: BLE001 - keep the popup responsive on configuration failures
            self._set_status(f"Vietnamese voice update failed: {exc}")
            self._sync_voice_tts_voice_selection(self._controller.voice_tts_voice_name)
            return

        if self._controller.voice_tts_voice_name != voice_name:
            self._sync_voice_tts_voice_selection(previous_voice_name)

    def _on_voice_stt_selected(self, _index: int) -> None:
        if self._updating:
            return

        self._apply_selected_voice_stt_model()

    def _on_voice_tts_selected(self, _index: int) -> None:
        if self._updating:
            return

        self._apply_selected_voice_tts_model()

    def _on_voice_tts_voice_selected(self, _index: int) -> None:
        if self._updating:
            return

        self._apply_selected_voice_tts_voice_name()

    def _on_session_selected(self, _index: int) -> None:
        if self._updating:
            return

        session_id = self.session_combo.currentData()
        if isinstance(session_id, str) and session_id:
            self._controller.select_session(session_id)

    def _on_new_session(self, _checked: bool = False) -> None:
        if self._updating:
            return
        self._controller.create_session()

    def _on_delete_session(self, _checked: bool = False) -> None:
        if self._updating:
            return

        session_id = self.session_combo.currentData()
        if isinstance(session_id, str) and session_id:
            self._controller.delete_session(session_id)

    def _on_language_changed(self) -> None:
        if self._updating:
            return
        self._controller.set_target_language(self._selected_language())

    def _on_hotkey_toggled(self, enabled: bool) -> None:
        if self._updating:
            return

        try:
            if enabled and not self._hotkey_listener.is_running:
                self._hotkey_listener.start()
            elif not enabled and self._hotkey_listener.is_running:
                self._hotkey_listener.stop()
        except Exception as exc:  # noqa: BLE001 - surface toggle failures in the popup itself
            self.hotkey_checkbox.blockSignals(True)
            self.hotkey_checkbox.setChecked(self._hotkey_listener.is_running)
            self.hotkey_checkbox.blockSignals(False)
            self._set_status(f"Hotkey toggle failed: {exc}")

    def _on_screen_ocr_toggled(self, enabled: bool) -> None:
        if self._updating:
            return

        self._controller.set_screen_ocr_enabled(enabled)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
