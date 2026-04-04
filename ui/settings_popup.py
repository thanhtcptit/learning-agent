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
from prompts.templates import DEFAULT_TARGET_LANGUAGE, PromptMode


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

        self.model_label = QLabel("LLM name")
        self.model_label.setObjectName("PopupFieldLabel")
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("PopupModelCombo")

        self.provider_row = QWidget()
        self.provider_row.setObjectName("PopupProviderRow")
        provider_row_layout = QHBoxLayout(self.provider_row)
        provider_row_layout.setContentsMargins(0, 0, 0, 0)
        provider_row_layout.setSpacing(8)
        self.provider_label = QLabel("Provider")
        self.provider_label.setObjectName("PopupFieldLabel")
        self.provider_combo = QComboBox()
        self.provider_combo.setObjectName("PopupProviderCombo")
        provider_row_layout.addWidget(self.provider_label)
        provider_row_layout.addWidget(self.provider_combo, 1)
        self.provider_row.setVisible(False)
        self.provider_row.setEnabled(False)

        llm_layout = QVBoxLayout()
        llm_layout.setSpacing(6)
        llm_layout.addWidget(self.model_label)
        llm_layout.addWidget(self.model_combo)
        llm_layout.addWidget(self.provider_row)

        self.hotkey_checkbox = QCheckBox("Enable global hotkeys")
        self.hotkey_checkbox.setObjectName("PopupHotkeyToggle")

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

        self.mode_combo = QComboBox()
        for mode in PromptMode:
            self.mode_combo.addItem(mode.label, mode)

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
        card_layout.addLayout(session_row)
        card_layout.addWidget(self.mode_combo)
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
            QComboBox#PopupModelCombo, QComboBox#PopupProviderCombo {
                min-width: 0px;
            }
            QCheckBox#PopupHotkeyToggle {
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
        self._controller.preferred_language_changed.connect(self._sync_language_input)
        self._controller.current_language_changed.connect(self._sync_current_language_status)
        self._controller.status_changed.connect(self._set_status)
        self._hotkey_listener.status_changed.connect(self._set_status)

        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_selected)
        self.hotkey_checkbox.toggled.connect(self._on_hotkey_toggled)
        self.session_combo.currentIndexChanged.connect(self._on_session_selected)
        self.new_session_button.clicked.connect(self._on_new_session)
        self.delete_session_button.clicked.connect(self._on_delete_session)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_selected)
        self.language_input.editingFinished.connect(self._on_language_changed)

    def _apply_state(self) -> None:
        self._sync_sessions(self._controller.sessions)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(self._controller.default_mode))
        self.language_input.setText(self._controller.preferred_language)
        self._sync_current_language_status(self._controller.target_language)
        self.hotkey_checkbox.blockSignals(True)
        self.hotkey_checkbox.setChecked(self._hotkey_listener.is_running)
        self.hotkey_checkbox.blockSignals(False)
        self._sync_llm_selection(self._controller.provider_config)

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

            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(self._controller.default_mode))
            self.mode_combo.blockSignals(False)

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
        self.status_label.setText(f"Current language: {language}")

    def _refresh_catalog(self) -> None:
        self._llm_entries = discover_llm_catalog()

    def _find_entry_for_config(self, provider_config: ProviderConfig) -> LLMModelEntry | None:
        for entry in self._llm_entries:
            if entry.family == provider_config.family and entry.name == provider_config.name:
                return entry

        for entry in self._llm_entries:
            if entry.name == provider_config.name:
                return entry

        if provider_config.display_name:
            for entry in self._llm_entries:
                if entry.display_name == provider_config.display_name:
                    return entry

        return None

    def _sync_llm_selection(
        self,
        selected_config: object | None = None,
        *,
        selected_model: LLMModelEntry | None = None,
        selected_provider_name: str | None = None,
    ) -> None:
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

                self.provider_combo.blockSignals(True)
                self.provider_combo.clear()
                self.provider_combo.addItem("No providers found", None)
                self.provider_combo.setEnabled(False)
                self.provider_combo.blockSignals(False)

                self.provider_row.setVisible(False)
                self._set_status("No LLM configs found")
                return

            provider_config = selected_config if isinstance(selected_config, ProviderConfig) else None
            resolved_model = None
            if selected_model is not None:
                resolved_model = next((entry for entry in self._llm_entries if entry == selected_model), None)
            if resolved_model is None and provider_config is not None:
                resolved_model = self._find_entry_for_config(provider_config)
            if resolved_model is None:
                current_model = self._selected_model()
                if current_model is not None:
                    resolved_model = next((entry for entry in self._llm_entries if entry == current_model), None)
            if resolved_model is None:
                resolved_model = self._llm_entries[0]

            provider_options = list(resolved_model.providers)
            if not provider_options:
                self.model_combo.setEnabled(False)
                self.provider_combo.setEnabled(False)
                self.provider_row.setVisible(False)
                self._set_status("Selected LLM has no providers")
                return

            if selected_provider_name is None:
                selected_provider_name = self._selected_provider_name()
            if selected_provider_name is None and provider_config is not None:
                selected_provider_name = provider_config.provider

            provider_names = {option.provider for option in provider_options}
            if selected_provider_name not in provider_names:
                selected_provider_name = provider_options[0].provider

            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            for entry in self._llm_entries:
                self.model_combo.addItem(entry.display_name, entry)
            self.model_combo.setEnabled(True)
            model_index = self.model_combo.findData(resolved_model)
            if model_index < 0:
                model_index = 0
            self.model_combo.setCurrentIndex(model_index)
            self.model_combo.blockSignals(False)

            self.provider_combo.blockSignals(True)
            self.provider_combo.clear()
            for option in provider_options:
                self.provider_combo.addItem(option.provider, option)
            provider_index = next(
                (index for index, option in enumerate(provider_options) if option.provider == selected_provider_name),
                0,
            )
            self.provider_combo.setCurrentIndex(provider_index)
            self.provider_combo.setEnabled(True)
            self.provider_combo.blockSignals(False)

            self.provider_row.setVisible(True)
            self.provider_row.setEnabled(True)
        finally:
            self._updating = False

    def _selected_mode(self) -> PromptMode:
        mode = self.mode_combo.currentData()
        if isinstance(mode, PromptMode):
            return mode
        return self._controller.default_mode

    def _selected_language(self) -> str:
        language = self.language_input.text().strip()
        return language or DEFAULT_TARGET_LANGUAGE

    def _selected_model(self) -> LLMModelEntry | None:
        model = self.model_combo.currentData()
        if isinstance(model, LLMModelEntry):
            return model
        return None

    def _selected_provider_name(self) -> str | None:
        provider_config = self.provider_combo.currentData()
        if isinstance(provider_config, ProviderConfig):
            return provider_config.provider
        return None

    def _selected_provider_config(self) -> ProviderConfig | None:
        provider_config = self.provider_combo.currentData()
        if isinstance(provider_config, ProviderConfig):
            return provider_config
        return None

    def _apply_selected_provider(self) -> None:
        provider_config = self._selected_provider_config()
        if provider_config is None or provider_config == self._controller.provider_config:
            return

        try:
            self._controller.set_provider(provider_config)
        except Exception as exc:  # noqa: BLE001 - keep the popup responsive on configuration failures
            self._set_status(f"LLM update failed: {exc}")
            self._sync_llm_selection(self._controller.provider_config)

    def _on_model_selected(self, _index: int) -> None:
        if self._updating:
            return

        self._sync_llm_selection(
            selected_model=self._selected_model(),
            selected_provider_name=self._selected_provider_name(),
        )
        self._apply_selected_provider()

    def _on_provider_selected(self, _index: int) -> None:
        if self._updating:
            return
        self._apply_selected_provider()

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

    def _on_mode_selected(self, _index: int) -> None:
        if self._updating:
            return
        self._controller.set_default_mode(self._selected_mode())

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

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
