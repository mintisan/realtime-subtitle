import time
from ctypes import c_void_p

from PyQt6.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# macOS: Make window visible on all desktops (Spaces)
try:
    from AppKit import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary
    import objc

    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False


def _available_geometry():
    """Return a usable screen geometry even in limited/offscreen environments."""
    screen = QApplication.primaryScreen()
    if screen is not None:
        return screen.availableGeometry()
    return QRect(0, 0, 1280, 720)


class LogItem(QFrame):
    """A widget representing a single chunk of transcription/translation."""

    ORIGINAL_SECONDARY_STYLE = "color: #aaaaaa; font-family: Arial; font-size: 14px;"
    ORIGINAL_PRIMARY_STYLE = "color: #ffffff; font-family: Arial; font-size: 20px; font-weight: bold;"
    TRANSLATED_STYLE = "color: #ffffff; font-family: Arial; font-size: 20px; font-weight: bold;"

    def __init__(self, chunk_id, timestamp, original_text, translated_text=""):
        super().__init__()
        self.chunk_id = chunk_id
        self.timestamp = timestamp
        self.original_text = original_text
        self.translated_text = translated_text
        self.show_bilingual = True

        self.setStyleSheet("background-color: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 15)
        layout.setSpacing(2)

        self.original_label = QLabel()
        self.original_label.setWordWrap(True)
        layout.addWidget(self.original_label)

        self.translated_label = QLabel()
        self.translated_label.setWordWrap(True)
        self.translated_label.setStyleSheet(self.TRANSLATED_STYLE)
        layout.addWidget(self.translated_label)
        self._apply_display_mode()

    def update_translated(self, text):
        self.translated_text = text
        self._apply_display_mode()

    def update_original(self, text):
        self.original_text = text
        self.timestamp = time.strftime("%H:%M:%S")
        self._apply_display_mode()

    def set_bilingual_mode(self, enabled):
        self.show_bilingual = enabled
        self._apply_display_mode()

    def _apply_display_mode(self):
        if self.show_bilingual:
            self.original_label.setStyleSheet(self.ORIGINAL_SECONDARY_STYLE)
            self.original_label.setText(f"[{self.timestamp}] {self.original_text}")
            self.translated_label.setVisible(bool(self.translated_text))
            self.translated_label.setText(self.translated_text)
        else:
            self.original_label.setStyleSheet(self.ORIGINAL_PRIMARY_STYLE)
            self.original_label.setText(self.original_text)
            self.translated_label.setVisible(False)


class ResizeHandle(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setText("◢")
        self.setStyleSheet("color: rgba(255, 255, 255, 100); font-size: 16px;")
        self.setFixedSize(20, 20)
        self.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.startPos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.startPos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.startPos:
            delta = event.globalPosition().toPoint() - self.startPos
            new_width = max(self.parent_window.minimumWidth(), self.parent_window.width() + delta.x())
            new_height = max(self.parent_window.minimumHeight(), self.parent_window.height() + delta.y())

            self.parent_window.resize(new_width, new_height)
            self.startPos = event.globalPosition().toPoint()
            event.accept()

    def mouseReleaseEvent(self, event):
        self.startPos = None


class OverlayWindow(QWidget):
    stop_requested = pyqtSignal()
    bilingual_toggled = pyqtSignal(bool)
    MIN_HEIGHT = 280

    def __init__(self, display_duration=None, window_width=400, window_height=None):
        super().__init__()
        self.window_width = window_width

        screen_geometry = _available_geometry()
        requested_height = window_height if window_height else screen_geometry.height()
        self.window_height = max(requested_height, self.MIN_HEIGHT)

        self.items = []
        self.transcript_data = {}
        self.is_moving = False
        self.show_bilingual = True

        self.initUI()
        self.oldPos = self.pos()

    def showEvent(self, event):
        """Called when window is shown - set all-spaces behavior here."""
        super().showEvent(event)
        if HAS_APPKIT:
            self._set_all_spaces()

    def _set_all_spaces(self):
        """Make window appear on all macOS Spaces/Desktops."""
        try:
            win_id = int(self.winId())
            ns_view = objc.objc_object(c_void_p=c_void_p(win_id))
            ns_window = ns_view.window()
            ns_window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
            )
            print("Window set to appear on all Spaces")
        except Exception as e:
            print(f"Could not set all-spaces behavior: {e}")

    def initUI(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setMinimumHeight(self.MIN_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                width: 0px;
            }
            """
        )
        self.scroll_area.viewport().setStyleSheet("background: transparent;")

        self.container = QFrame()
        self.container.setObjectName("overlayContainer")
        self.container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.container.setStyleSheet(
            """
            QFrame#overlayContainer {
                background-color: rgba(10, 10, 10, 210);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 10px;
            }
            """
        )

        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)

        grip_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFixedWidth(80)
        self.save_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(255, 255, 255, 50);
                color: white;
                border-radius: 5px;
                padding: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 100);
            }
            """
        )
        self.save_btn.clicked.connect(self._save_transcript)
        grip_layout.addWidget(self.save_btn)

        self.bilingual_checkbox = QCheckBox("中英同显")
        self.bilingual_checkbox.setChecked(True)
        self.bilingual_checkbox.setStyleSheet(
            """
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: rgba(255, 255, 255, 30);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: rgba(166, 227, 161, 180);
                border: 1px solid rgba(166, 227, 161, 220);
                border-radius: 4px;
            }
            """
        )
        self.bilingual_checkbox.toggled.connect(self._set_bilingual_mode)
        grip_layout.addWidget(self.bilingual_checkbox)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setToolTip("Stop Translator")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setFixedSize(30, 30)
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(243, 139, 168, 150);
                color: white;
                border-radius: 15px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(243, 139, 168, 200);
            }
            """
        )
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        grip_layout.addWidget(self.stop_btn)

        grip_layout.addStretch()
        self.grip_label = ResizeHandle(self)
        grip_layout.addWidget(self.grip_label)
        layout.addLayout(grip_layout)

        self.resize(self.window_width, self.window_height)

        screen = _available_geometry()
        x = screen.x() + screen.width() - self.window_width - 20
        y = screen.y()
        self.move(x, y)
        print(f"[Overlay] Window geometry: x={x}, y={y}, width={self.window_width}, height={self.window_height}")

    def update_text(self, chunk_id, original_text, translated_text):
        """Append new text or update existing text."""
        print(f"[Overlay] Received update for #{chunk_id}: {original_text} -> {translated_text}")

        if chunk_id not in self.transcript_data:
            self.transcript_data[chunk_id] = {
                "timestamp": time.strftime("%H:%M:%S"),
                "original": original_text,
                "translated": translated_text,
            }
        else:
            if original_text:
                self.transcript_data[chunk_id]["original"] = original_text
            if translated_text:
                self.transcript_data[chunk_id]["translated"] = translated_text

        existing_widget = None
        for cid, widget in self.items:
            if cid == chunk_id:
                existing_widget = widget
                break

        if existing_widget:
            if original_text:
                existing_widget.update_original(original_text)
            if translated_text:
                existing_widget.update_translated(translated_text)
            existing_widget.set_bilingual_mode(self.show_bilingual)
            print(f"[Overlay] Updated existing widget #{chunk_id}")
        else:
            timestamp = self.transcript_data[chunk_id]["timestamp"]
            new_widget = LogItem(chunk_id, timestamp, original_text, translated_text)
            new_widget.set_bilingual_mode(self.show_bilingual)

            insert_idx = len(self.items)
            for i, (cid, _) in enumerate(self.items):
                if cid > chunk_id:
                    insert_idx = i
                    break

            self.items.insert(insert_idx, (chunk_id, new_widget))
            self.container_layout.insertWidget(insert_idx, new_widget)
            print(f"[Overlay] Inserted new widget #{chunk_id} at index {insert_idx}")

        self._refresh_layout()
        QTimer.singleShot(10, self._scroll_to_bottom)

    def _set_bilingual_mode(self, enabled):
        self.show_bilingual = enabled
        for chunk_id, widget in self.items:
            if not enabled and widget.translated_text == "(translating...)":
                widget.update_translated("")
                if chunk_id in self.transcript_data:
                    self.transcript_data[chunk_id]["translated"] = ""
            widget.set_bilingual_mode(enabled)
        self._refresh_layout()
        self.bilingual_toggled.emit(enabled)
        print(f"[Overlay] Bilingual mode: {'on' if enabled else 'off'}")

    def _refresh_layout(self):
        """Force a repaint for translucent macOS overlays."""
        self.container.adjustSize()
        self.container.updateGeometry()
        self.scroll_area.viewport().update()
        self.scroll_area.update()
        self.update()

    def _scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _save_transcript(self):
        """Save history to file."""
        import os

        if not self.transcript_data:
            print("[Overlay] Nothing to save.")
            return

        os.makedirs("transcripts", exist_ok=True)
        filename = f"transcripts/transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        sorted_ids = sorted(self.transcript_data.keys())

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Transcript saved at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                for cid in sorted_ids:
                    data = self.transcript_data[cid]
                    f.write(
                        f"[{data['timestamp']}] (ID: {cid})\n"
                        f"Original: {data['original']}\n"
                        f"Translation: {data['translated']}\n"
                        f"{'-' * 30}\n"
                    )

            print(f"[Overlay] Saved to {filename}")
            original_text = self.save_btn.text()
            self.save_btn.setText("Saved!")
            QTimer.singleShot(2000, lambda: self.save_btn.setText(original_text))
        except Exception as e:
            print(f"[Overlay] Error saving transcript: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_moving = True
            self.oldPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.is_moving:
            delta = event.globalPosition().toPoint() - self.oldPos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.is_moving = False
        self.setCursor(Qt.CursorShape.ArrowCursor)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = OverlayWindow(window_height=300)
    window.show()
    window.update_text(1, "Hello world", "")
    QTimer.singleShot(1000, lambda: window.update_text(1, "Hello world", "你好，世界"))
    window.update_text(2, "Sequence test", "")
    sys.exit(app.exec())
