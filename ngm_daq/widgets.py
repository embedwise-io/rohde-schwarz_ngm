"""Small Qt widgets for instrument controls."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt, QVariantAnimation, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton, QDoubleSpinBox, QSizePolicy

SWITCH_ON = QColor("#2e7d32")
SWITCH_OFF = QColor("#757575")
SWITCH_DISABLED = QColor("#c4c4c4")


class ToggleSwitch(QAbstractButton):
    """A two-position slide switch labelled OFF and ON. Click or press Space to flip it.

    Like any checkable button it emits `clicked(bool)` when the user flips it and `toggled(bool)`
    whenever its state changes, including from setChecked().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._position = 0.0                      # knob: 0 = OFF (left) ... 1 = ON (right)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(120)
        self._animation.valueChanged.connect(self._move)
        self.toggled.connect(self._slide)

    def _label_font(self) -> QFont:
        font = QFont(self.font())
        font.setBold(True)
        return font

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self._label_font())
        height = fm.height() + 8
        text = max(fm.horizontalAdvance("OFF"), fm.horizontalAdvance("ON"))
        return QSize(text + height + 14, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _slide(self, on: bool) -> None:
        self._animation.stop()
        if not self.isVisible():
            self._move(1.0 if on else 0.0)
            return
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(1.0 if on else 0.0)
        self._animation.start()

    def _move(self, value) -> None:
        self._position = float(value)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = r.height() / 2
        t = self._position
        if self.isEnabled():
            track = QColor.fromRgbF(*(a + (b - a) * t for a, b in zip(SWITCH_OFF.getRgbF()[:3],
                                                                         SWITCH_ON.getRgbF()[:3])))
        else:
            track = SWITCH_DISABLED
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, radius, radius)

        d = r.height() - 6
        knob = QRectF(r.left() + 3 + t * (r.width() - d - 6), r.top() + 3, d, d)
        p.setFont(self._label_font())
        p.setPen(QColor("white"))
        if t >= 0.5:
            p.drawText(QRectF(r.left() + 4, r.top(), knob.left() - r.left() - 4, r.height()),
                       Qt.AlignmentFlag.AlignCenter, "ON")
        else:
            p.drawText(QRectF(knob.right(), r.top(), r.right() - knob.right() - 4, r.height()),
                       Qt.AlignmentFlag.AlignCenter, "OFF")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("white"))
        p.drawEllipse(knob)
        if self.hasFocus():
            p.setPen(QPen(self.palette().highlight().color(), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


class SetpointSpinBox(QDoubleSpinBox):
    """Spin box for an instrument setpoint.

    `committed` fires on Enter or when leaving the field, and only if the user changed the value,
    so a value that was merely displayed from the instrument is never sent back to it. show_value()
    displays the instrument's value unless the user is editing; set_unknown() shows a dash. The
    mouse wheel only acts while the field has focus, so scrolling past it can't change a setpoint.
    """

    committed = Signal(float)

    def __init__(self, suffix: str, decimals: int, lo: float, hi: float, step: float):
        super().__init__()
        self.setDecimals(decimals)
        self.setRange(lo, hi)
        self.setSingleStep(step)
        self.setSuffix(suffix)
        self.setKeyboardTracking(False)
        self.setAccelerated(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._lowest = lo
        self._edited = False
        self.lineEdit().textEdited.connect(self._mark_edited)    # typing
        self.valueChanged.connect(self._mark_edited)             # arrows, wheel; not show_value()
        self.editingFinished.connect(self._commit)

    @property
    def edited(self) -> bool:
        return self._edited

    def _mark_edited(self, *_) -> None:
        self._edited = True

    def _commit(self) -> None:
        if self._edited:
            self._edited = False
            self.committed.emit(self.value())

    def show_value(self, value: float) -> None:
        """Display a value from the instrument, unless the user has an uncommitted edit."""
        if self._edited:
            return
        self.blockSignals(True)
        self.setSpecialValueText("")
        self.setMinimum(self._lowest)
        self.setValue(value)
        self.blockSignals(False)

    def set_unknown(self) -> None:
        """Show '—' (no value known) and drop any uncommitted edit."""
        self._edited = False
        self.blockSignals(True)
        self.setMinimum(self._lowest - self.singleStep())
        self.setSpecialValueText("—")
        self.setValue(self.minimum())
        self.blockSignals(False)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
