"""Two official brand assets in a keyboard-accessible, click-to-switch control."""
from pathlib import Path
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QToolButton


class BrandLogoButton(QToolButton):
    def __init__(self, resource_dir, parent=None):
        super().__init__(parent)
        base = Path(resource_dir)
        self._brands = [('ShanghaiTech', base / 'shanghaitech.svg'),
                        ('UIH', base / 'logo.jpg')]
        self._index = 0
        self.setObjectName('brandLogoButton')
        self.setFixedSize(202, 100)
        self.setIconSize(QSize(190, 88))
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.clicked.connect(self.toggle_brand)
        self._render()

    @property
    def brand(self):
        return self._brands[self._index][0]

    def toggle_brand(self):
        self._index = (self._index + 1) % len(self._brands)
        self._render()

    def set_brand(self, name):
        for index, (brand, _) in enumerate(self._brands):
            if name == brand:
                self._index = index
                self._render()
                return
        raise ValueError(f'Unknown brand: {name}')

    def _render(self):
        brand, path = self._brands[self._index]
        # White backing preserves the official brand colors in both UI themes.
        self.setStyleSheet('QToolButton#brandLogoButton {background: white; border: 1px solid #d2d2d7; border-radius: 8px; padding: 3px;} QToolButton#brandLogoButton:hover, QToolButton#brandLogoButton:focus {border: 1px solid #397aad;}')
        canvas = QPixmap(380, 176)
        canvas.fill(Qt.white)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if path.suffix.lower() == '.svg':
            renderer = QSvgRenderer(str(path))
            if renderer.isValid():
                size = renderer.defaultSize()
                size.scale(364, 164, Qt.KeepAspectRatio)
                target = QRectF((380-size.width())/2,
                    (176-size.height())/2, size.width(), size.height())
                # The supplied cropped SVG retains paths outside its viewBox.
                # QtSvg requires an explicit clip to honor that crop.
                painter.save()
                painter.setClipRect(target)
                renderer.render(painter, target)
                painter.restore()
            else:
                painter.drawText(canvas.rect(), Qt.AlignCenter, brand)
        else:
            image = QPixmap(str(path))
            if not image.isNull():
                image = image.scaled(364, 164, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap((380-image.width())//2, (176-image.height())//2, image)
            else:
                painter.drawText(canvas.rect(), Qt.AlignCenter, brand)
        painter.end()
        self.setIcon(QIcon(canvas))
        other = self._brands[(self._index+1) % len(self._brands)][0]
        self.setToolTip(f'{brand} — click to switch to {other}')
        self.setAccessibleName(f'{brand} logo; switch to {other}')

    def refresh(self):
        self._render()
