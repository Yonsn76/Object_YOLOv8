import sys
import os
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QFrame, QFileDialog, QComboBox,
    QStyle, QToolBar, QMessageBox, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction, QIcon, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, pyqtSlot
from ultralytics import YOLO

# --- Hilo para el procesamiento de Medios (Cámara o Video) ---
class MediaProcessingThread(QThread):
    frame_ready = pyqtSignal(QPixmap)
    status_update = pyqtSignal(str) # Solo emite el string del mensaje
    processing_finished = pyqtSignal()

    def __init__(self, yolo_model, source_type="webcam", file_path=None):
        super().__init__()
        self.yolo_model = yolo_model
        self.source_type = source_type
        self.file_path = file_path
        self._is_running = False
        self._is_paused = False
        self.cap = None

    def run(self):
        self._is_running = True
        self._is_paused = False

        if self.source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_update.emit("Error: No se pudo abrir la cámara.")
                self._is_running = False
        elif self.source_type == "video":
            if not self.file_path:
                self.status_update.emit("Error: No se proporcionó ruta de video.")
                self._is_running = False
            else:
                self.cap = cv2.VideoCapture(self.file_path)
                if not self.cap.isOpened():
                    self.status_update.emit(f"Error: No se pudo abrir el video: {self.file_path.split('/')[-1]}")
                    self._is_running = False
        else:
            self.status_update.emit("Error: Tipo de fuente no reconocido.")
            self._is_running = False

        if not self._is_running:
            self.processing_finished.emit()
            return

        if self.source_type == "webcam":
             self.status_update.emit("Cámara iniciada. Detectando...")
        elif self.source_type == "video":
            self.status_update.emit(f"Procesando video: {self.file_path.split('/')[-1]}")

        while self._is_running:
            if self._is_paused:
                QThread.msleep(100)
                continue

            if self.cap and self.cap.isOpened():
                ret, frame_cv = self.cap.read()
                if not ret:
                    if self.source_type == "video":
                        self.status_update.emit("Video finalizado.") # Solo el mensaje
                    else:
                        self.status_update.emit("Error al leer fotograma. Intentando reconectar...")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(0)
                        if not self.cap.isOpened():
                            self.status_update.emit("Fallo al reconectar la cámara.")
                            self._is_running = False
                        time.sleep(0.5)
                    break
            else:
                self.status_update.emit("Fuente de video no disponible.")
                break

            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(frame_rgb, verbose=False)

            for r in results:
                boxes, names = r.boxes, r.names
                if boxes is not None:
                    for i in range(len(boxes)):
                        box_coords = boxes[i].xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(boxes[i].cls.cpu().numpy()[0])
                        conf = float(boxes[i].conf.cpu().numpy()[0])
                        label = f"{names[cls_id]}: {conf:.2f}"
                        cv2.rectangle(frame_cv, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 180, 0), 2) # Color verde más visible
                        cv2.putText(frame_cv, label, (box_coords[0], box_coords[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2) # Ligeramente más pequeño

            rgb_image_display = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image_display.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(QPixmap.fromImage(qt_image))

            if self.source_type == "webcam":
                 QThread.msleep(10)

        if self.cap:
            self.cap.release()
        self._is_running = False
        self.processing_finished.emit()

    def stop(self):
        self._is_running = False

    def toggle_pause(self):
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.status_update.emit("Procesamiento pausado.")
        else:
            self.status_update.emit("Procesamiento reanudado.")
        return self._is_paused

# --- Ventana Principal ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Vision")
        self.setGeometry(50, 50, 1200, 800)

        # Determinar si usar tema oscuro o claro
        self.dark_mode = True
        self.setStyleSheet(self._get_app_style_dark() if self.dark_mode else self._get_app_style_light())

        self.yolo_model = None
        self.media_thread = None
        self.current_media_path = None
        self.current_source_type = None

        self._init_ui()
        self._load_yolo_model_async()

    def _get_app_style_dark(self):
        # Estilo QSS para un look minimalista oscuro
        return """
            QMainWindow {
                background-color: #121212; /* Fondo oscuro */
                color: #E0E0E0;
            }
            QWidget {
                color: #E0E0E0;
            }
            QLabel#VideoLabel {
                background-color: #1E1E1E; /* Gris oscuro */
                border: 1px solid #333333;
                border-radius: 8px;
                color: #E0E0E0;
                padding: 2px;
            }
            QPushButton {
                background-color: #2A2A2A;
                color: #00E5FF; /* Cyan */
                border: none;
                padding: 10px 20px;
                font-size: 10pt;
                border-radius: 4px;
                font-weight: 500;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #00B8D4; /* Cyan más oscuro */
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #555555;
            }
            QComboBox {
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-height: 20px;
                selection-background-color: #00B8D4;
            }
            QComboBox::drop-down {
                border: none;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                color: #E0E0E0;
                selection-background-color: #00B8D4;
                selection-color: #FFFFFF;
                border: none;
            }
            QToolBar {
                background-color: #1A1A1A;
                border: none;
                border-bottom: 1px solid #333333;
                padding: 6px;
                spacing: 10px;
            }
            QToolBar QToolButton {
                background-color: transparent;
                color: #00E5FF;
                padding: 8px;
                margin: 2px;
                border-radius: 4px;
            }
            QToolBar QToolButton:hover {
                background-color: #2A2A2A;
            }
            QToolBar QToolButton:pressed {
                background-color: #00B8D4;
            }
            QToolBar QToolButton:disabled {
                color: #555555;
                background-color: transparent;
            }
            QStatusBar {
                background-color: #1A1A1A;
                color: #B0B0B0;
                font-size: 9pt;
                border-top: 1px solid #333333;
            }
            QMenuBar {
                background-color: #1A1A1A;
                color: #E0E0E0;
                border-bottom: 1px solid #333333;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #2A2A2A;
                color: #00E5FF;
            }
            QMenuBar::item:pressed {
                background-color: #00B8D4;
                color: #FFFFFF;
            }
            QMenu {
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 25px 8px 20px;
                border-radius: 3px;
                margin: 2px;
            }
            QMenu::item:selected {
                background-color: #00B8D4;
                color: #FFFFFF;
            }
            QMenu::separator {
                height: 1px;
                background: #333333;
                margin: 5px 10px;
            }
            QScrollBar:vertical {
                background-color: #1A1A1A;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #3A3A3A;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00B8D4;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #1A1A1A;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #3A3A3A;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #00B8D4;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QMessageBox {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            QMessageBox QLabel {
                color: #E0E0E0;
            }
            QFileDialog {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
        """

    def _get_app_style_light(self):
        # Estilo QSS para un look minimalista claro
        return """
            QMainWindow {
                background-color: #F5F5F5; /* Gris muy claro de fondo */
                color: #212121;
            }
            QWidget {
                color: #212121;
            }
            QLabel#VideoLabel {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                color: #212121;
                padding: 2px;
            }
            QPushButton {
                background-color: #FFFFFF;
                color: #00ACC1; /* Cyan */
                border: none;
                padding: 10px 20px;
                font-size: 10pt;
                border-radius: 4px;
                font-weight: 500;
                min-height: 20px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #F5F5F5;
                color: #00838F;
            }
            QPushButton:pressed {
                background-color: #E0E0E0;
                color: #006064;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
            }
            QComboBox {
                background-color: #FFFFFF;
                color: #212121;
                border: 1px solid #E0E0E0;
                padding: 8px;
                border-radius: 4px;
                min-height: 20px;
                selection-background-color: #00ACC1;
            }
            QComboBox::drop-down {
                border: none;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #212121;
                selection-background-color: #00ACC1;
                selection-color: #FFFFFF;
                border: 1px solid #E0E0E0;
            }
            QToolBar {
                background-color: #FFFFFF;
                border: none;
                border-bottom: 1px solid #E0E0E0;
                padding: 6px;
                spacing: 10px;
            }
            QToolBar QToolButton {
                background-color: transparent;
                color: #00ACC1;
                padding: 8px;
                margin: 2px;
                border-radius: 4px;
            }
            QToolBar QToolButton:hover {
                background-color: #F5F5F5;
            }
            QToolBar QToolButton:pressed {
                background-color: #E0E0E0;
            }
            QToolBar QToolButton:disabled {
                color: #BDBDBD;
                background-color: transparent;
            }
            QStatusBar {
                background-color: #FFFFFF;
                color: #757575;
                font-size: 9pt;
                border-top: 1px solid #E0E0E0;
            }
            QMenuBar {
                background-color: #FFFFFF;
                color: #212121;
                border-bottom: 1px solid #E0E0E0;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #F5F5F5;
                color: #00ACC1;
            }
            QMenuBar::item:pressed {
                background-color: #E0E0E0;
                color: #00838F;
            }
            QMenu {
                background-color: #FFFFFF;
                color: #212121;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 25px 8px 20px;
                border-radius: 3px;
                margin: 2px;
            }
            QMenu::item:selected {
                background-color: #F5F5F5;
                color: #00ACC1;
            }
            QMenu::separator {
                height: 1px;
                background: #E0E0E0;
                margin: 5px 10px;
            }
            QScrollBar:vertical {
                background-color: #F5F5F5;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #BDBDBD;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00ACC1;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #F5F5F5;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #BDBDBD;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #00ACC1;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QMessageBox {
                background-color: #FFFFFF;
                color: #212121;
            }
            QMessageBox QLabel {
                color: #212121;
            }
            QFileDialog {
                background-color: #FFFFFF;
                color: #212121;
            }
        """

    def _init_ui(self):
        self._create_menu_bar()
        self._create_tool_bar() # La toolbar ahora va primero para que el estilo se aplique bien
        self._create_central_widget()
        self._create_status_bar()

    def _load_yolo_model_async(self):
        self.status_bar.showMessage("Cargando modelo YOLOv8n, por favor espera...")
        QTimer.singleShot(100, self._perform_model_load)

    def _perform_model_load(self):
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.status_bar.showMessage("Modelo YOLOv8n cargado. Sistema listo.", 5000) # Uso directo
            self._update_button_states()
            print("Modelo YOLOv8n cargado.")
        except Exception as e:
            self.status_bar.showMessage(f"Error crítico al cargar modelo YOLO: {e}") # Uso directo
            QMessageBox.critical(self, "Error de Modelo", f"No se pudo cargar el modelo YOLOv8n:\n{e}")
            self.open_image_action.setEnabled(False)
            self.open_video_action.setEnabled(False)
            self.start_webcam_action.setEnabled(False)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&Archivo")

        style = QApplication.style()
        self.open_image_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Abrir Imagen...", self)
        self.open_image_action.triggered.connect(self._select_image_file)
        file_menu.addAction(self.open_image_action)

        self.open_video_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DriveDVDIcon), "Abrir Video...", self) # Icono diferente
        self.open_video_action.triggered.connect(self._select_video_file)
        file_menu.addAction(self.open_video_action)

        file_menu.addSeparator()
        exit_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),"&Salir", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        camera_menu = menu_bar.addMenu("&Cámara")
        self.start_webcam_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Iniciar Cámara Web", self) # Icono más genérico
        self.start_webcam_action.triggered.connect(self._start_webcam_mode)
        camera_menu.addAction(self.start_webcam_action)

    def _create_tool_bar(self):
        # Crear una barra de herramientas más minimalista
        tool_bar = QToolBar("Herramientas")
        tool_bar.setIconSize(QSize(24, 24))  # Iconos ligeramente más grandes
        tool_bar.setMovable(False)
        tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)  # Texto bajo el icono
        tool_bar.setObjectName("MainToolBar")

        # Aplicar estilo específico para la barra de herramientas
        if self.dark_mode:
            tool_bar.setStyleSheet("""
                QToolBar#MainToolBar {
                    background-color: #1A1A1A;
                    border: none;
                    border-bottom: 1px solid #333333;
                    padding: 8px;
                    spacing: 15px;
                }
                QToolButton {
                    background-color: transparent;
                    color: #E0E0E0;
                    border: none;
                    border-radius: 6px;
                    padding: 8px;
                    margin: 2px;
                    font-size: 9pt;
                }
                QToolButton:hover {
                    background-color: #2A2A2A;
                    color: #00E5FF;
                }
                QToolButton:pressed {
                    background-color: #00B8D4;
                    color: #FFFFFF;
                }
                QToolButton:disabled {
                    color: #555555;
                }
            """)
        else:
            tool_bar.setStyleSheet("""
                QToolBar#MainToolBar {
                    background-color: #FFFFFF;
                    border: none;
                    border-bottom: 1px solid #E0E0E0;
                    padding: 8px;
                    spacing: 15px;
                }
                QToolButton {
                    background-color: transparent;
                    color: #212121;
                    border: none;
                    border-radius: 6px;
                    padding: 8px;
                    margin: 2px;
                    font-size: 9pt;
                }
                QToolButton:hover {
                    background-color: #F5F5F5;
                    color: #00ACC1;
                }
                QToolButton:pressed {
                    background-color: #E0E0E0;
                    color: #00838F;
                }
                QToolButton:disabled {
                    color: #BDBDBD;
                }
            """)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tool_bar)

        # Crear iconos personalizados para un look más moderno
        # Usar iconos del sistema pero con nombres más descriptivos
        style = QApplication.style()

        # Acción para abrir imagen
        self.open_image_action.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogStart))
        self.open_image_action.setText("Imagen")
        self.open_image_action.setToolTip("Abrir archivo de imagen")
        tool_bar.addAction(self.open_image_action)

        # Acción para abrir video
        self.open_video_action.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.open_video_action.setText("Video")
        self.open_video_action.setToolTip("Abrir archivo de video")
        tool_bar.addAction(self.open_video_action)

        # Acción para iniciar cámara
        self.start_webcam_action.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.start_webcam_action.setText("Cámara")
        self.start_webcam_action.setToolTip("Iniciar detección desde cámara web")
        tool_bar.addAction(self.start_webcam_action)

        # Separador
        tool_bar.addSeparator()

        # Acción para reproducir/pausar
        self.play_pause_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPause), "Pausar", self)
        self.play_pause_action.setToolTip("Reproducir/Pausar")
        self.play_pause_action.triggered.connect(self._toggle_play_pause_media)
        tool_bar.addAction(self.play_pause_action)

        # Acción para detener
        self.stop_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Detener", self)
        self.stop_action.setToolTip("Detener procesamiento actual")
        self.stop_action.triggered.connect(self._stop_current_media)
        tool_bar.addAction(self.stop_action)

        # Añadir un espaciador para alinear a la derecha
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tool_bar.addWidget(spacer)

        # Botón para alternar entre modo claro y oscuro
        self.toggle_theme_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton),
                                          "Tema", self)
        self.toggle_theme_action.setToolTip("Cambiar tema claro/oscuro")
        self.toggle_theme_action.triggered.connect(self._toggle_theme)
        tool_bar.addAction(self.toggle_theme_action)

        self._update_button_states()

    def _create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)  # Más margen para un look minimalista
        main_layout.setSpacing(20)  # Mayor espaciado entre elementos

        # Crear un contenedor para el video con sombra
        video_container = QFrame()
        video_container.setObjectName("VideoContainer")
        video_container.setStyleSheet("""
            QFrame#VideoContainer {
                background-color: transparent;
                border-radius: 10px;
            }
        """)

        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Mensaje de bienvenida más minimalista
        welcome_message = "YOLO Vision" if self.dark_mode else "YOLO Vision"
        self.video_label = QLabel(welcome_message)
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        # Fuente personalizada para el mensaje inicial
        font = QFont("Segoe UI", 24)  # Fuente moderna
        font.setWeight(QFont.Weight.Light)  # Peso ligero para un look minimalista
        self.video_label.setFont(font)

        # Añadir un icono o logo
        if self.dark_mode:
            self.video_label.setStyleSheet("""
                QLabel#VideoLabel {
                    background-color: #1E1E1E;
                    border: 1px solid #333333;
                    border-radius: 10px;
                    color: #00E5FF;
                    padding: 20px;
                }
            """)
        else:
            self.video_label.setStyleSheet("""
                QLabel#VideoLabel {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 10px;
                    color: #00ACC1;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
            """)

        video_layout.addWidget(self.video_label)
        main_layout.addWidget(video_container, 1)

        # Añadir un panel de información minimalista
        info_panel = QFrame()
        info_panel.setObjectName("InfoPanel")
        info_panel.setMaximumHeight(40)

        info_layout = QHBoxLayout(info_panel)
        info_layout.setContentsMargins(10, 5, 10, 5)

        # Etiqueta para mostrar información adicional
        self.info_label = QLabel("Seleccione una fuente o inicie la cámara")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        info_layout.addWidget(self.info_label)

        # Estilo del panel de información
        if self.dark_mode:
            info_panel.setStyleSheet("""
                QFrame#InfoPanel {
                    background-color: #1A1A1A;
                    border-radius: 5px;
                    border: 1px solid #333333;
                }
                QLabel#InfoLabel {
                    color: #B0B0B0;
                    font-size: 10pt;
                }
            """)
        else:
            info_panel.setStyleSheet("""
                QFrame#InfoPanel {
                    background-color: #FFFFFF;
                    border-radius: 5px;
                    border: 1px solid #E0E0E0;
                }
                QLabel#InfoLabel {
                    color: #757575;
                    font-size: 10pt;
                }
            """)

        main_layout.addWidget(info_panel)

        self.setCentralWidget(central_widget)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo.")

    def _update_button_states(self):
        model_loaded = self.yolo_model is not None
        media_active = self.media_thread is not None and self.media_thread.isRunning()

        self.open_image_action.setEnabled(model_loaded and not media_active)
        self.open_video_action.setEnabled(model_loaded and not media_active)
        self.start_webcam_action.setEnabled(model_loaded and not media_active)

        is_pausable = media_active and (self.current_source_type == "video" or self.current_source_type == "webcam")
        self.play_pause_action.setEnabled(is_pausable)
        self.stop_action.setEnabled(media_active)

        if media_active and self.media_thread._is_paused:
            self.play_pause_action.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.play_pause_action.setText("Reanudar") # El texto no se ve si es IconOnly
            self.play_pause_action.setToolTip("Reanudar procesamiento")
        else:
            self.play_pause_action.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.play_pause_action.setText("Pausar") # El texto no se ve si es IconOnly
            self.play_pause_action.setToolTip("Pausar procesamiento")

    @pyqtSlot(QPixmap)
    def _update_display_pixmap(self, pixmap):
        """Actualizar la imagen mostrada con mejor calidad"""
        if self.video_label:
            # Usar una fuente pequeña para el texto de detección
            self.video_label.setFont(QFont("Segoe UI", 10))

            # Calcular el tamaño disponible
            available_width = self.video_label.width() - 40  # Restar el padding
            available_height = self.video_label.height() - 40

            # Escalar la imagen manteniendo la proporción
            scaled_pixmap = pixmap.scaled(
                available_width,
                available_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation  # Alta calidad
            )

            # Establecer la imagen
            self.video_label.setPixmap(scaled_pixmap)

            # Actualizar el panel de información con el tamaño de la imagen
            if hasattr(self, 'info_label') and self.info_label:
                img_size = f"{scaled_pixmap.width()}x{scaled_pixmap.height()}"

                # Mostrar información sobre la fuente actual
                if self.current_source_type == "webcam":
                    self.info_label.setText(f"Cámara web activa | {img_size}")
                elif self.current_source_type == "video":
                    video_name = self.current_media_path.split('/')[-1] if self.current_media_path else "Video"
                    self.info_label.setText(f"{video_name} | {img_size}")
                elif self.current_source_type == "image":
                    img_name = self.current_media_path.split('/')[-1] if self.current_media_path else "Imagen"
                    self.info_label.setText(f"{img_name} | {img_size}")

                # Estilo normal
                if self.dark_mode:
                    self.info_label.setStyleSheet("color: #B0B0B0;")
                else:
                    self.info_label.setStyleSheet("color: #757575;")

    @pyqtSlot(str) # Este slot solo recibe el mensaje
    def _update_status(self, message):
        """Actualizar el estado en la barra de estado y el panel de información"""
        if self.status_bar:
            self.status_bar.showMessage(message) # El timeout se maneja en las llamadas directas

        # Actualizar también el panel de información si existe
        if hasattr(self, 'info_label') and self.info_label:
            self.info_label.setText(message)

            # Aplicar estilo según el tipo de mensaje
            if "Error" in message or "error" in message:
                # Estilo para mensajes de error
                if self.dark_mode:
                    self.info_label.setStyleSheet("color: #FF5252; font-weight: bold;")
                else:
                    self.info_label.setStyleSheet("color: #D32F2F; font-weight: bold;")
            elif "éxito" in message or "completado" in message or "procesada" in message:
                # Estilo para mensajes de éxito
                if self.dark_mode:
                    self.info_label.setStyleSheet("color: #69F0AE; font-weight: bold;")
                else:
                    self.info_label.setStyleSheet("color: #00C853; font-weight: bold;")
            else:
                # Estilo normal
                if self.dark_mode:
                    self.info_label.setStyleSheet("color: #B0B0B0;")
                else:
                    self.info_label.setStyleSheet("color: #757575;")

    def _clear_display(self):
        """Limpiar la pantalla de visualización"""
        if self.video_label:
            # Mensaje más minimalista
            welcome_message = "YOLO Vision" if self.dark_mode else "YOLO Vision"
            self.video_label.setText(welcome_message)
            self.video_label.setPixmap(QPixmap())

            # Restaurar la fuente grande
            font = QFont("Segoe UI", 24)
            font.setWeight(QFont.Weight.Light)
            self.video_label.setFont(font)

        # Actualizar el panel de información
        if hasattr(self, 'info_label') and self.info_label:
            self.info_label.setText("Seleccione una fuente o inicie la cámara")
            # Restaurar estilo normal
            if self.dark_mode:
                self.info_label.setStyleSheet("color: #B0B0B0;")
            else:
                self.info_label.setStyleSheet("color: #757575;")

    def _stop_current_media_if_running(self):
        if self.media_thread and self.media_thread.isRunning():
            self.media_thread.stop()
            return True
        return False

    def _select_image_file(self):
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return
        if self._stop_current_media_if_running():
             QTimer.singleShot(100, self.__proceed_with_image_selection)
        else:
            self.__proceed_with_image_selection()

    def __proceed_with_image_selection(self):
        """Procesar la selección de imagen con un diseño más moderno"""
        self.current_source_type = "image"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if file_path:
            self.current_media_path = file_path
            file_name = os.path.basename(file_path)

            # Mostrar mensaje de procesamiento
            self.status_bar.showMessage(f"Procesando imagen: {file_name}...")

            # Actualizar el panel de información
            if hasattr(self, 'info_label') and self.info_label:
                self.info_label.setText(f"Procesando {file_name}...")
                # Estilo de procesamiento
                if self.dark_mode:
                    self.info_label.setStyleSheet("color: #40C4FF; font-weight: bold;")  # Azul claro
                else:
                    self.info_label.setStyleSheet("color: #0091EA; font-weight: bold;")  # Azul oscuro

            QApplication.processEvents()

            try:
                # Cargar la imagen con OpenCV directamente
                img_cv = cv2.imread(file_path)

                if img_cv is None:
                    # Si OpenCV falla, intentar con PIL si está disponible
                    try:
                        from PIL import Image
                        img_pil = Image.open(file_path)
                        img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                    except (ImportError, Exception) as pil_error:
                        raise Exception(f"No se pudo cargar la imagen con OpenCV ni PIL: {pil_error}")

                # Convertir a RGB para procesamiento
                frame_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # Ejecutar detección
                results = self.yolo_model(frame_rgb, verbose=False)

                # Dibujar resultados con un estilo más moderno
                for r in results:
                    boxes, names = r.boxes, r.names
                    if boxes is not None:
                        for i in range(len(boxes)):
                            # Obtener coordenadas y datos
                            box_coords = boxes[i].xyxy[0].cpu().numpy().astype(int)
                            cls_id = int(boxes[i].cls.cpu().numpy()[0])
                            conf = float(boxes[i].conf.cpu().numpy()[0])

                            # Crear etiqueta con formato mejorado
                            label = f"{names[cls_id]}: {conf:.2f}"

                            # Dibujar rectángulo con estilo moderno
                            # Usar colores más vibrantes y líneas más delgadas
                            color = (0, 220, 255)  # Naranja-amarillo más vibrante
                            cv2.rectangle(
                                img_cv,
                                (box_coords[0], box_coords[1]),
                                (box_coords[2], box_coords[3]),
                                color,
                                2  # Grosor de línea
                            )

                            # Añadir fondo a la etiqueta para mejor legibilidad
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(
                                img_cv,
                                (box_coords[0], box_coords[1] - 25),
                                (box_coords[0] + text_size[0], box_coords[1]),
                                color,
                                -1  # Relleno
                            )

                            # Texto con mejor contraste
                            cv2.putText(
                                img_cv,
                                label,
                                (box_coords[0], box_coords[1] - 7),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,  # Tamaño de fuente
                                (0, 0, 0),  # Color negro para contraste
                                2  # Grosor
                            )

                # Convertir a formato Qt
                final_rgb_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = final_rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(final_rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                # Actualizar la visualización
                self._update_display_pixmap(QPixmap.fromImage(qt_image))

                # Mensaje de éxito
                success_msg = f"Imagen procesada: {file_name}"
                self.status_bar.showMessage(success_msg, 5000)

                # Actualizar el panel de información
                if hasattr(self, 'info_label') and self.info_label:
                    # Contar objetos detectados
                    num_objects = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
                    self.info_label.setText(f"{file_name} | {num_objects} objetos detectados")

                    # Estilo de éxito
                    if self.dark_mode:
                        self.info_label.setStyleSheet("color: #69F0AE; font-weight: bold;")  # Verde claro
                    else:
                        self.info_label.setStyleSheet("color: #00C853; font-weight: bold;")  # Verde oscuro

            except Exception as e:
                # Mensaje de error
                error_msg = f"Error al procesar imagen: {e}"
                self.status_bar.showMessage(error_msg)

                # Actualizar el panel de información
                if hasattr(self, 'info_label') and self.info_label:
                    self.info_label.setText(f"Error: {e}")
                    # Estilo de error
                    if self.dark_mode:
                        self.info_label.setStyleSheet("color: #FF5252; font-weight: bold;")  # Rojo claro
                    else:
                        self.info_label.setStyleSheet("color: #D32F2F; font-weight: bold;")  # Rojo oscuro

                QMessageBox.warning(self, "Error de Imagen", f"No se pudo procesar la imagen:\n{e}")

        # Actualizar estado de botones
        self._update_button_states()

    def _start_media_processing_thread(self, source_type, file_path=None):
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return
        if self._stop_current_media_if_running():
            QTimer.singleShot(200, lambda: self.__actually_start_thread(source_type, file_path))
        else:
            self.__actually_start_thread(source_type, file_path)

    def __actually_start_thread(self, source_type, file_path):
        self.current_source_type = source_type
        self.current_media_path = file_path
        self._clear_display()

        self.media_thread = MediaProcessingThread(self.yolo_model, source_type, file_path)
        self.media_thread.frame_ready.connect(self._update_display_pixmap)
        self.media_thread.status_update.connect(self._update_status) # Conectado al slot correcto
        self.media_thread.processing_finished.connect(self._on_media_processing_finished)

        self.media_thread.start()
        self._update_button_states()

    def _select_video_file(self):
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Archivos de Video (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self._start_media_processing_thread("video", file_path)

    def _start_webcam_mode(self):
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return
        self._start_media_processing_thread("webcam")

    def _toggle_play_pause_media(self):
        if self.media_thread and self.media_thread.isRunning():
            self.media_thread.toggle_pause()
        self._update_button_states()

    def _stop_current_media(self):
        stopped_something = self._stop_current_media_if_running()
        if stopped_something:
            # El slot _on_media_processing_finished se encargará del mensaje principal
            # y de la actualización de botones cuando el hilo realmente termine.
            self.status_bar.showMessage("Deteniendo procesamiento...", 2000) # Mensaje temporal
        else:
            self.status_bar.showMessage("No había procesamiento activo para detener.", 3000) # Uso directo
            self._clear_display()
            self.media_thread = None
            self._update_button_states()

    @pyqtSlot()
    def _on_media_processing_finished(self):
        final_message = "Procesamiento finalizado."
        if self.current_source_type == "video":
            final_message = "Procesamiento de video finalizado."
        elif self.current_source_type == "webcam":
             final_message = "Cámara detenida."

        self.status_bar.showMessage(final_message, 5000) # Uso directo

        self.media_thread = None
        # No limpiar display aquí si fue stop manual, para ver último frame
        # self._clear_display()
        self._update_button_states()

    def _toggle_theme(self):
        """Cambiar entre tema claro y oscuro"""
        self.dark_mode = not self.dark_mode

        # Aplicar el nuevo tema
        self.setStyleSheet(self._get_app_style_dark() if self.dark_mode else self._get_app_style_light())

        # Actualizar el estilo del video_label
        if self.dark_mode:
            self.video_label.setStyleSheet("""
                QLabel#VideoLabel {
                    background-color: #1E1E1E;
                    border: 1px solid #333333;
                    border-radius: 10px;
                    color: #00E5FF;
                    padding: 20px;
                }
            """)

            # Actualizar el panel de información
            if hasattr(self, 'info_panel'):
                self.info_panel.setStyleSheet("""
                    QFrame#InfoPanel {
                        background-color: #1A1A1A;
                        border-radius: 5px;
                        border: 1px solid #333333;
                    }
                    QLabel#InfoLabel {
                        color: #B0B0B0;
                        font-size: 10pt;
                    }
                """)
        else:
            self.video_label.setStyleSheet("""
                QLabel#VideoLabel {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E0E0;
                    border-radius: 10px;
                    color: #00ACC1;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
            """)

            # Actualizar el panel de información
            if hasattr(self, 'info_panel'):
                self.info_panel.setStyleSheet("""
                    QFrame#InfoPanel {
                        background-color: #FFFFFF;
                        border-radius: 5px;
                        border: 1px solid #E0E0E0;
                    }
                    QLabel#InfoLabel {
                        color: #757575;
                        font-size: 10pt;
                    }
                """)

        # Actualizar la barra de herramientas
        self._recreate_toolbar()

        # Mostrar mensaje de cambio de tema
        theme_name = "oscuro" if self.dark_mode else "claro"
        self.status_bar.showMessage(f"Tema {theme_name} aplicado", 3000)

    def _recreate_toolbar(self):
        """Recrear la barra de herramientas con el nuevo tema"""
        # Guardar las acciones actuales
        toolbar = self.findChild(QToolBar, "MainToolBar")
        if toolbar:
            self.removeToolBar(toolbar)

        # Crear una nueva barra de herramientas
        self._create_tool_bar()

    def closeEvent(self, event):
        self.status_bar.showMessage("Cerrando aplicación...", 2000) # Uso directo
        QApplication.processEvents()
        if self.media_thread and self.media_thread.isRunning():
            self.media_thread.stop()
        print("Aplicación cerrada.")
        event.accept()

# --- Punto de Entrada ---
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Verificar dependencias
    pillow_available = True
    try:
        from PIL import Image
    except ImportError:
        pillow_available = False
        QMessageBox.warning(None, "Dependencia Faltante",
                          "Pillow (PIL) no está instalado. El modo imagen no funcionará correctamente.\n"
                          "Por favor, instálalo con: pip install Pillow")

    # Iniciar la aplicación
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())