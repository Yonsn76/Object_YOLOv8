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
                    break # Salir del bucle si no hay frame (fin de video o error de cámara)
            else:
                self.status_update.emit("Fuente de video no disponible.")
                break # Salir del bucle si la captura no está disponible

            # Procesamiento del frame
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(frame_rgb, verbose=False)

            # Dibujar detecciones
            for r in results:
                boxes, names = r.boxes, r.names
                if boxes is not None:
                    for i in range(len(boxes)):
                        box_coords = boxes[i].xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(boxes[i].cls.cpu().numpy()[0])
                        conf = float(boxes[i].conf.cpu().numpy()[0])
                        label = f"{names[cls_id]}: {conf:.2f}"
                        
                        # Colores y estilo de detección
                        detection_color = (50, 205, 50) # Verde lima brillante
                        text_color_bg = (40, 160, 40) # Verde más oscuro para fondo de texto
                        text_color_fg = (255, 255, 255) # Blanco para texto

                        cv2.rectangle(frame_cv, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), detection_color, 2)
                        
                        # Fondo para el texto
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame_cv, (box_coords[0], box_coords[1] - h - 10), (box_coords[0] + w, box_coords[1] - 5), text_color_bg, -1)
                        cv2.putText(frame_cv, label, (box_coords[0], box_coords[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color_fg, 2)

            rgb_image_display = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image_display.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(QPixmap.fromImage(qt_image))

            if self.source_type == "webcam":
                QThread.msleep(10) # Control de FPS para webcam

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
        self.setWindowTitle("YOLO Vision Pro")
        self.setGeometry(50, 50, 1280, 850) # Ligeramente más grande

        self.dark_mode = True # Iniciar en modo oscuro por defecto
        self.setStyleSheet(self._get_app_style_dark() if self.dark_mode else self._get_app_style_light())

        self.yolo_model = None
        self.media_thread = None
        self.current_media_path = None
        self.current_source_type = None

        self._init_ui()
        self._load_yolo_model_async()

    def _get_app_style_dark(self):
        # Estilo QSS mejorado para un look moderno oscuro
        return """
            QMainWindow {
                background-color: #1C1C2E; /* Azul oscuro profundo */
                color: #F0F0F0;
                font-family: 'Segoe UI', Arial, sans-serif; /* Fuente principal */
            }
            QWidget {
                color: #F0F0F0;
                font-size: 10pt; /* Tamaño de fuente base */
            }
            QLabel#VideoLabel {
                background-color: #28283A; /* Fondo de video ligeramente más claro */
                border: 1px solid #3A3A4C;
                border-radius: 12px; /* Bordes más redondeados */
                color: #E0E0E0;
                padding: 5px;
            }
            QPushButton {
                background-color: #3D3D56; /* Botones con tono púrpura oscuro */
                color: #00CFE0; /* Texto Cyan vibrante */
                border: none;
                padding: 12px 24px; /* Más padding */
                font-size: 11pt; /* Letra más grande */
                border-radius: 6px; /* Bordes redondeados */
                font-weight: bold; /* Letra más gruesa */
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #4A4A6A; /* Hover más claro */
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #00A8B8; /* Cyan más oscuro al presionar */
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #2A2A3F;
                color: #606070;
            }
            QComboBox {
                background-color: #3D3D56;
                color: #F0F0F0;
                border: 1px solid #4A4A6A;
                padding: 10px;
                border-radius: 6px;
                min-height: 26px;
                font-weight: 600; /* Semi-gruesa */
                selection-background-color: #00A8B8;
            }
            QComboBox::drop-down {
                border: none;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                image: url(resources/arrow_down_light.png); /* Necesitarás un icono */
            }
            QComboBox QAbstractItemView {
                background-color: #2C2C3A;
                color: #F0F0F0;
                selection-background-color: #00A8B8;
                selection-color: #FFFFFF;
                border: 1px solid #3A3A4C;
                padding: 5px;
            }
            QToolBar { /* Estilo base, se refina en _create_tool_bar */
                background-color: #222233;
                border: none;
                border-bottom: 1px solid #333344;
                padding: 5px;
                spacing: 8px;
            }
            QToolBar QToolButton { /* Estilo base para botones de toolbar */
                font-weight: bold;
                font-size: 10pt;
            }
            QStatusBar {
                background-color: #222233;
                color: #C0C0D0;
                font-size: 9pt;
                font-weight: 500; /* Ligeramente más gruesa */
                border-top: 1px solid #333344;
            }
            QMenuBar {
                background-color: #222233;
                color: #F0F0F0;
                border-bottom: 1px solid #333344;
                font-weight: 600;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 15px;
            }
            QMenuBar::item:selected {
                background-color: #3D3D56;
                color: #00CFE0;
            }
            QMenuBar::item:pressed {
                background-color: #00A8B8;
                color: #FFFFFF;
            }
            QMenu {
                background-color: #2C2C3A;
                color: #F0F0F0;
                border: 1px solid #3A3A4C;
                border-radius: 6px;
                padding: 8px;
                font-weight: 500;
            }
            QMenu::item {
                padding: 10px 30px 10px 25px;
                border-radius: 4px;
                margin: 3px;
            }
            QMenu::item:selected {
                background-color: #00A8B8;
                color: #FFFFFF;
            }
            QMenu::separator {
                height: 1px;
                background: #3A3A4C;
                margin: 6px 12px;
            }
            QScrollBar:vertical {
                background-color: #222233;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #4A4A6A;
                min-height: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00A8B8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #222233;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #4A4A6A;
                min-width: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #00A8B8;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QMessageBox {
                background-color: #28283A;
                color: #F0F0F0;
                font-size: 10pt;
            }
            QMessageBox QLabel {
                color: #F0F0F0;
            }
            QFileDialog { /* Estilo básico, puede variar por OS */
                background-color: #28283A;
                color: #F0F0F0;
            }
        """

    def _get_app_style_light(self):
        # Estilo QSS mejorado para un look moderno claro
        return """
            QMainWindow {
                background-color: #F4F6F8; /* Gris claro azulado */
                color: #2C3E50; /* Azul oscuro para texto */
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QWidget {
                color: #2C3E50;
                font-size: 10pt;
            }
            QLabel#VideoLabel {
                background-color: #FFFFFF;
                border: 1px solid #DDE2E6;
                border-radius: 12px;
                color: #2C3E50;
                padding: 5px;
            }
            QPushButton {
                background-color: #3498DB; /* Azul brillante */
                color: #FFFFFF;
                border: none;
                padding: 12px 24px;
                font-size: 11pt;
                border-radius: 6px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #2980B9; /* Azul más oscuro en hover */
            }
            QPushButton:pressed {
                background-color: #1F618D; /* Azul aún más oscuro */
            }
            QPushButton:disabled {
                background-color: #BDC3C7; /* Gris claro */
                color: #7F8C8D; /* Gris oscuro para texto */
            }
            QComboBox {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                padding: 10px;
                border-radius: 6px;
                min-height: 26px;
                font-weight: 600;
                selection-background-color: #3498DB;
            }
            QComboBox::drop-down {
                border: none;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                image: url(resources/arrow_down_dark.png); /* Necesitarás un icono */
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #2C3E50;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
                border: 1px solid #BDC3C7;
                padding: 5px;
            }
            QToolBar {
                background-color: #ECF0F1; /* Gris muy claro */
                border: none;
                border-bottom: 1px solid #DDE2E6;
                padding: 5px;
                spacing: 8px;
            }
            QToolBar QToolButton {
                font-weight: bold;
                font-size: 10pt;
            }
            QStatusBar {
                background-color: #ECF0F1;
                color: #566573;
                font-size: 9pt;
                font-weight: 500;
                border-top: 1px solid #DDE2E6;
            }
            QMenuBar {
                background-color: #ECF0F1;
                color: #2C3E50;
                border-bottom: 1px solid #DDE2E6;
                font-weight: 600;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 15px;
            }
            QMenuBar::item:selected {
                background-color: #E0E6E8;
                color: #1A5276;
            }
            QMenuBar::item:pressed {
                background-color: #3498DB;
                color: #FFFFFF;
            }
            QMenu {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #DDE2E6;
                border-radius: 6px;
                padding: 8px;
                font-weight: 500;
            }
            QMenu::item {
                padding: 10px 30px 10px 25px;
                border-radius: 4px;
                margin: 3px;
            }
            QMenu::item:selected {
                background-color: #E0E6E8; /* Fondo sutil */
                color: #1A5276; /* Texto más oscuro */
            }
            QMenu::separator {
                height: 1px;
                background: #DDE2E6;
                margin: 6px 12px;
            }
            QScrollBar:vertical {
                background-color: #ECF0F1;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #BDC3C7; /* Gris */
                min-height: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #AAB0B4;
            }
            QScrollBar:horizontal {
                background-color: #ECF0F1;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #BDC3C7;
                min-width: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #AAB0B4;
            }
            QMessageBox {
                background-color: #FFFFFF;
                color: #2C3E50;
                font-size: 10pt;
            }
            QMessageBox QLabel {
                color: #2C3E50;
            }
            QFileDialog {
                background-color: #FFFFFF;
                color: #2C3E50;
            }
        """

    def _init_ui(self):
        self._create_menu_bar()
        self._create_tool_bar() 
        self._create_central_widget()
        self._create_status_bar()

    def _load_yolo_model_async(self):
        self.status_bar.showMessage("Cargando modelo YOLOv8n, por favor espera...")
        QTimer.singleShot(100, self._perform_model_load)

    def _perform_model_load(self):
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.status_bar.showMessage("Modelo YOLOv8n cargado. Sistema listo.", 5000)
            self._update_button_states()
            print("Modelo YOLOv8n cargado.")
        except Exception as e:
            self.status_bar.showMessage(f"Error crítico al cargar modelo YOLO: {e}")
            QMessageBox.critical(self, "Error de Modelo", f"No se pudo cargar el modelo YOLOv8n:\n{e}")
            self.open_image_action.setEnabled(False)
            self.open_video_action.setEnabled(False)
            self.start_webcam_action.setEnabled(False)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&Archivo")

        style = QApplication.style()
        self.open_image_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "&Abrir Imagen...", self)
        self.open_image_action.triggered.connect(self._select_image_file)
        file_menu.addAction(self.open_image_action)

        self.open_video_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DriveDVDIcon), "Abrir &Video...", self)
        self.open_video_action.triggered.connect(self._select_video_file)
        file_menu.addAction(self.open_video_action)

        file_menu.addSeparator()
        exit_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),"&Salir", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        camera_menu = menu_bar.addMenu("&Cámara")
        self.start_webcam_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Iniciar &Cámara Web", self)
        self.start_webcam_action.triggered.connect(self._start_webcam_mode)
        camera_menu.addAction(self.start_webcam_action)

    def _create_tool_bar(self):
        tool_bar = QToolBar("Herramientas Principales")
        tool_bar.setIconSize(QSize(28, 28)) # Iconos más grandes
        tool_bar.setMovable(True) # Permitir moverla
        tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon) # Texto al lado del icono
        tool_bar.setObjectName("MainToolBar")
        tool_bar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea | Qt.ToolBarArea.BottomToolBarArea)


        # Aplicar estilo específico para la barra de herramientas
        toolbar_qss_dark = """
            QToolBar#MainToolBar {
                background-color: #222233; /* Fondo de la barra */
                border: none;
                border-bottom: 2px solid #00CFE0; /* Borde inferior acentuado */
                padding: 8px; /* Más padding */
                spacing: 12px; /* Más espaciado */
            }
            QToolButton {
                background-color: transparent;
                color: #E0E0F0; /* Color de texto */
                border: 1px solid transparent; /* Borde sutil */
                border-radius: 6px;
                padding: 8px 12px; /* Padding interno */
                margin: 2px;
                font-size: 10pt; /* Tamaño de fuente */
                font-weight: bold; /* Letra gruesa */
            }
            QToolButton:hover {
                background-color: #3D3D56; /* Fondo en hover */
                color: #00CFE0; /* Color de texto en hover */
                border-color: #00CFE0;
            }
            QToolButton:pressed {
                background-color: #00A8B8; /* Fondo al presionar */
                color: #FFFFFF;
            }
            QToolButton:disabled {
                color: #606070;
                background-color: transparent;
            }
            QToolBarSeparator {
                background-color: #3A3A4C;
                width: 2px; /* Separador más grueso */
                margin: 4px 8px; /* Margen para el separador */
            }
        """
        toolbar_qss_light = """
            QToolBar#MainToolBar {
                background-color: #ECF0F1;
                border: none;
                border-bottom: 2px solid #3498DB;
                padding: 8px;
                spacing: 12px;
            }
            QToolButton {
                background-color: transparent;
                color: #2C3E50;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 8px 12px;
                margin: 2px;
                font-size: 10pt;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #E0E6E8;
                color: #1A5276;
                border-color: #3498DB;
            }
            QToolButton:pressed {
                background-color: #3498DB;
                color: #FFFFFF;
            }
            QToolButton:disabled {
                color: #7F8C8D;
            }
            QToolBarSeparator {
                background-color: #DDE2E6;
                width: 2px;
                margin: 4px 8px;
            }
        """
        tool_bar.setStyleSheet(toolbar_qss_dark if self.dark_mode else toolbar_qss_light)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tool_bar)

        style = QApplication.style()

        # Reutilizar acciones del menú para consistencia
        self.open_image_action.setText("Abrir Imagen") # Texto más corto para toolbar
        tool_bar.addAction(self.open_image_action)

        self.open_video_action.setText("Abrir Video")
        tool_bar.addAction(self.open_video_action)
        
        self.start_webcam_action.setText("Cámara Web")
        tool_bar.addAction(self.start_webcam_action)

        tool_bar.addSeparator()

        self.play_pause_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPause), "Pausar", self)
        self.play_pause_action.setToolTip("Reproducir/Pausar procesamiento")
        self.play_pause_action.triggered.connect(self._toggle_play_pause_media)
        tool_bar.addAction(self.play_pause_action)

        self.stop_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Detener", self)
        self.stop_action.setToolTip("Detener procesamiento actual")
        self.stop_action.triggered.connect(self._stop_current_media)
        tool_bar.addAction(self.stop_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tool_bar.addWidget(spacer)

        self.toggle_theme_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DesktopIcon), # Icono más genérico
                                         "Tema", self)
        self.toggle_theme_action.setToolTip("Cambiar tema (Claro/Oscuro)")
        self.toggle_theme_action.triggered.connect(self._toggle_theme)
        tool_bar.addAction(self.toggle_theme_action)

        self._update_button_states()


    def _create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25) 
        main_layout.setSpacing(25)

        video_container = QFrame()
        video_container.setObjectName("VideoContainer")
        # El estilo del contenedor se maneja por QLabel#VideoLabel principalmente
        video_container.setStyleSheet("QFrame#VideoContainer { background-color: transparent; border-radius: 12px; }")

        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        welcome_message = "YOLO Vision Pro"
        self.video_label = QLabel(welcome_message)
        self.video_label.setObjectName("VideoLabel") # Para aplicar QSS específico
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        font = QFont("Segoe UI Light", 30, QFont.Weight.ExtraLight) # Fuente más elegante y grande
        self.video_label.setFont(font)
        
        # El estilo del video_label se aplica a través del QSS general
        # No es necesario setStyleSheet aquí si ya está en _get_app_style_...

        video_layout.addWidget(self.video_label)
        main_layout.addWidget(video_container, 1) # El 1 da más peso al video label

        info_panel = QFrame()
        info_panel.setObjectName("InfoPanel")
        info_panel.setMaximumHeight(45) # Un poco más de altura

        info_layout = QHBoxLayout(info_panel)
        info_layout.setContentsMargins(15, 8, 15, 8) # Más padding

        self.info_label = QLabel("Seleccione una fuente o inicie la cámara")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        info_font = QFont("Segoe UI", 10) # Fuente para el panel de info
        info_font.setWeight(QFont.Weight.Medium) # Ligeramente más gruesa
        self.info_label.setFont(info_font)

        info_layout.addWidget(self.info_label)

        # Estilo del panel de información
        info_panel_qss_dark = """
            QFrame#InfoPanel {
                background-color: #28283A; /* Similar al video label bg */
                border-radius: 8px;
                border: 1px solid #3A3A4C;
            }
            QLabel#InfoLabel {
                color: #B0B0C0; /* Texto informativo */
                font-size: 10pt;
                font-weight: 600; /* Más grueso */
            }
        """
        info_panel_qss_light = """
            QFrame#InfoPanel {
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #DDE2E6;
            }
            QLabel#InfoLabel {
                color: #566573; /* Texto informativo */
                font-size: 10pt;
                font-weight: 600;
            }
        """
        info_panel.setStyleSheet(info_panel_qss_dark if self.dark_mode else info_panel_qss_light)
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

        style = QApplication.style()
        if media_active and self.media_thread._is_paused:
            self.play_pause_action.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.play_pause_action.setText("Reanudar")
            self.play_pause_action.setToolTip("Reanudar procesamiento")
        else:
            self.play_pause_action.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.play_pause_action.setText("Pausar")
            self.play_pause_action.setToolTip("Pausar procesamiento")

    @pyqtSlot(QPixmap)
    def _update_display_pixmap(self, pixmap):
        if self.video_label:
            self.video_label.setFont(QFont("Segoe UI", 10)) # Fuente normal para el video

            available_width = self.video_label.width() - 10 # Menos padding
            available_height = self.video_label.height() - 10

            scaled_pixmap = pixmap.scaled(
                available_width,
                available_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            if hasattr(self, 'info_label') and self.info_label:
                img_size = f"{scaled_pixmap.width()}x{scaled_pixmap.height()}"
                source_name = ""
                if self.current_source_type == "webcam":
                    source_name = "Cámara web activa"
                elif self.current_source_type == "video":
                    source_name = self.current_media_path.split('/')[-1] if self.current_media_path else "Video"
                elif self.current_source_type == "image":
                    source_name = self.current_media_path.split('/')[-1] if self.current_media_path else "Imagen"
                
                self.info_label.setText(f"{source_name} | {img_size}")
                # Restaurar estilo normal del info_label (se colorea en _update_status)
                self._set_info_label_style("normal")


    @pyqtSlot(str)
    def _update_status(self, message):
        if self.status_bar:
            self.status_bar.showMessage(message)

        if hasattr(self, 'info_label') and self.info_label:
            self.info_label.setText(message)
            if "Error" in message or "error" in message or "Fallo" in message:
                self._set_info_label_style("error", message)
            elif "éxito" in message or "completado" in message or "procesada" in message or "listo" in message or "Cámara iniciada" in message or "Procesando video" in message:
                self._set_info_label_style("success", message)
            elif "pausado" in message or "reanudado" in message:
                 self._set_info_label_style("info", message)
            else:
                self._set_info_label_style("normal", message)
    
    def _set_info_label_style(self, style_type, message=""):
        """Establece el estilo del info_label basado en el tipo de mensaje."""
        if not hasattr(self, 'info_label') or not self.info_label:
            return

        # Base style
        base_style_dark = "color: #B0B0C0; font-weight: 600;"
        base_style_light = "color: #566573; font-weight: 600;"
        current_base_style = base_style_dark if self.dark_mode else base_style_light

        if style_type == "error":
            color = "#FF6B6B" if self.dark_mode else "#E74C3C" # Rojo
            self.info_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        elif style_type == "success":
            color = "#69F0AE" if self.dark_mode else "#2ECC71" # Verde
            self.info_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        elif style_type == "info":
            color = "#4FC3F7" if self.dark_mode else "#3498DB" # Azul
            self.info_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        else: # normal
            self.info_label.setStyleSheet(current_base_style)
        
        if message: self.info_label.setText(message)


    def _clear_display(self):
        if self.video_label:
            welcome_message = "YOLO Vision Pro"
            self.video_label.setText(welcome_message)
            self.video_label.setPixmap(QPixmap())
            font = QFont("Segoe UI Light", 30, QFont.Weight.ExtraLight)
            self.video_label.setFont(font)

        if hasattr(self, 'info_label') and self.info_label:
            self._set_info_label_style("normal", "Seleccione una fuente o inicie la cámara")


    def _stop_current_media_if_running(self):
        if self.media_thread and self.media_thread.isRunning():
            self.media_thread.stop()
            # Esperar un poco para que el hilo termine limpiamente
            # self.media_thread.wait(500) # Puede causar bloqueo, usar con cuidado o señales
            return True
        return False

    def _select_image_file(self):
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return
        if self._stop_current_media_if_running():
            QTimer.singleShot(150, self.__proceed_with_image_selection) # Dar tiempo al hilo a parar
        else:
            self.__proceed_with_image_selection()

    def __proceed_with_image_selection(self):
        self.current_source_type = "image"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "",
            "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp *.webp *.gif)"
        )

        if file_path:
            self.current_media_path = file_path
            file_name = os.path.basename(file_path)
            self._update_status(f"Procesando imagen: {file_name}...")
            self._set_info_label_style("info", f"Procesando {file_name}...")
            QApplication.processEvents()

            try:
                img_cv = cv2.imread(file_path)
                if img_cv is None:
                    try:
                        from PIL import Image as PILImage # Renombrar para evitar conflicto
                        img_pil = PILImage.open(file_path)
                        img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                    except (ImportError, Exception) as pil_error:
                        raise Exception(f"No se pudo cargar la imagen con OpenCV ni PIL: {pil_error}")

                frame_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                results = self.yolo_model(frame_rgb, verbose=False)
                
                # Dibujar resultados con estilo mejorado
                for r in results:
                    boxes, names = r.boxes, r.names
                    if boxes is not None:
                        for i in range(len(boxes)):
                            box_coords = boxes[i].xyxy[0].cpu().numpy().astype(int)
                            cls_id = int(boxes[i].cls.cpu().numpy()[0])
                            conf = float(boxes[i].conf.cpu().numpy()[0])
                            label = f"{names[cls_id]}: {conf:.2f}"

                            detection_color = (50, 220, 90) # Verde más brillante
                            text_bg_color = (30, 180, 70) 
                            text_fg_color = (255, 255, 255)

                            cv2.rectangle(img_cv, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), detection_color, 2)
                            
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1) # FONT_HERSHEY_DUPLEX
                            cv2.rectangle(img_cv, (box_coords[0], box_coords[1] - h - 10), (box_coords[0] + w + 4, box_coords[1] - 5), text_bg_color, -1)
                            cv2.putText(img_cv, label, (box_coords[0] + 2, box_coords[1] - 7), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.6, text_fg_color, 1, cv2.LINE_AA) # Antialiasing

                final_rgb_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                h_img, w_img, ch_img = final_rgb_image.shape
                bytes_per_line_img = ch_img * w_img
                qt_image = QImage(final_rgb_image.data, w_img, h_img, bytes_per_line_img, QImage.Format.Format_RGB888)
                
                self._update_display_pixmap(QPixmap.fromImage(qt_image))
                
                num_objects = len(results[0].boxes) if hasattr(results[0], 'boxes') and results[0].boxes is not None else 0
                success_msg = f"Imagen procesada: {file_name} ({num_objects} objetos)"
                self._update_status(success_msg)
                self._set_info_label_style("success", success_msg)

            except Exception as e:
                error_msg = f"Error al procesar imagen: {e}"
                self._update_status(error_msg)
                self._set_info_label_style("error", f"Error: {e}")
                QMessageBox.warning(self, "Error de Imagen", f"No se pudo procesar la imagen:\n{e}")
            
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
        self.media_thread.status_update.connect(self._update_status)
        self.media_thread.processing_finished.connect(self._on_media_processing_finished)
        
        self.media_thread.start()
        self._update_button_states()

    def _select_video_file(self):
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", 
                                                 "Archivos de Video (*.mp4 *.avi *.mkv *.mov *.webm)")
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
        if self._stop_current_media_if_running():
            self.status_bar.showMessage("Deteniendo procesamiento...", 2000)
            # _on_media_processing_finished se llamará cuando el hilo termine
        else:
            self._update_status("No hay procesamiento activo para detener.")
            self._clear_display() # Limpiar si no había nada corriendo
            self.media_thread = None 
        self._update_button_states()


    @pyqtSlot()
    def _on_media_processing_finished(self):
        final_message = "Procesamiento finalizado."
        if self.current_source_type == "video":
            final_message = "Procesamiento de video finalizado."
        elif self.current_source_type == "webcam":
            final_message = "Cámara detenida."
        
        self._update_status(final_message)
        # No limpiar display aquí para ver el último frame si fue un stop.
        # Si es fin de video, el hilo ya habrá emitido "Video finalizado".
        
        self.media_thread = None # Asegurarse que el hilo se marca como no existente
        self._update_button_states()


    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.setStyleSheet(self._get_app_style_dark() if self.dark_mode else self._get_app_style_light())
        
        # Re-aplicar estilos específicos que no se heredan bien
        self._recreate_toolbar() # La toolbar necesita recrearse para aplicar su QSS interno
        
        # Actualizar estilo del panel de información
        info_panel = self.findChild(QFrame, "InfoPanel")
        if info_panel:
            info_panel_qss_dark = """
                QFrame#InfoPanel { background-color: #28283A; border-radius: 8px; border: 1px solid #3A3A4C; }
                QLabel#InfoLabel { color: #B0B0C0; font-size: 10pt; font-weight: 600; }
            """
            info_panel_qss_light = """
                QFrame#InfoPanel { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #DDE2E6; }
                QLabel#InfoLabel { color: #566573; font-size: 10pt; font-weight: 600; }
            """
            info_panel.setStyleSheet(info_panel_qss_dark if self.dark_mode else info_panel_qss_light)
            # Actualizar el texto del info_label con el estilo correcto
            self._set_info_label_style("normal", self.info_label.text())


        # Actualizar el video_label (por si tiene texto de bienvenida)
        if self.video_label and not (self.media_thread and self.media_thread.isRunning()):
             self._clear_display() # Restaura el mensaje de bienvenida con la fuente correcta

        theme_name = "oscuro" if self.dark_mode else "claro"
        self.status_bar.showMessage(f"Tema {theme_name} aplicado.", 3000)

    def _recreate_toolbar(self):
        # Eliminar la barra de herramientas existente si existe
        old_toolbar = self.findChild(QToolBar, "MainToolBar")
        if old_toolbar:
            self.removeToolBar(old_toolbar)
            old_toolbar.deleteLater()
        
        # Volver a crear la barra de herramientas
        self._create_tool_bar()


    def closeEvent(self, event):
        self.status_bar.showMessage("Cerrando aplicación...", 2000)
        QApplication.processEvents() # Procesar eventos pendientes
        if self.media_thread and self.media_thread.isRunning():
            self.media_thread.stop()
            self.media_thread.wait(500) # Esperar un poco a que el hilo termine
        print("Aplicación cerrada.")
        event.accept()

# --- Punto de Entrada ---
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Opcional: Crear un directorio 'resources' si no existe para los iconos de QComboBox
    # if not os.path.exists("resources"):
    #     os.makedirs("resources")
    # Aquí deberías añadir los archivos arrow_down_light.png y arrow_down_dark.png
    # o eliminar la referencia a ellos en el QSS de QComboBox::drop-down

    pillow_available = True
    try:
        from PIL import Image as PILImage
    except ImportError:
        pillow_available = False
        # No mostrar QMessageBox aquí, ya que MainWindow aún no está creada.
        # Se podría manejar después de crear MainWindow o simplemente dejar que falle si se usa una imagen.
        print("Advertencia: Pillow (PIL) no está instalado. Algunas funciones de imagen podrían fallar.")
        print("Por favor, instálalo con: pip install Pillow")


    main_win = MainWindow()
    main_win.show()

    if not pillow_available:
         QMessageBox.warning(main_win, "Dependencia Faltante",
                            "Pillow (PIL) no está instalado. El procesamiento de algunos formatos de imagen podría fallar.\n"
                            "Se recomienda instalarlo con: pip install Pillow")
    sys.exit(app.exec())
