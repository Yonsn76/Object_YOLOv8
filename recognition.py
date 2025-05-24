import sys
import os
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QFrame, QFileDialog, QComboBox,
    QStyle, QToolBar, QMessageBox, QSizePolicy, QSlider, QMenu
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction, QIcon, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, pyqtSlot
from ultralytics import YOLO

# --- Hilo para el procesamiento de Medios (Cámara o Video) ---
class MediaProcessingThread(QThread):
    frame_ready = pyqtSignal(QPixmap)
    status_update = pyqtSignal(str)
    processing_finished = pyqtSignal()
    frame_position = pyqtSignal(int)
    total_frames = pyqtSignal(int)

    def __init__(self, yolo_model, source_type="webcam", file_path=None):
        super().__init__()
        self.yolo_model = yolo_model
        self.source_type = source_type
        self.file_path = file_path
        self._is_running = False
        self._is_paused = False
        self.cap = None
        self.current_frame = 0
        self.total_frame_count = 0
        self.frame_rate = 30

    def stop(self):
        """Detiene el procesamiento y libera recursos"""
        self._is_running = False
        self._is_paused = False
        # Esperar un momento para asegurar que el bucle principal termine
        self.msleep(100)
        # Liberar la cámara si está abierta
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def run(self):
        self._is_running = True
        self._is_paused = False

        try:
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
                        self.total_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
                        self.total_frames.emit(self.total_frame_count)
            else:
                self.status_update.emit("Error: Tipo de fuente no reconocido.")
                self._is_running = False

            if not self._is_running:
                if self.cap:
                    self.cap.release()
                self.processing_finished.emit()
                return

            if self.source_type == "webcam":
                self.status_update.emit("Cámara iniciada. Detectando...")
            elif self.source_type == "video":
                self.status_update.emit(f"Procesando video: {self.file_path.split('/')[-1]}")

            while self._is_running:
                if self._is_paused:
                    self.msleep(100)
                    continue

                if not self.cap or not self.cap.isOpened():
                    break

                ret, frame_cv = self.cap.read()
                if not ret:
                    if self.source_type == "video":
                        self.status_update.emit("Video finalizado.")
                    else:
                        self.status_update.emit("Error al leer fotograma. Intentando reconectar...")
                        if self.cap:
                            self.cap.release()
                        self.cap = cv2.VideoCapture(0)
                        if not self.cap.isOpened():
                            self.status_update.emit("Fallo al reconectar la cámara.")
                            self._is_running = False
                        self.msleep(500)
                    break

                if self.source_type == "video":
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.frame_position.emit(self.current_frame)

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
                            
                            detection_color = (79, 70, 229)
                            text_bg_color = (67, 56, 202)
                            text_fg_color = (255, 255, 255)

                            cv2.rectangle(frame_cv, (box_coords[0], box_coords[1]), 
                                        (box_coords[2], box_coords[3]), detection_color, 2)
                            
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                            cv2.rectangle(frame_cv, (box_coords[0], box_coords[1] - h - 10), 
                                        (box_coords[0] + w + 4, box_coords[1] - 5), text_bg_color, -1)
                            cv2.putText(frame_cv, label, (box_coords[0] + 2, box_coords[1] - 7), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.6, text_fg_color, 1, cv2.LINE_AA)

                rgb_image_display = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image_display.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(QPixmap.fromImage(qt_image))

                if self.source_type == "webcam":
                    self.msleep(10)

        except Exception as e:
            self.status_update.emit(f"Error en el procesamiento: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.cap = None
            self._is_running = False
            self.processing_finished.emit()

    def _stop_current_media_if_running(self):
        """Detiene el procesamiento actual si hay alguno en curso"""
        if self.media_thread and self.media_thread.isRunning():
            try:
                # Detener el hilo
                self.media_thread.stop()
                
                # Esperar a que el hilo termine (con timeout)
                start_time = time.time()
                while self.media_thread.isRunning() and (time.time() - start_time) < 2.0:
                    QApplication.processEvents()
                    self.media_thread.wait(100)
                
                if self.media_thread.isRunning():
                    print("Advertencia: El hilo no se detuvo correctamente")
                    self.media_thread.terminate()
                
                # Asegurar que la cámara se ha liberado
                if hasattr(self.media_thread, 'cap') and self.media_thread.cap:
                    self.media_thread.cap.release()
                    self.media_thread.cap = None
                
                self.media_thread = None
                
                # Limpiar la interfaz
                self._clear_display()
                if hasattr(self, 'video_controls'):
                    self.video_controls.setVisible(False)
                
                return True
            except Exception as e:
                print(f"Error al detener el medio actual: {e}")
                if self.media_thread:
                    self.media_thread.terminate()
                    self.media_thread = None
                return True
        return False

    def _start_webcam_mode(self):
        """Inicia el modo de cámara web"""
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return

        try:
            # Detener cualquier procesamiento activo y esperar a que termine
            if self._stop_current_media_if_running():
                # Esperar un momento para asegurar que todo se ha liberado
                QTimer.singleShot(1000, self._actually_start_webcam)
            else:
                self._actually_start_webcam()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar la cámara web:\n{str(e)}")

    def _actually_start_webcam(self):
        """Función interna para iniciar la cámara web"""
        try:
            # Verificar nuevamente que no haya procesamiento activo
            if self.media_thread and self.media_thread.isRunning():
                QMessageBox.warning(self, "Error", "No se pudo detener el procesamiento anterior.")
                return

            # Ocultar controles de video si están visibles
            if hasattr(self, 'video_controls'):
                self.video_controls.setVisible(False)

            # Limpiar cualquier estado anterior
            self._clear_display()
            self.current_source_type = "webcam"
            self.current_media_path = None

            # Iniciar la cámara
            self._start_media_processing_thread("webcam")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar la cámara web:\n{str(e)}")

    def toggle_pause(self):
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.status_update.emit("Procesamiento pausado.")
        else:
            self.status_update.emit("Procesamiento reanudado.")
        return self._is_paused

    def seek_to_frame(self, frame_number):
        if self.cap and self.source_type == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number

    def get_video_duration(self):
        if self.cap and self.source_type == "video":
            return self.total_frame_count / self.frame_rate
        return 0

# --- Ventana Principal ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Vision Pro - Tomson")
        self.setGeometry(50, 50, 1280, 850)

        # Quitar barra de título estándar pero mantener el título en la barra de tareas
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowMinMaxButtonsHint)
        
        self.dark_mode = True
        self.setStyleSheet(self._get_app_style_dark() if self.dark_mode else self._get_app_style_light())

        self.yolo_model = None
        self.media_thread = None
        self.current_media_path = None
        self.current_source_type = None
        self._is_dragging = False
        self._drag_position = None

        # Referencias a los botones principales
        self.archivo_btn = None
        self.camara_btn = None
        self.play_pause_btn = None
        self.stop_btn = None

        self._init_ui()
        self._load_yolo_model_async()

    def _get_app_style_dark(self):
        return """
            QMainWindow {
                background-color: #171721; /* Fondo más oscuro y elegante */
                color: #E0E0FF;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QWidget {
                color: #E0E0FF;
                font-size: 10pt;
            }
            QLabel#VideoLabel {
                background-color: #1E1E2A;
                border: 2px solid #2A2A3A;
                border-radius: 15px;
                color: #E0E0FF;
                padding: 8px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 11pt;
                border-radius: 8px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #5A4AFF, stop:1 #4AFFFD);
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #3A2AEF, stop:1 #2AEFDD);
            }
            QPushButton:disabled {
                background: #2A2A3A;
                color: #505060;
            }
            QComboBox {
                background-color: #2A2A3A;
                color: #E0E0FF;
                border: 2px solid #3A3A4A;
                padding: 10px;
                border-radius: 8px;
                min-height: 28px;
                font-weight: 600;
            }
            QComboBox:hover {
                border-color: #4A3AFF;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A3A;
                color: #E0E0FF;
                selection-background-color: #4A3AFF;
                selection-color: white;
                border: 1px solid #3A3A4A;
                padding: 5px;
            }
            QToolBar {
                background-color: #1A1A25;
                border: none;
                border-bottom: 2px solid #4A3AFF;
                padding: 8px;
                spacing: 10px;
            }
            QToolBar QToolButton {
                font-weight: bold;
                font-size: 10pt;
                color: #E0E0FF;
                background-color: transparent;
                border: 2px solid transparent;
                border-radius: 8px;
                padding: 8px;
            }
            QToolBar QToolButton:hover {
                background-color: rgba(74, 58, 255, 0.2);
                border-color: #4A3AFF;
            }
            QToolBar QToolButton:pressed {
                background-color: rgba(74, 58, 255, 0.3);
            }
            QStatusBar {
                background-color: #1A1A25;
                color: #B0B0D0;
                font-size: 9pt;
                font-weight: 500;
                border-top: 1px solid #2A2A3A;
            }
            QMenuBar {
                background-color: #1A1A25;
                color: #E0E0FF;
                border-bottom: 2px solid #2A2A3A;
                font-weight: 600;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 15px;
            }
            QMenuBar::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                color: white;
            }
            QMenu {
                background-color: #2A2A3A;
                color: #E0E0FF;
                border: 1px solid #3A3A4A;
                border-radius: 8px;
                padding: 8px;
            }
            QMenu::item {
                padding: 10px 30px 10px 25px;
                border-radius: 4px;
                margin: 3px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #3A3A4A;
                margin: 6px 12px;
            }
            QScrollBar:vertical {
                background-color: #1A1A25;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                min-height: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #5A4AFF, stop:1 #4AFFFD);
            }
            QScrollBar:horizontal {
                background-color: #1A1A25;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                min-width: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #5A4AFF, stop:1 #4AFFFD);
            }
            QMessageBox {
                background-color: #2A2A3A;
                color: #E0E0FF;
            }
            QMessageBox QLabel {
                color: #E0E0FF;
            }
            QFileDialog {
                background-color: #2A2A3A;
                color: #E0E0FF;
            }
            QFrame#InfoPanel {
                background-color: #1E1E2A;
                border-radius: 10px;
                border: 2px solid #2A2A3A;
            }
            QLabel#InfoLabel {
                color: #B0B0D0;
                font-size: 10pt;
                font-weight: 600;
            }
        """

    def _get_app_style_light(self):
        return """
            QMainWindow {
                background-color: #F8F9FC;
                color: #2C3E50;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QWidget {
                color: #2C3E50;
                font-size: 10pt;
            }
            QLabel#VideoLabel {
                background-color: white;
                border: 2px solid #E8EAF6;
                border-radius: 15px;
                color: #2C3E50;
                padding: 8px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #6366F1, stop:1 #60A5FA);
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 11pt;
                border-radius: 8px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #818CF8, stop:1 #7DD3FC);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4F46E5, stop:1 #3B82F6);
            }
            QPushButton:disabled {
                background: #E2E8F0;
                color: #94A3B8;
            }
            QComboBox {
                background-color: white;
                color: #2C3E50;
                border: 2px solid #E2E8F0;
                padding: 10px;
                border-radius: 8px;
                min-height: 28px;
                font-weight: 600;
            }
            QComboBox:hover {
                border-color: #6366F1;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #2C3E50;
                selection-background-color: #6366F1;
                selection-color: white;
                border: 1px solid #E2E8F0;
                padding: 5px;
            }
            QToolBar {
                background-color: #F1F5F9;
                border: none;
                border-bottom: 2px solid #6366F1;
                padding: 8px;
                spacing: 10px;
            }
            QToolBar QToolButton {
                font-weight: bold;
                font-size: 10pt;
                color: #2C3E50;
                background-color: transparent;
                border: 2px solid transparent;
                border-radius: 8px;
                padding: 8px;
            }
            QToolBar QToolButton:hover {
                background-color: rgba(99, 102, 241, 0.1);
                border-color: #6366F1;
            }
            QToolBar QToolButton:pressed {
                background-color: rgba(99, 102, 241, 0.2);
            }
            QStatusBar {
                background-color: #F1F5F9;
                color: #64748B;
                font-size: 9pt;
                font-weight: 500;
                border-top: 1px solid #E2E8F0;
            }
            QMenuBar {
                background-color: #F1F5F9;
                color: #2C3E50;
                border-bottom: 2px solid #E2E8F0;
                font-weight: 600;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 15px;
            }
            QMenuBar::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #6366F1, stop:1 #60A5FA);
                color: white;
            }
            QMenu {
                background-color: white;
                color: #2C3E50;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 8px;
            }
            QMenu::item {
                padding: 10px 30px 10px 25px;
                border-radius: 4px;
                margin: 3px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #6366F1, stop:1 #60A5FA);
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #E2E8F0;
                margin: 6px 12px;
            }
            QScrollBar:vertical {
                background-color: #F1F5F9;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #6366F1, stop:1 #60A5FA);
                min-height: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #818CF8, stop:1 #7DD3FC);
            }
            QScrollBar:horizontal {
                background-color: #F1F5F9;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #6366F1, stop:1 #60A5FA);
                min-width: 25px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #818CF8, stop:1 #7DD3FC);
            }
            QMessageBox {
                background-color: white;
                color: #2C3E50;
            }
            QMessageBox QLabel {
                color: #2C3E50;
            }
            QFileDialog {
                background-color: white;
                color: #2C3E50;
            }
            QFrame#InfoPanel {
                background-color: white;
                border-radius: 10px;
                border: 2px solid #E8EAF6;
            }
            QLabel#InfoLabel {
                color: #64748B;
                font-size: 10pt;
                font-weight: 600;
            }
        """

    def _init_ui(self):
        self._create_central_widget()
        self._create_status_bar()

    def _create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header unificado
        self.header = QFrame()
        self.header.setObjectName("Header")
        self.header.setFixedHeight(45)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        header_layout.setSpacing(10)

        # Menú izquierdo con iconos
        menu_container = QFrame()
        menu_container.setObjectName("MenuContainer")
        menu_layout = QHBoxLayout(menu_container)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(15)

        self.archivo_btn = QPushButton("Archivo")
        self.archivo_btn.setObjectName("MenuButton")
        self.archivo_btn.setIcon(QIcon.fromTheme("document-open"))
        self.archivo_btn.clicked.connect(self._show_archivo_menu)
        
        self.camara_btn = QPushButton("Cámara")
        self.camara_btn.setObjectName("MenuButton")
        self.camara_btn.setIcon(QIcon.fromTheme("camera-web"))
        self.camara_btn.clicked.connect(self._show_camara_menu)

        menu_layout.addWidget(self.archivo_btn)
        menu_layout.addWidget(self.camara_btn)

        # Título central con logo
        title_container = QFrame()
        title_container.setObjectName("TitleContainer")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(10)

        logo_label = QLabel("🛡️")
        logo_label.setObjectName("LogoLabel")
        title_label = QLabel("YOLO Vision Pro - Tomson")
        title_label.setObjectName("TitleLabel")
        
        title_layout.addWidget(logo_label)
        title_layout.addWidget(title_label)

        # Controles de ventana y tema
        controls_container = QFrame()
        controls_container.setObjectName("WindowControls")
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        # Botón de tema
        self.theme_btn = QPushButton("🌙" if self.dark_mode else "☀️")
        self.theme_btn.setObjectName("ThemeButton")
        self.theme_btn.setToolTip("Cambiar tema (Claro/Oscuro)")
        self.theme_btn.clicked.connect(self._show_theme_menu)
        self.theme_btn.setFixedSize(30, 30)
        controls_layout.addWidget(self.theme_btn)

        controls_layout.addSpacing(10)

        minimize_btn = QPushButton("─")
        minimize_btn.setObjectName("MinimizeButton")
        minimize_btn.clicked.connect(self.showMinimized)

        maximize_btn = QPushButton("□")
        maximize_btn.setObjectName("MaximizeButton")
        maximize_btn.clicked.connect(self._toggle_maximize)

        close_btn = QPushButton("×")
        close_btn.setObjectName("CloseButton")
        close_btn.clicked.connect(self.close)

        for btn in [minimize_btn, maximize_btn, close_btn]:
            btn.setFixedSize(30, 30)
            controls_layout.addWidget(btn)

        # Agregar todos los elementos al header
        header_layout.addWidget(menu_container)
        header_layout.addStretch()
        header_layout.addWidget(title_container)
        header_layout.addStretch()
        header_layout.addWidget(controls_container)

        # Estilo del header actualizado
        self.header.setStyleSheet("""
            QFrame#Header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #1a1a2e, stop:1 #2d2d44);
                border-bottom: 2px solid rgba(74, 58, 255, 0.3);
            }
            QLabel#LogoLabel {
                color: #4A3AFF;
                font-size: 18px;
                padding: 2px;
            }
            QLabel#TitleLabel {
                color: #E0E0FF;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI';
            }
            QPushButton#MenuButton {
                background: transparent;
                border: none;
                color: #E0E0FF;
                font-family: 'Segoe UI';
                font-size: 13px;
                padding: 5px 15px;
                border-radius: 4px;
                text-align: left;
            }
            QPushButton#MenuButton:hover {
                background: rgba(74, 58, 255, 0.2);
            }
            QPushButton#MenuButton:pressed {
                background: rgba(74, 58, 255, 0.3);
            }
            QPushButton#ThemeButton {
                background: transparent;
                border: 1px solid rgba(74, 58, 255, 0.3);
                border-radius: 15px;
                color: #E0E0FF;
                font-size: 16px;
                padding: 2px;
            }
            QPushButton#ThemeButton:hover {
                background: rgba(74, 58, 255, 0.2);
                border-color: rgba(74, 58, 255, 0.5);
            }
            QPushButton#ThemeButton:pressed {
                background: rgba(74, 58, 255, 0.3);
            }
            QPushButton#MinimizeButton, QPushButton#MaximizeButton, QPushButton#CloseButton {
                background: transparent;
                border: none;
                border-radius: 15px;
                color: #E0E0FF;
                font-family: 'Segoe UI';
                font-size: 14px;
                padding: 2px;
            }
            QPushButton#MinimizeButton:hover, QPushButton#MaximizeButton:hover {
                background: rgba(74, 58, 255, 0.2);
            }
            QPushButton#CloseButton:hover {
                background: #FF4444;
                color: white;
            }
        """)

        main_layout.addWidget(self.header)

        # Crear y agregar la barra de herramientas después del header
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        # Contenedor principal
        content_container = QFrame()
        content_container.setObjectName("ContentContainer")
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(25)

        # Video container
        video_container = QFrame()
        video_container.setObjectName("VideoContainer")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        welcome_message = "YOLO Vision Pro - Tomson"
        self.video_label = QLabel(welcome_message)
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        font = QFont("Segoe UI Light", 30, QFont.Weight.ExtraLight)
        self.video_label.setFont(font)
        
        video_layout.addWidget(self.video_label)

        # Controles de video mejorados
        self.video_controls = QFrame()
        self.video_controls.setObjectName("VideoControls")
        video_controls_layout = QVBoxLayout(self.video_controls)
        video_controls_layout.setContentsMargins(15, 10, 15, 10)
        video_controls_layout.setSpacing(8)

        # Barra de progreso y tiempo
        progress_container = QFrame()
        progress_container.setObjectName("ProgressContainer")
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)

        self.time_label_current = QLabel("00:00")
        self.time_label_current.setObjectName("TimeLabel")
        
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setObjectName("ProgressSlider")
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(1000)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderMoved.connect(self._on_slider_moved)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)
        
        self.time_label_total = QLabel("00:00")
        self.time_label_total.setObjectName("TimeLabel")

        progress_layout.addWidget(self.time_label_current)
        progress_layout.addWidget(self.progress_slider, 1)
        progress_layout.addWidget(self.time_label_total)

        # Botones de control
        controls_container = QFrame()
        controls_container.setObjectName("ControlsContainer")
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(15)

        # Botón de velocidad de reproducción
        self.speed_btn = QPushButton("1.0x")
        self.speed_btn.setObjectName("SpeedButton")
        self.speed_btn.clicked.connect(self._toggle_playback_speed)
        self._current_speed = 1.0

        # Botones principales
        self.prev_frame_btn = QPushButton()
        self.prev_frame_btn.setIcon(QIcon.fromTheme("media-skip-backward"))
        self.prev_frame_btn.setToolTip("Frame anterior")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_pause_btn.setToolTip("Reproducir/Pausar")
        self.play_pause_btn.clicked.connect(self._toggle_play_pause_media)
        
        self.next_frame_btn = QPushButton()
        self.next_frame_btn.setIcon(QIcon.fromTheme("media-skip-forward"))
        self.next_frame_btn.setToolTip("Frame siguiente")
        self.next_frame_btn.clicked.connect(self._next_frame)
        
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_btn.setToolTip("Detener")
        self.stop_btn.clicked.connect(self._stop_current_media)

        # Nuevo botón de reload
        self.reload_btn = QPushButton()
        self.reload_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.reload_btn.setToolTip("Recargar video")
        self.reload_btn.clicked.connect(self._reload_current_media)
        self.reload_btn.setObjectName("VideoControlButton")

        controls_layout.addStretch()
        for btn in [self.speed_btn, self.prev_frame_btn, self.play_pause_btn, 
                   self.next_frame_btn, self.stop_btn, self.reload_btn]:
            btn.setObjectName("VideoControlButton")
            controls_layout.addWidget(btn)
        controls_layout.addStretch()

        video_controls_layout.addWidget(progress_container)
        video_controls_layout.addWidget(controls_container)

        # Estilo para los controles de video
        self.video_controls.setStyleSheet("""
            QFrame#VideoControls {
                background-color: rgba(23, 23, 33, 0.95);
                border: 2px solid rgba(74, 58, 255, 0.2);
                border-radius: 15px;
                margin: 0px 15px;
            }
            QFrame#ProgressContainer, QFrame#ControlsContainer {
                background: transparent;
                border: none;
            }
            QLabel#TimeLabel {
                color: #E0E0FF;
                font-family: 'Segoe UI';
                font-size: 10pt;
                font-weight: bold;
                min-width: 60px;
            }
            QPushButton#VideoControlButton {
                background: rgba(74, 58, 255, 0.1);
                border: 2px solid rgba(74, 58, 255, 0.2);
                border-radius: 20px;
                padding: 10px;
                min-width: 40px;
                min-height: 40px;
            }
            QPushButton#VideoControlButton:hover {
                background: rgba(74, 58, 255, 0.2);
                border-color: rgba(74, 58, 255, 0.4);
            }
            QPushButton#VideoControlButton:pressed {
                background: rgba(74, 58, 255, 0.3);
                border-color: rgba(74, 58, 255, 0.6);
            }
            QPushButton#SpeedButton {
                background: rgba(74, 58, 255, 0.1);
                border: 2px solid rgba(74, 58, 255, 0.2);
                border-radius: 15px;
                padding: 5px 15px;
                color: #E0E0FF;
                font-weight: bold;
            }
            QPushButton#SpeedButton:hover {
                background: rgba(74, 58, 255, 0.2);
                border-color: rgba(74, 58, 255, 0.4);
            }
            QSlider#ProgressSlider {
                height: 30px;
            }
            QSlider#ProgressSlider::groove:horizontal {
                border: none;
                height: 6px;
                background: rgba(74, 58, 255, 0.1);
                margin: 0px;
                border-radius: 3px;
            }
            QSlider#ProgressSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                border: 2px solid #E0E0FF;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider#ProgressSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4A3AFF, stop:1 #3AFFED);
                border-radius: 3px;
            }
        """)

        video_container.setLayout(video_layout)
        content_layout.addWidget(video_container, 1)
        content_layout.addWidget(self.video_controls)

        # Info panel
        info_panel = QFrame()
        info_panel.setObjectName("InfoPanel")
        info_panel.setMaximumHeight(45)
        info_layout = QHBoxLayout(info_panel)
        info_layout.setContentsMargins(15, 8, 15, 8)

        self.info_label = QLabel("Sistema de seguridad iniciado. Seleccione una fuente.")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        info_font = QFont("Segoe UI", 10)
        info_font.setWeight(QFont.Weight.Medium)
        self.info_label.setFont(info_font)

        info_layout.addWidget(self.info_label)
        content_layout.addWidget(info_panel)

        main_layout.addWidget(content_container, 1)
        self.setCentralWidget(central_widget)

    def _create_toolbar(self):
        # Crear la barra de herramientas principal
        toolbar = QToolBar()
        toolbar.setObjectName("MainToolBar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setFixedHeight(60)  # Aumentado para mejor espaciado
        
        # Crear el contenedor para los grupos de botones
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(30)  # Aumentado el espacio entre grupos
        toolbar_layout.setContentsMargins(25, 5, 25, 5)  # Aumentados los márgenes
        
        # Grupo Archivo
        archivo_group = QFrame()
        archivo_group.setObjectName("ToolbarGroup")
        archivo_layout = QHBoxLayout(archivo_group)
        archivo_layout.setSpacing(10)  # Aumentado el espacio entre botones
        archivo_layout.setContentsMargins(0, 0, 0, 0)
        
        # Botón Abrir Imagen
        btn_imagen = QPushButton("Abrir Imagen")
        btn_imagen.setIcon(QIcon.fromTheme("document-open"))
        btn_imagen.setObjectName("ToolbarButton")
        btn_imagen.clicked.connect(self._select_image_file)
        archivo_layout.addWidget(btn_imagen)
        
        # Botón Abrir Video
        btn_video = QPushButton("Abrir Video")
        btn_video.setIcon(QIcon.fromTheme("video-x-generic"))
        btn_video.setObjectName("ToolbarButton")
        btn_video.clicked.connect(self._select_video_file)
        archivo_layout.addWidget(btn_video)
        
        # Grupo Cámara
        camara_group = QFrame()
        camara_group.setObjectName("ToolbarGroup")
        camara_layout = QHBoxLayout(camara_group)
        camara_layout.setSpacing(10)
        camara_layout.setContentsMargins(0, 0, 0, 0)
        
        # Botón Cámara Web
        btn_camara = QPushButton("Cámara Web")
        btn_camara.setIcon(QIcon.fromTheme("camera-web"))
        btn_camara.setObjectName("ToolbarButton")
        btn_camara.clicked.connect(self._start_webcam_mode)
        camara_layout.addWidget(btn_camara)
        
        # Grupo Control
        control_group = QFrame()
        control_group.setObjectName("ToolbarGroup")
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        # Botón Pausar
        self.btn_pausar = QPushButton("Pausar")
        self.btn_pausar.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.btn_pausar.setObjectName("ToolbarButton")
        self.btn_pausar.clicked.connect(self._toggle_play_pause_media)
        self.btn_pausar.setEnabled(False)
        control_layout.addWidget(self.btn_pausar)
        
        # Botón Detener
        self.btn_detener = QPushButton("Detener")
        self.btn_detener.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.btn_detener.setObjectName("ToolbarButton")
        self.btn_detener.clicked.connect(self._stop_current_media)
        self.btn_detener.setEnabled(False)
        control_layout.addWidget(self.btn_detener)
        
        # Agregar grupos al layout de la toolbar
        toolbar_layout.addWidget(archivo_group)
        toolbar_layout.addWidget(camara_group)
        toolbar_layout.addWidget(control_group)
        toolbar_layout.addStretch()
        
        # Crear un widget contenedor para el layout
        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar_layout)
        toolbar.addWidget(toolbar_widget)

        # Definir estilos para ambos temas
        style_dark = """
            QToolBar {
                background-color: #1a1a2e;
                border: none;
                padding: 5px;
                border-bottom: 1px solid #3A3A4C;
            }
            QFrame#ToolbarGroup {
                background: transparent;
            }
            QPushButton#ToolbarButton {
                background-color: #28283A;
                border: 1px solid #3A3A4C;
                border-radius: 6px;
                padding: 10px 20px;
                color: #E0E0FF;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton#ToolbarButton:hover {
                background-color: #323248;
                border-color: #4A3AFF;
            }
            QPushButton#ToolbarButton:pressed {
                background-color: #3A3A58;
            }
            QPushButton#ToolbarButton:disabled {
                background-color: #28283A;
                border-color: #3A3A4C;
                color: #666680;
            }
        """
        
        style_light = """
            QToolBar {
                background-color: #f8f9fa;
                border: none;
                padding: 5px;
                border-bottom: 1px solid #dee2e6;
            }
            QFrame#ToolbarGroup {
                background: transparent;
            }
            QPushButton#ToolbarButton {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px 20px;
                color: #495057;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton#ToolbarButton:hover {
                background-color: #e9ecef;
                border-color: #ced4da;
            }
            QPushButton#ToolbarButton:pressed {
                background-color: #dee2e6;
            }
            QPushButton#ToolbarButton:disabled {
                background-color: #e9ecef;
                border-color: #dee2e6;
                color: #adb5bd;
            }
        """
        
        # Guardar los estilos como atributos de la clase
        self.toolbar_style_dark = style_dark
        self.toolbar_style_light = style_light
        
        # Aplicar el estilo inicial según el modo
        toolbar.setStyleSheet(style_dark if self.dark_mode else style_light)
        
        self.toolbar = toolbar
        return toolbar

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
            if self.archivo_btn:
                self.archivo_btn.setEnabled(False)
            if self.camara_btn:
                self.camara_btn.setEnabled(False)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo.")

    def _update_button_states(self):
        """Actualiza el estado de todos los botones según el estado actual"""
        model_loaded = self.yolo_model is not None
        media_active = self.media_thread is not None and self.media_thread.isRunning()
        is_video = self.current_source_type == "video"

        # Actualizar estado de los botones del menú
        if self.archivo_btn:
            self.archivo_btn.setEnabled(model_loaded and not media_active)
        if self.camara_btn:
            self.camara_btn.setEnabled(model_loaded and not media_active)

        # Actualizar estado de los botones de la barra de herramientas
        if hasattr(self, 'btn_pausar'):
            self.btn_pausar.setEnabled(media_active)
            if media_active and self.media_thread._is_paused:
                self.btn_pausar.setText("Reanudar")
                self.btn_pausar.setIcon(QIcon.fromTheme("media-playback-start"))
            else:
                self.btn_pausar.setText("Pausar")
                self.btn_pausar.setIcon(QIcon.fromTheme("media-playback-pause"))

        if hasattr(self, 'btn_detener'):
            self.btn_detener.setEnabled(media_active)

        # Actualizar estado de los controles de video
        if is_video and media_active:
            # Habilitar todos los controles de video
            if hasattr(self, 'video_controls'):
                self.video_controls.setVisible(True)
                
                # Habilitar botones de control
                for control in ['play_pause_btn', 'stop_btn', 'prev_frame_btn', 
                              'next_frame_btn', 'speed_btn', 'reload_btn']:
                    if hasattr(self, control):
                        btn = getattr(self, control)
                        btn.setEnabled(True)
                
                # Habilitar slider y actualizar su estado
                if hasattr(self, 'progress_slider'):
                    self.progress_slider.setEnabled(True)
                
                # Actualizar estado del botón play/pause
                if self.play_pause_btn:
                    self.play_pause_btn.setEnabled(True)
                    if self.media_thread._is_paused:
                        self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
                        self.play_pause_btn.setToolTip("Reanudar")
                    else:
                        self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
                        self.play_pause_btn.setToolTip("Pausar")
        else:
            # Deshabilitar controles de video si no es video o no está activo
            if hasattr(self, 'video_controls'):
                self.video_controls.setVisible(is_video)
                
                # Deshabilitar todos los controles
                for control in ['play_pause_btn', 'stop_btn', 'prev_frame_btn', 
                              'next_frame_btn', 'speed_btn', 'reload_btn']:
                    if hasattr(self, control):
                        btn = getattr(self, control)
                        btn.setEnabled(False)
                
                if hasattr(self, 'progress_slider'):
                    self.progress_slider.setEnabled(False)

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
        """Limpia la pantalla y resetea los controles"""
        if self.video_label:
            welcome_message = "YOLO Vision Pro - Tomson"
            self.video_label.setText(welcome_message)
            self.video_label.setPixmap(QPixmap())
            font = QFont("Segoe UI Light", 30, QFont.Weight.ExtraLight)
            self.video_label.setFont(font)

        # Resetear controles de video
        if hasattr(self, 'progress_slider'):
            self.progress_slider.setValue(0)
        if hasattr(self, 'time_label_current'):
            self.time_label_current.setText("00:00")
        if hasattr(self, 'time_label_total'):
            self.time_label_total.setText("00:00")
        if hasattr(self, 'speed_btn'):
            self.speed_btn.setText("1.0x")
            self._current_speed = 1.0

        if hasattr(self, 'info_label') and self.info_label:
            self._set_info_label_style("normal", "Seleccione una fuente o inicie la cámara")

    def _stop_current_media_if_running(self):
        """Detiene el procesamiento actual si hay alguno en curso"""
        if self.media_thread and self.media_thread.isRunning():
            try:
                # Detener el hilo
                self.media_thread.stop()
                
                # Esperar a que el hilo termine (con timeout)
                if not self.media_thread.wait(2000):  # aumentado a 2 segundos
                    print("Advertencia: El hilo no se detuvo correctamente")
                    self.media_thread.terminate()  # Forzar terminación
                
                # Liberar recursos de la cámara si estaba activa
                if hasattr(self.media_thread, 'cap') and self.media_thread.cap:
                    self.media_thread.cap.release()
                
                self.media_thread = None
                
                # Limpiar la interfaz
                self._clear_display()
                if hasattr(self, 'video_controls'):
                    self.video_controls.setVisible(False)
                
                return True
            except Exception as e:
                print(f"Error al detener el medio actual: {e}")
                self.media_thread = None
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

                            detection_color = (79, 70, 229) # Indigo-600
                            text_bg_color = (67, 56, 202) # Indigo-700
                            text_fg_color = (255, 255, 255) # Blanco

                            cv2.rectangle(img_cv, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), detection_color, 2)
                            
                            # Fondo para el texto con estilo moderno
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                            cv2.rectangle(img_cv, (box_coords[0], box_coords[1] - h - 10), (box_coords[0] + w + 4, box_coords[1] - 5), text_bg_color, -1)
                            cv2.putText(img_cv, label, (box_coords[0] + 2, box_coords[1] - 7), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.6, text_fg_color, 1, cv2.LINE_AA)

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
        """Inicia un nuevo hilo de procesamiento de medios"""
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return

        try:
            # Asegurarse de que cualquier procesamiento anterior se ha detenido
            if self.media_thread and self.media_thread.isRunning():
                self._stop_current_media_if_running()
                # Si aún está corriendo después de intentar detenerlo, salir
                if self.media_thread and self.media_thread.isRunning():
                    QMessageBox.warning(self, "Error", "No se pudo detener el procesamiento anterior.")
                    return

            # Limpiar el estado actual
            self._clear_display()
            self.current_source_type = source_type
            self.current_media_path = file_path

            # Crear y configurar el nuevo hilo
            self.media_thread = MediaProcessingThread(self.yolo_model, source_type, file_path)
            
            # Conectar señales
            self.media_thread.frame_ready.connect(self._update_display_pixmap)
            self.media_thread.status_update.connect(self._update_status)
            self.media_thread.processing_finished.connect(self._on_media_processing_finished)
            self.media_thread.frame_position.connect(self._on_frame_position_update)
            self.media_thread.total_frames.connect(self._on_total_frames_update)

            # Actualizar la interfaz antes de iniciar
            self._update_video_controls_visibility()
            self._update_button_states()

            # Iniciar el procesamiento
            self.media_thread.start()

            # Actualizar estados después de un breve delay
            QTimer.singleShot(100, self._update_button_states)

        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Error al iniciar el procesamiento:\n{str(e)}")
            self.media_thread = None
            self._update_button_states()

    def _select_video_file(self):
        """Selecciona y procesa un archivo de video"""
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Seleccionar Video", "", 
                "Archivos de Video (*.mp4 *.avi *.mkv *.mov *.webm)"
            )

            if file_path:
                # Detener cualquier procesamiento activo y esperar
                if self._stop_current_media_if_running():
                    # Esperar más tiempo si venimos de la cámara web
                    delay = 500 if self.current_source_type == "webcam" else 200
                    QTimer.singleShot(delay, lambda: self._actually_start_video(file_path))
                else:
                    self._actually_start_video(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al seleccionar video:\n{str(e)}")

    def _actually_start_video(self, file_path):
        """Función interna para iniciar el video"""
        try:
            # Verificar nuevamente que no haya procesamiento activo
            if self.media_thread and self.media_thread.isRunning():
                QMessageBox.warning(self, "Error", "No se pudo detener el procesamiento anterior.")
                return

            # Mostrar y preparar controles de video
            if hasattr(self, 'video_controls'):
                self.video_controls.setVisible(True)
                # Resetear controles
                if hasattr(self, 'progress_slider'):
                    self.progress_slider.setValue(0)
                if hasattr(self, 'time_label_current'):
                    self.time_label_current.setText("00:00")
                if hasattr(self, 'time_label_total'):
                    self.time_label_total.setText("00:00")

            # Iniciar el video
            self._start_media_processing_thread("video", file_path)
            
            # Actualizar estado de los botones después de iniciar
            QTimer.singleShot(100, self._update_button_states)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar el video:\n{str(e)}")

    def _start_webcam_mode(self):
        """Inicia el modo de cámara web"""
        if not self.yolo_model:
            QMessageBox.warning(self, "Modelo no cargado", "El modelo YOLO aún no ha terminado de cargar.")
            return

        try:
            # Detener cualquier procesamiento activo y esperar a que termine
            if self._stop_current_media_if_running():
                # Esperar un momento para asegurar que todo se ha liberado
                QTimer.singleShot(500, self._actually_start_webcam)
            else:
                self._actually_start_webcam()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar la cámara web:\n{str(e)}")

    def _actually_start_webcam(self):
        """Función interna para iniciar la cámara web"""
        try:
            # Verificar nuevamente que no haya procesamiento activo
            if self.media_thread and self.media_thread.isRunning():
                QMessageBox.warning(self, "Error", "No se pudo detener el procesamiento anterior.")
                return

            # Ocultar controles de video si están visibles
            if hasattr(self, 'video_controls'):
                self.video_controls.setVisible(False)

            # Iniciar la cámara
            self._start_media_processing_thread("webcam")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar la cámara web:\n{str(e)}")

    def _toggle_play_pause_media(self):
        if self.media_thread and self.media_thread.isRunning():
            is_paused = self.media_thread.toggle_pause()
            # Actualizar icono y tooltip del botón de play/pause
            if is_paused:
                self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
                self.play_pause_btn.setToolTip("Reanudar")
                self.btn_pausar.setText("Reanudar")
                self.btn_pausar.setIcon(QIcon.fromTheme("media-playback-start"))
            else:
                self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
                self.play_pause_btn.setToolTip("Pausar")
                self.btn_pausar.setText("Pausar")
                self.btn_pausar.setIcon(QIcon.fromTheme("media-playback-pause"))
        self._update_button_states()

    def _stop_current_media(self):
        if self._stop_current_media_if_running():
            self.status_bar.showMessage("Deteniendo procesamiento...", 2000)
            # Resetear controles de video
            if hasattr(self, 'progress_slider'):
                self.progress_slider.setValue(0)
            if hasattr(self, 'time_label_current'):
                self.time_label_current.setText("00:00")
            if hasattr(self, 'time_label_total'):
                self.time_label_total.setText("00:00")
            if hasattr(self, 'speed_btn'):
                self.speed_btn.setText("1.0x")
                self._current_speed = 1.0
        else:
            self._update_status("No hay procesamiento activo para detener.")
            self._clear_display()
            self.media_thread = None 
        self._update_button_states()

    def _on_slider_moved(self):
        """Maneja el movimiento del slider de progreso"""
        if self.media_thread and self.media_thread.source_type == "video":
            try:
                value = self.progress_slider.value()
                total_frames = self.media_thread.total_frame_count
                if total_frames > 0:
                    frame = int((value / 1000.0) * total_frames)
                    duration = self.media_thread.get_video_duration()
                    current_time = (value / 1000.0) * duration
                    self.time_label_current.setText(self._format_time(current_time))
            except Exception as e:
                print(f"Error al mover el slider: {e}")

    def _on_slider_released(self):
        """Maneja cuando se suelta el slider de progreso"""
        if self.media_thread and self.media_thread.source_type == "video":
            try:
                value = self.progress_slider.value()
                total_frames = self.media_thread.total_frame_count
                if total_frames > 0:
                    frame = int((value / 1000.0) * total_frames)
                    self.media_thread.seek_to_frame(frame)
            except Exception as e:
                print(f"Error al soltar el slider: {e}")

    @pyqtSlot(int)
    def _on_frame_position_update(self, frame_position):
        """Actualiza la posición del slider y el tiempo actual"""
        if self.media_thread and not self.progress_slider.isSliderDown():
            try:
                total_frames = self.media_thread.total_frame_count
                if total_frames > 0:
                    value = int((frame_position / total_frames) * 1000)
                    self.progress_slider.setValue(value)
                    duration = self.media_thread.get_video_duration()
                    current_time = (frame_position / total_frames) * duration
                    self.time_label_current.setText(self._format_time(current_time))
            except Exception as e:
                print(f"Error al actualizar posición: {e}")

    def _prev_frame(self):
        """Retrocede un frame en el video"""
        if self.media_thread and self.media_thread.source_type == "video":
            try:
                current_frame = self.media_thread.current_frame
                if current_frame > 0:
                    # Pausar el video si está reproduciendo
                    if not self.media_thread._is_paused:
                        self._toggle_play_pause_media()
                    self.media_thread.seek_to_frame(current_frame - 1)
            except Exception as e:
                print(f"Error al retroceder frame: {e}")

    def _next_frame(self):
        """Avanza un frame en el video"""
        if self.media_thread and self.media_thread.source_type == "video":
            try:
                current_frame = self.media_thread.current_frame
                if current_frame < self.media_thread.total_frame_count - 1:
                    # Pausar el video si está reproduciendo
                    if not self.media_thread._is_paused:
                        self._toggle_play_pause_media()
                    self.media_thread.seek_to_frame(current_frame + 1)
            except Exception as e:
                print(f"Error al avanzar frame: {e}")

    def _toggle_playback_speed(self):
        """Cambia la velocidad de reproducción del video"""
        if not hasattr(self, '_current_speed'):
            self._current_speed = 1.0
        
        try:
            speeds = [0.25, 0.5, 1.0, 1.5, 2.0]
            current_index = speeds.index(self._current_speed)
            next_index = (current_index + 1) % len(speeds)
            self._current_speed = speeds[next_index]
            self.speed_btn.setText(f"{self._current_speed}x")
            
            if self.media_thread and self.media_thread.isRunning():
                # Aquí se podría implementar el cambio de velocidad real
                pass
        except Exception as e:
            print(f"Error al cambiar velocidad: {e}")

    @pyqtSlot(int)
    def _on_total_frames_update(self, total_frames):
        """Actualiza la duración total del video"""
        if self.media_thread:
            try:
                duration = self.media_thread.get_video_duration()
                if duration > 0:
                    self.time_label_total.setText(self._format_time(duration))
                else:
                    self.time_label_total.setText("00:00")
            except Exception as e:
                print(f"Error al actualizar frames totales: {e}")

    def _update_video_controls_visibility(self):
        """Actualiza la visibilidad de los controles de video según el estado actual"""
        try:
            is_video = self.current_source_type == "video"
            is_media_active = self.media_thread is not None and self.media_thread.isRunning()
            
            # Verificar que los controles existan antes de usarlos
            if hasattr(self, 'video_controls'):
                self.video_controls.setVisible(bool(is_video))  # Mostrar solo para videos
                if is_video:
                    self.progress_slider.setValue(0)
                    self.time_label_current.setText("00:00")
                    self.time_label_total.setText("00:00")
            
            # Actualizar estado de los botones
            for control in ['play_pause_btn', 'stop_btn', 'prev_frame_btn', 
                          'next_frame_btn', 'speed_btn', 'reload_btn']:
                if hasattr(self, control):
                    btn = getattr(self, control)
                    btn.setEnabled(bool(is_video and is_media_active))
            
            # Actualizar el slider y las etiquetas de tiempo
            if hasattr(self, 'progress_slider'):
                self.progress_slider.setEnabled(bool(is_video and is_media_active))
                if not (is_video and is_media_active):
                    self.progress_slider.setValue(0)
            
            for label in ['time_label_current', 'time_label_total']:
                if hasattr(self, label):
                    lbl = getattr(self, label)
                    if not (is_video and is_media_active):
                        lbl.setText("00:00")

        except Exception as e:
            print(f"Error al actualizar controles de video: {e}")

    def _on_media_processing_finished(self):
        final_message = "Procesamiento finalizado."
        if self.current_source_type == "video":
            final_message = "Procesamiento de video finalizado."
        elif self.current_source_type == "webcam":
            final_message = "Cámara detenida."
        
        self._update_status(final_message)
        self.media_thread = None
        self._update_button_states()
        self._update_video_controls_visibility()

    def _get_theme_style(self, colors):
        return f"""
            /* Estilo general de la ventana */
            QMainWindow {{
                background-color: {colors['bg']};
                color: {colors['text']};
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}
            QWidget {{
                color: {colors['text']};
                font-size: 10pt;
            }}

            /* Header */
            QFrame#Header {{
                background: {colors['bg']};
                border-bottom: 2px solid {colors['border']};
            }}
            QLabel#LogoLabel {{
                color: {colors['accent']};
                font-size: 18px;
                padding: 2px;
            }}
            QLabel#TitleLabel {{
                color: {colors['text']};
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#MenuButton {{
                background: transparent;
                border: none;
                color: {colors['text']};
                font-size: 13px;
                padding: 5px 15px;
                border-radius: 4px;
            }}
            QPushButton#MenuButton:hover {{
                background: {colors['hover']};
            }}
            QPushButton#MenuButton:pressed {{
                background: {colors['pressed']};
            }}
            QPushButton#ThemeButton {{
                background: transparent;
                border: 1px solid {colors['border']};
                border-radius: 15px;
                color: {colors['text']};
                font-size: 16px;
                padding: 2px;
            }}
            QPushButton#ThemeButton:hover {{
                background: {colors['hover']};
                border-color: {colors['accent']};
            }}
            QPushButton#MinimizeButton, 
            QPushButton#MaximizeButton, 
            QPushButton#CloseButton {{
                background: transparent;
                border: none;
                border-radius: 15px;
                color: {colors['text']};
                font-size: 14px;
                padding: 2px;
            }}
            QPushButton#MinimizeButton:hover, 
            QPushButton#MaximizeButton:hover {{
                background: {colors['hover']};
            }}
            QPushButton#CloseButton:hover {{
                background: #FF4444;
                color: white;
            }}

            /* Barra de herramientas */
            QToolBar {{
                background-color: {colors['bg']};
                border: none;
                padding: 5px;
                border-bottom: 1px solid {colors['border']};
            }}
            QPushButton#ToolbarButton {{
                background-color: {colors['secondary']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 10px 20px;
                color: {colors['text']};
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton#ToolbarButton:hover {{
                background-color: {colors['hover']};
                border-color: {colors['accent']};
            }}
            QPushButton#ToolbarButton:pressed {{
                background-color: {colors['pressed']};
            }}
            QPushButton#ToolbarButton:disabled {{
                background-color: {colors['secondary']};
                border-color: {colors['border']};
                color: {colors['disabled']};
            }}

            /* Área de video */
            QLabel#VideoLabel {{
                background-color: {colors['secondary']};
                border: 2px solid {colors['border']};
                border-radius: 15px;
                color: {colors['text']};
                padding: 8px;
            }}

            /* Controles de video */
            QFrame#VideoControls {{
                background-color: {colors['bg']};
                border: 2px solid {colors['border']};
                border-radius: 15px;
            }}
            QLabel#TimeLabel {{
                color: {colors['text']};
                font-weight: bold;
                min-width: 60px;
            }}
            QPushButton#VideoControlButton {{
                background: {colors['secondary']};
                border: 2px solid {colors['border']};
                border-radius: 20px;
                padding: 10px;
                min-width: 40px;
                min-height: 40px;
                color: {colors['text']};
            }}
            QPushButton#VideoControlButton:hover {{
                background: {colors['hover']};
                border-color: {colors['accent']};
            }}
            QPushButton#VideoControlButton:pressed {{
                background: {colors['pressed']};
            }}
            QPushButton#SpeedButton {{
                background: {colors['secondary']};
                border: 2px solid {colors['border']};
                border-radius: 15px;
                padding: 5px 15px;
                color: {colors['text']};
                font-weight: bold;
            }}
            QPushButton#SpeedButton:hover {{
                background: {colors['hover']};
                border-color: {colors['accent']};
            }}
            QSlider#ProgressSlider {{
                height: 30px;
            }}
            QSlider#ProgressSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {colors['border']};
                margin: 0px;
                border-radius: 3px;
            }}
            QSlider#ProgressSlider::handle:horizontal {{
                background: {colors['accent']};
                border: 2px solid {colors['text']};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider#ProgressSlider::sub-page:horizontal {{
                background: {colors['accent']};
                border-radius: 3px;
            }}

            /* Panel de información */
            QFrame#InfoPanel {{
                background-color: {colors['secondary']};
                border-radius: 8px;
                border: 1px solid {colors['border']};
            }}
            QLabel#InfoLabel {{
                color: {colors['text_secondary']};
                font-size: 10pt;
                font-weight: 600;
            }}

            /* Barra de estado */
            QStatusBar {{
                background-color: {colors['bg']};
                color: {colors['text_secondary']};
                font-size: 9pt;
                font-weight: 500;
                border-top: 1px solid {colors['border']};
            }}

            /* Menús */
            QMenu {{
                background-color: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 8px;
            }}
            QMenu::item {{
                color: {colors['text']};
                padding: 8px 20px;
                border-radius: 4px;
                margin: 4px;
            }}
            QMenu::item:selected {{
                background: {colors['hover']};
                color: {colors['text']};
            }}
            QMenu::separator {{
                height: 1px;
                background: {colors['border']};
                margin: 6px 12px;
            }}
        """

    def _show_theme_menu(self):
        menu = QMenu(self)
        
        # Temas predefinidos con colores completos y mejorados
        themes = {
            "🌙 Oscuro": {
                "bg": "#1a1a2e",
                "accent": "#4A3AFF",
                "secondary": "#28283A",
                "border": "#3A3A4C",
                "text": "#E0E0FF",
                "text_secondary": "#B0B0C0",
                "disabled": "#666680",
                "hover": "#323248",
                "pressed": "#3A3A58"
            },
            "☀️ Claro": {
                "bg": "#FFFFFF",
                "accent": "#6366F1",
                "secondary": "#F8F9FA",
                "border": "#E2E8F0",
                "text": "#1E293B",
                "text_secondary": "#64748B",
                "disabled": "#CBD5E1",
                "hover": "#F1F5F9",
                "pressed": "#E2E8F0"
            },
            "🌺 Rosa": {
                "bg": "#2D1A2E",
                "accent": "#FF3AFF",
                "secondary": "#3A1A3C",
                "border": "#4C1A4E",
                "text": "#FFE0FF",
                "text_secondary": "#FFB0FF",
                "disabled": "#806680",
                "hover": "#4C2A4E",
                "pressed": "#5C3A5E"
            },
            "🌊 Océano": {
                "bg": "#1A2D2E",
                "accent": "#3AFFFF",
                "secondary": "#1A3A3C",
                "border": "#1A4C4E",
                "text": "#E0FFFF",
                "text_secondary": "#B0FFFF",
                "disabled": "#668080",
                "hover": "#2A4D4E",
                "pressed": "#3A5D5E"
            },
            "🍃 Bosque": {
                "bg": "#1A2E1A",
                "accent": "#3AFF3A",
                "secondary": "#1A3C1A",
                "border": "#1A4E1A",
                "text": "#E0FFE0",
                "text_secondary": "#B0FFB0",
                "disabled": "#668066",
                "hover": "#2A4E2A",
                "pressed": "#3A5E3A"
            },
            "🌅 Atardecer": {
                "bg": "#2E1A1A",
                "accent": "#FF3A3A",
                "secondary": "#3C1A1A",
                "border": "#4E1A1A",
                "text": "#FFE0E0",
                "text_secondary": "#FFB0B0",
                "disabled": "#806666",
                "hover": "#4E2A2A",
                "pressed": "#5E3A3A"
            }
        }

        # Aplicar el estilo del menú según el tema actual
        menu.setStyleSheet(self._get_theme_style(self.theme_colors if hasattr(self, 'theme_colors') else themes["🌙 Oscuro"]))

        for theme_name, colors in themes.items():
            action = QAction(theme_name, self)
            action.setData(colors)
            action.triggered.connect(lambda checked, t=theme_name, c=colors: self._apply_theme(t, c))
            menu.addAction(action)

        button = self.sender()
        if button:
            pos = button.mapToGlobal(button.rect().bottomLeft())
            menu.exec(pos)

    def _apply_theme(self, theme_name, colors):
        self.dark_mode = colors["bg"].startswith("#1") or colors["bg"].startswith("#2")
        self.theme_colors = colors
        
        # Actualizar el ícono del botón de tema
        self.theme_btn.setText(theme_name.split()[0])
        
        # Aplicar el estilo global
        style = self._get_theme_style(colors)
        self.setStyleSheet(style)

        # Actualizar estilos específicos de componentes
        if hasattr(self, 'toolbar'):
            self.toolbar.setStyleSheet(style)
        
        # Actualizar el estilo del header
        if hasattr(self, 'header'):
            self.header.setStyleSheet(style)

        # Actualizar los controles de video
        if hasattr(self, 'video_controls'):
            self.video_controls.setStyleSheet(style)
            
        # Actualizar el panel de información
        if hasattr(self, 'info_label'):
            self.info_label.setStyleSheet(style)

        # Actualizar la barra de estado
        if hasattr(self, 'status_bar'):
            self.status_bar.setStyleSheet(style)

        # Forzar la actualización visual de los widgets
        self.repaint()

        # Actualizar el video_label
        if hasattr(self, 'video_label'):
            if not (hasattr(self, 'media_thread') and self.media_thread and self.media_thread.isRunning()):
                self._clear_display()

        # Actualizar los menús
        for menu in self.findChildren(QMenu):
            menu.setStyleSheet(style)

        # Actualizar todos los botones
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(style)

        # Actualizar todos los frames
        for frame in self.findChildren(QFrame):
            frame.setStyleSheet(style)

        # Actualizar todos los labels
        for label in self.findChildren(QLabel):
            label.setStyleSheet(style)

        # Actualizar todos los sliders
        for slider in self.findChildren(QSlider):
            slider.setStyleSheet(style)

        self.status_bar.showMessage(f"Tema {theme_name} aplicado.", 3000)

    def _recreate_toolbar(self):
        # Eliminar la barra de herramientas existente si existe
        old_toolbar = self.findChild(QToolBar, "MainToolBar")
        if old_toolbar:
            self.removeToolBar(old_toolbar)
            old_toolbar.deleteLater()
        
        # Volver a crear la barra de herramientas
        self._create_toolbar()

    def _format_time(self, seconds):
        """Formatea el tiempo en formato MM:SS"""
        try:
            minutes = int(seconds / 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
        except Exception as e:
            print(f"Error al formatear tiempo: {e}")
            return "00:00"

    def closeEvent(self, event):
        """Maneja el cierre de la aplicación"""
        try:
            self.status_bar.showMessage("Cerrando aplicación...", 2000)
            QApplication.processEvents()  # Procesar eventos pendientes
            
            # Detener cualquier procesamiento activo
            if self.media_thread and self.media_thread.isRunning():
                self.media_thread.stop()
                if not self.media_thread.wait(1000):  # espera máximo 1 segundo
                    self.media_thread.terminate()  # Forzar terminación si es necesario
            
            print("Aplicación cerrada correctamente.")
            event.accept()
        except Exception as e:
            print(f"Error al cerrar la aplicación: {e}")
            event.accept()  # Aceptar el cierre incluso si hay error

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() <= 40:  # Altura de la barra de título
                self._is_dragging = True
                self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False

    def _show_archivo_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a2e;
                border: 1px solid rgba(74, 58, 255, 0.3);
                border-radius: 5px;
                padding: 5px;
            }
            QMenu::item {
                color: #E0E0FF;
                padding: 5px 20px;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 13px;
            }
            QMenu::item:selected {
                background: rgba(74, 58, 255, 0.2);
            }
            QMenu::separator {
                height: 1px;
                background: rgba(74, 58, 255, 0.3);
                margin: 5px 0px;
            }
        """)
        
        abrir_imagen = QAction("Abrir Imagen", self)
        abrir_imagen.triggered.connect(self._select_image_file)
        menu.addAction(abrir_imagen)

        abrir_video = QAction("Abrir Video", self)
        abrir_video.triggered.connect(self._select_video_file)
        menu.addAction(abrir_video)

        menu.addSeparator()

        salir = QAction("Salir", self)
        salir.triggered.connect(self.close)
        menu.addAction(salir)
        
        # Mostrar menú bajo el botón
        button = self.sender()
        if button:
            pos = button.mapToGlobal(button.rect().bottomLeft())
            menu.exec(pos)

    def _show_camara_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a2e;
                border: 1px solid rgba(74, 58, 255, 0.3);
                border-radius: 5px;
                padding: 5px;
            }
            QMenu::item {
                color: #E0E0FF;
                padding: 5px 20px;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 13px;
            }
            QMenu::item:selected {
                background: rgba(74, 58, 255, 0.2);
            }
        """)
        
        iniciar_camara = QAction("Iniciar Cámara Web", self)
        iniciar_camara.triggered.connect(self._start_webcam_mode)
        menu.addAction(iniciar_camara)
        
        # Mostrar menú bajo el botón
        button = self.sender()
        if button:
            pos = button.mapToGlobal(button.rect().bottomLeft())
            menu.exec(pos)

    def _reload_current_media(self):
        """Recarga el video actual desde el principio"""
        if self.current_source_type == "video" and self.current_media_path:
            self._start_media_processing_thread("video", self.current_media_path)

# --- Punto de Entrada ---
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)

        # Configurar información de la aplicación
        app.setApplicationName("YOLO Vision Pro(Tomson)")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Security Systems")
        app.setOrganizationDomain("securitysystems.com")

        # Verificar dependencias críticas
        dependencies_ok = True
        missing_deps = []

        try:
            import cv2
        except ImportError:
            dependencies_ok = False
            missing_deps.append("opencv-python")

        try:
            from ultralytics import YOLO
        except ImportError:
            dependencies_ok = False
            missing_deps.append("ultralytics")

        try:
            from PIL import Image
        except ImportError:
            dependencies_ok = False
            missing_deps.append("Pillow")

        if not dependencies_ok:
            from PyQt6.QtWidgets import QMessageBox
            error_msg = "Faltan las siguientes dependencias:\n\n"
            error_msg += "\n".join([f"- {dep}" for dep in missing_deps])
            error_msg += "\n\nPor favor, instálalas usando:\n"
            error_msg += "pip install " + " ".join(missing_deps)
            QMessageBox.critical(None, "Error de Dependencias", error_msg)
            sys.exit(1)

        # Verificar si el modelo YOLO está disponible
        model_path = 'yolov8n.pt'
        if not os.path.exists(model_path):
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Descarga de Modelo")
            msg.setText("El modelo YOLOv8n no está presente.")
            msg.setInformativeText("¿Deseas descargarlo ahora? (Aproximadamente 6MB)")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if msg.exec() == QMessageBox.StandardButton.Yes:
                try:
                    # Descargar el modelo
                    YOLO("yolov8n")
                except Exception as e:
                    QMessageBox.critical(None, "Error de Descarga",
                                      f"No se pudo descargar el modelo:\n{str(e)}")
                    sys.exit(1)
            else:
                QMessageBox.information(None, "Información",
                                      "La aplicación necesita el modelo para funcionar.\nCerrando aplicación.")
                sys.exit(0)

        # Crear y mostrar la ventana principal
        main_win = MainWindow()
        main_win.show()

        # Configurar manejo de excepciones no capturadas
        def exception_hook(exctype, value, traceback):
            print('Exception:', exctype, value, traceback)
            sys.__excepthook__(exctype, value, traceback)
            if main_win:
                error_msg = f"Error inesperado:\n{str(value)}"
                QMessageBox.critical(main_win, "Error", error_msg)

        sys.excepthook = exception_hook

        # Ejecutar la aplicación
        sys.exit(app.exec())

    except Exception as e:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Error Crítico",
                           f"Error al iniciar la aplicación:\n{str(e)}")
        sys.exit(1)
