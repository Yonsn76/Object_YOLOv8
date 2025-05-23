YOLO Vision Pro - Detector de Objetos con YOLOv8
YOLO Vision Pro es una aplicación de escritorio para la detección de objetos en imágenes, videos y transmisiones de cámara web, utilizando YOLOv8 y PyQt6.

Características
Detección en imágenes, videos y cámara web.

Interfaz gráfica con temas claro/oscuro.

Controles para reproducir, pausar y detener.

Usa el modelo yolov8n.pt.

Estructura del Proyecto
Object_YOLOv8/
├── recognition.py     # Script principal
├── requirements.txt   # Dependencias
└── yolov8n.pt         # Modelo YOLO

Requisitos
Python 3.x

opencv-python

numpy

PyQt6

ultralytics

Pillow (Recomendado)

Instalación
Clona el repositorio:

git clone https://github.com/Yonsn76/Object_YOLOv8.git

Instala las dependencias:

pip install -r requirements.txt

(Asegúrate de tener yolov8n.pt en el directorio).

Uso
Ejecuta la aplicación con:

python recognition.py

Usa el menú o la barra de herramientas para:

Abrir una imagen.

Abrir un video.

Iniciar la cámara web.

Cambiar el tema visual.

Licencia
Este proyecto es de código abierto (Licencia MIT propuesta).

Creado por Yonsn76