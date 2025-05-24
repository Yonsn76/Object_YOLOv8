# YOLO Vision Pro 🎯

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Descripción

YOLO Vision Pro es una aplicación de escritorio moderna para la detección de objetos en tiempo real. Diseñada para ser fácil de usar pero potente, esta aplicación combina la precisión del modelo YOLOv8 con una interfaz gráfica elegante construida en PyQt6.

### 🎯 ¿Para qué sirve?
- Detecta objetos en tiempo real usando la cámara web
- Analiza imágenes estáticas
- Procesa videos grabados
- Identifica múltiples objetos simultáneamente
- Muestra la confianza de cada detección

## ✨ Características Principales

### 🎥 Capacidades de Detección
- Detección en tiempo real con cámara web
- Análisis de imágenes (.jpg, .png, .jpeg)
- Procesamiento de videos (.mp4, .avi, .mkv)
- Detección de múltiples clases de objetos

### 🎨 Interfaz de Usuario
- Diseño moderno y profesional
- Temas claro y oscuro
- Controles intuitivos
- Visualización en tiempo real

### 🛠️ Funcionalidades Técnicas
- Procesamiento rápido y eficiente
- Control de velocidad de reproducción
- Navegación frame por frame
- Barra de progreso interactiva

## 📦 Guía de Instalación Completa

### Paso 1: Preparar el Entorno

1. **Instalar Python**:
   - Descarga Python 3.x desde [python.org](https://www.python.org/downloads/)
   - ✅ IMPORTANTE: Marca "Add Python to PATH" durante la instalación
   - Verifica la instalación:
     ```bash
     python --version
     ```

2. **Instalar Git**:
   - Descarga Git desde [git-scm.com](https://git-scm.com/downloads)
   - Verifica la instalación:
     ```bash
     git --version
     ```

### Paso 2: Obtener el Código

1. **Abrir Terminal**:
   ```bash
   # Windows
   Windows + R, escribe 'cmd', Enter

   # Mac
   Command + Space, escribe 'terminal', Enter
   ```

2. **Clonar el Proyecto**:
   ```bash
   # Navegar a tu carpeta preferida
   cd Documents

   # Clonar el repositorio
   git clone https://github.com/Yonsn76/Object_YOLOv8.git
   cd Object_YOLOv8
   ```

### Paso 3: Configurar el Entorno Virtual

1. **Crear Entorno Virtual**:
   ```bash
   # Windows
   python -m venv venv

   # Mac/Linux
   python3 -m venv venv
   ```

2. **Activar el Entorno**:
   ```bash
   # Windows (CMD)
   venv\Scripts\activate

   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1

   # Mac/Linux
   source venv/bin/activate
   ```

### Paso 4: Instalar Dependencias

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar requisitos
pip install -r requirements.txt
```

### Paso 5: Ejecutar la Aplicación

```bash
python recognition.py
```

## 🚀 Guía de Uso

### 1. Iniciar la Aplicación
- Ejecuta el programa como se indicó arriba
- Espera a que se cargue el modelo YOLOv8
- La interfaz principal aparecerá

### 2. Funciones Principales
- **Analizar Imagen**:
  1. Clic en "Abrir Imagen"
  2. Selecciona una imagen
  3. Espera el análisis

- **Procesar Video**:
  1. Clic en "Abrir Video"
  2. Selecciona un video
  3. Usa los controles de reproducción

- **Usar Cámara Web**:
  1. Clic en "Cámara Web"
  2. Permite el acceso a la cámara
  3. La detección comenzará automáticamente

### 3. Controles Adicionales
- **Reproducción de Video**:
  - ⏯️ Play/Pause
  - ⏹️ Stop
  - ⏪ Frame anterior
  - ⏩ Frame siguiente
  - 🔄 Cambiar velocidad

- **Personalización**:
  - 🎨 Cambiar tema
  - 📊 Ajustar visualización

## 🔧 Solución de Problemas

### Errores Comunes

1. **"Python no encontrado"**
   - Solución: Reinstalar Python marcando "Add to PATH"
   - Reiniciar la computadora

2. **"Git no encontrado"**
   - Solución: Reinstalar Git
   - Asegurar acceso desde terminal

3. **Error de Entorno Virtual**
   - En PowerShell:
     ```powershell
     Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```

4. **Errores de Dependencias**
   ```bash
   pip install opencv-python
   pip install numpy
   pip install PyQt6
   pip install ultralytics
   pip install Pillow
   ```

## 📁 Estructura del Proyecto

```
Object_YOLOv8/
├── recognition.py     # Programa principal
├── requirements.txt   # Lista de dependencias
└── yolov8n.pt        # Modelo de IA (se descarga auto.)
```

Desarrollado con ❤️ por Yonsn76
