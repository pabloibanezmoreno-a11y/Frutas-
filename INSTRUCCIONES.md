# Instrucciones de Ejecución Técnicas

Siga estos pasos para desplegar el sistema de inferencia de clasificación de frutas.

## 1. Preparación del Entorno (Terminal)

Se recomienda el uso de un entorno virtual para evitar conflictos de dependencias.

```bash
# Navegar al directorio del proyecto
cd (Ruta donde esté el proyecto)

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
.\venv\Scripts\activate
# En Linux/Jetson Nano:
source venv/bin/activate

# Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install opencv-python numpy
```

## 2. Configuración de Rutas

**IMPORTANTE:** El script `frutas.py` contiene rutas absolutas configuradas para un entorno específico (`/home/hadoop/frutas/`). Antes de ejecutar, asegúrese de que los archivos `fruits_resnet18_NUEVO.onnx` y `labels.txt` se encuentren en la ruta definida en el script o ajuste las variables `MODEL_PATH` y `LABELS_PATH` dentro de `frutas.py` si es necesario (aunque se recomienda mantener el original por compatibilidad).

## 3. Lanzamiento del Sistema

Ejecute el script de inferencia:

```bash
python frutas.py
```

## 4. Controles e Interfaz
- Al iniciar, se abrirá una ventana mostrando la señal de la cámara.
- Coloque la fruta dentro del recuadro verde ("Zona de escaneo").
- El sistema mostrará el nombre de la fruta y el porcentaje de confianza en la esquina superior izquierda.
- Presione la tecla **'q'** para cerrar la aplicación de forma segura.

## Consideraciones para NVIDIA Jetson Nano
Si ejecuta en una Jetson Nano y experimenta errores con la interfaz gráfica:
1. Asegúrese de tener un monitor conectado o use `X11 forwarding`.
2. Si falta el módulo de GTK, instálelo con:
   ```bash
   sudo apt-get install libcanberra-gtk-module
   ```
3. Para mejorar el rendimiento, puede cambiar la preferencia del backend en `frutas.py` (si OpenCV fue compilado con CUDA):
   ```python
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
   ```
