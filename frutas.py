import cv2
import numpy as np
import os

# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
MODEL_PATH = "/home/hadoop/frutas/fruits_resnet18_NUEVO.onnx"
LABELS_PATH = "/home/hadoop/frutas/labels.txt"

CAMERA_ID = 0
ROI_SIZE = 300
UMBRAL = 0.50

# =========================
# VERIFICACIÓN Y CARGA
# =========================
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print("[ERROR] Archivos no encontrados.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Etiquetas: {LABELS_PATH}")
    exit()

with open(LABELS_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(f"[INFO] Clases cargadas: {classes}")

try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print("[INFO] Modelo ONNX cargado correctamente en OpenCV.")
except Exception as e:
    print("\n[ERROR CRÍTICO] Falló la carga del modelo ONNX.")
    print("MOTIVO PROBABLE: El archivo pesa 87K en lugar de 45MB. El archivo está corrupto o vacío.")
    print(f"Detalle del error técnico: {e}")
    exit()

# =========================
# BUCLE DE CÁMARA
# =========================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("[ERROR] No se detecta señal en la cámara.")
    exit()

print("[INFO] Iniciando inferencia. Pulsa 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Definir ROI central
    x1 = max(0, int(w / 2 - ROI_SIZE / 2))
    y1 = max(0, int(h / 2 - ROI_SIZE / 2))
    x2 = min(w, x1 + ROI_SIZE)
    y2 = min(h, y1 + ROI_SIZE)
    
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # =========================
    # PREPROCESAMIENTO PARA RESNET
    # =========================
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(roi_rgb, 1.0 / 255.0, (224, 224), swapRB=False, crop=False)

    blob[0, 0, :, :] = (blob[0, 0, :, :] - 0.485) / 0.229
    blob[0, 1, :, :] = (blob[0, 1, :, :] - 0.456) / 0.224
    blob[0, 2, :, :] = (blob[0, 2, :, :] - 0.406) / 0.225

    # =========================
    # INFERENCIA
    # =========================
    net.setInput(blob)
    out = net.forward()

    exp_out = np.exp(out[0] - np.max(out[0]))
    probs = exp_out / np.sum(exp_out)

    class_id = np.argmax(probs)
    confidence = probs[class_id]
    label = classes[class_id]

    # =========================
    # RENDERIZADO VISUAL
    # =========================
    color = (0, 255, 0) if confidence >= UMBRAL else (0, 0, 255)
    texto = f"{label if confidence >= UMBRAL else 'Desconocido'} ({confidence * 100:.1f}%)"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "Zona de escaneo", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Inferencia Frutas - OpenCV DNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

