"""

Parte 1: Grabación del Dataset (record-dataset.py)
El primer script captura imágenes de la mano usando la cámara web y utiliza MediaPipe para detectar los puntos clave.
Los datos de los landmarks se almacenan junto con una etiqueta correspondiente a cada gesto (0 para Piedra, 1 para Papel, 2 para Tijeras).
"""

import cv2
import numpy as np
import mediapipe as mp

# Configuración de MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

data, labels = [], []

def capture_gesture(label):
    """Captura gestos y guarda los landmarks junto con la etiqueta."""
    print(f"Grabando gesto {label}. Presiona 'q' para detener.")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                data.append(landmarks)
                labels.append(label)

                # Dibujar los puntos en la mano detectada
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )

        cv2.imshow('Grabación de Gestos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Captura de gestos (0: Piedra, 1: Papel, 2: Tijeras)
for i, gesture in enumerate(["Piedra (0)", "Papel (1)", "Tijeras (2)"]):
    input(f"Presiona Enter para grabar el gesto: {gesture}")
    capture_gesture(i)

# Guardar los datos y etiquetas
cap.release()
cv2.destroyAllWindows()
np.save('rps_dataset.npy', np.array(data))
np.save('rps_labels.npy', np.array(labels))
print("Dataset guardado con éxito en 'rps_dataset.npy' y 'rps_labels.npy'.")

