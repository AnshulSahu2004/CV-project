import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

def detect_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0]

    h, w, _ = frame.shape
    points = []

    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (224,224))

    return face