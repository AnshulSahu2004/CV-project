import cv2
import numpy as np
import torch
from collections import deque
from PIL import Image
from torchvision import transforms

from liveliness_classification import LivenessNet
from face_detection import detect_face
from rppg import extract_rppg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LivenessNet(pretrained=False).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

cap = cv2.VideoCapture(0)

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

frame_buffer = deque(maxlen=180)
spoof_prob_buffer = deque(maxlen=15)


def draw_signal(signal, width=300, height=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if signal is None or len(signal) < 2:
        return canvas

    sig = np.array(signal, dtype=np.float32)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-6)
    sig = (sig * (height - 20)).astype(np.int32)

    step = width / max(len(sig), 1)

    for i in range(1, len(sig)):
        x1 = int((i - 1) * step)
        x2 = int(i * step)
        y1 = height - sig[i - 1]
        y2 = height - sig[i]
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return canvas


def detect_spoof_texture(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())

    if variance > 260:
        return "SCREEN/PRINT ARTIFACT", variance
    return "NATURAL SKIN", variance


def estimate_depth(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    depth = cv2.GaussianBlur(gray, (5, 5), 0)
    depth = cv2.equalizeHist(depth)

    h, w = depth.shape
    nose = depth[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    left_cheek = depth[h // 2 - 10:h // 2 + 10, w // 4 - 10:w // 4 + 10]
    right_cheek = depth[h // 2 - 10:h // 2 + 10, 3 * w // 4 - 10:3 * w // 4 + 10]

    nose_depth = float(np.mean(nose))
    cheek_depth = float((np.mean(left_cheek) + np.mean(right_cheek)) / 2.0)
    nose_protrusion = abs(nose_depth - cheek_depth)

    depth_center = depth[h // 4:3 * h // 4, w // 4:3 * w // 4]
    depth_variance = float(np.var(depth_center))

    if depth_variance < 40 and nose_protrusion < 10:
        depth_label = "FLAT SURFACE"
    else:
        depth_label = "3D FACE"

    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth_label, nose_protrusion, depth_color


def get_pulse_result(face):
    h, w, _ = face.shape
    cheek_roi = face[int(h * 0.4):int(h * 0.7), int(w * 0.3):int(w * 0.7)]

    signal_img = np.zeros((200, 300, 3), dtype=np.uint8)
    if cheek_roi.size == 0:
        return "INVALID ROI", None, None, signal_img

    frame_buffer.append(cheek_roi)

    if len(frame_buffer) < 45:
        return "COLLECTING...", None, None, signal_img

    signal = extract_rppg(list(frame_buffer))
    signal_img = draw_signal(signal)

    freq = np.fft.rfftfreq(len(signal), d=1 / 30.0)
    fft = np.abs(np.fft.rfft(signal))
    valid = (freq >= 0.75) & (freq <= 3.0)

    if not np.any(valid):
        return "INVALID SIGNAL", None, None, signal_img

    valid_freq = freq[valid]
    valid_fft = fft[valid]
    peak_idx = int(np.argmax(valid_fft))

    bpm = float(valid_freq[peak_idx] * 60.0)
    peak_strength = float(valid_fft[peak_idx])

    if 45 <= bpm <= 180 and peak_strength > 10:
        return "REAL PULSE", bpm, peak_strength, signal_img
    return "WEAK SIGNAL", bpm, peak_strength, signal_img


def final_decision(spoof_prob, texture_label, depth_label, pulse_label):
    spoof_votes = 0

    if spoof_prob >= 0.75:
        spoof_votes += 1
    if texture_label == "SCREEN/PRINT ARTIFACT" and spoof_prob >= 0.45:
        spoof_votes += 1
    if depth_label == "FLAT SURFACE" and spoof_prob >= 0.35:
        spoof_votes += 1
    if pulse_label == "WEAK SIGNAL" and spoof_prob >= 0.45:
        spoof_votes += 1

    if spoof_prob <= 0.45:
        return "LIVE", (0, 255, 0), max(0.65, 1.0 - spoof_prob)
    if spoof_prob >= 0.75 or spoof_votes >= 3:
        return "SPOOF", (0, 0, 255), max(0.85, spoof_prob)

    live_score = (
        0.75 * (1.0 - spoof_prob)
        + 0.20 * (1.0 if depth_label == "3D FACE" else 0.0)
        + 0.05 * (1.0 if pulse_label == "REAL PULSE" else 0.5)
    )
    label = "LIVE" if live_score >= 0.45 else "SPOOF"
    confidence = float(np.clip(abs(live_score - 0.5) * 2.0, 0.50, 0.99))
    color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
    return label, color, confidence


while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)
    dashboard = np.zeros((430, 920, 3), dtype=np.uint8)

    if face is not None:
        texture_label, texture_value = detect_spoof_texture(face)

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_input = infer_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, depth_map = model(face_input)

        spoof_prob = float(torch.sigmoid(pred).item())
        spoof_prob_buffer.append(spoof_prob)
        avg_spoof_prob = float(np.median(spoof_prob_buffer))

        depth_label, nose_protrusion, depth_color = estimate_depth(face)
        pulse_label, bpm, peak_strength, signal_img = get_pulse_result(face)
        final_label, final_color, final_confidence = final_decision(
            avg_spoof_prob, texture_label, depth_label, pulse_label
        )
        model_score = 1.0 - avg_spoof_prob

        dashboard[20:244, 20:244] = cv2.resize(face, (224, 224))
        dashboard[20:244, 300:524] = depth_color
        dashboard[20:220, 600:900] = signal_img

        cv2.putText(dashboard, "Face", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dashboard, "Depth Map", (300, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dashboard, "rPPG Signal", (600, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(
            dashboard,
            f"Classifier Score: {model_score:.3f}",
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            dashboard,
            f"Texture: {texture_label}  Var={texture_value:.1f}",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            dashboard,
            f"Depth: {depth_label}  NoseDelta={nose_protrusion:.1f}",
            (20, 355),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )

        if bpm is not None and peak_strength is not None:
            pulse_text = f"Pulse: {pulse_label}  BPM={bpm:.1f}  Peak={peak_strength:.1f}"
        else:
            pulse_text = f"Pulse: {pulse_label}"

        cv2.putText(
            dashboard,
            pulse_text,
            (20, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            dashboard,
            "Prediction display disabled",
            (650, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
        )
    else:
        cv2.putText(
            dashboard,
            "No face detected",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Liveness Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
