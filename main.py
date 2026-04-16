import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.signal import butter, lfilter
from torchvision import transforms

from liveliness_classification import LivenessNet

frame_count=0

last_bbox=None
last_landmarks=None
last_face=None
last_depth_data= None
last_rppg= (None,None,"COLLECTING...", np.zeros((180,300,3),dtype=np.uint8))

##device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##models
print("[INFO] Loading LivenessNet...")
model = LivenessNet(pretrained=False).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("[INFO] Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.small_transform

print("[INFO] Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


##transform
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


##landmarks
LANDMARKS = {
    "nose_tip": 1,
    "forehead": 10,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


##buffers
prob_buffer = deque(maxlen=15)
fps_buffer = deque(maxlen=30)
rppg_frame_buffer = deque(maxlen=180)       # ~6 sec at ~30 fps
depth_feature_buffer = deque(maxlen=15)
face_center_buffer = deque(maxlen=20)
face_area_buffer = deque(maxlen=20)


##signals
def bandpass_filter(signal, low=0.75, high=3.0, fs=30):
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, signal)


def extract_rppg(frames, fs=30):
    rgb_means = []

    for frame in frames:
        # OpenCV frame => BGR
        b = np.mean(frame[:, :, 0])
        g = np.mean(frame[:, :, 1])
        r = np.mean(frame[:, :, 2])
        rgb_means.append([r, g, b])

    rgb_means = np.array(rgb_means, dtype=np.float32)

    mean_rgb = np.mean(rgb_means, axis=0) + 1e-6
    rgb_means = rgb_means / mean_rgb

    r = rgb_means[:, 0]
    g = rgb_means[:, 1]
    b = rgb_means[:, 2]

    # POS projection
    s1 = g - b
    s2 = g + b - 2 * r
    alpha = np.std(s1) / (np.std(s2) + 1e-6)
    pulse = s1 + alpha * s2

    pulse = bandpass_filter(pulse, fs=fs)
    pulse = pulse - np.mean(pulse)
    pulse = pulse / (np.std(pulse) + 1e-6)

    return pulse


##face
def detect_face_and_landmarks(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None, None, None

    face_landmarks = result.multi_face_landmarks[0]
    h, w, _ = frame_bgr.shape

    points = []
    named_points = {}

    for i, lm in enumerate(face_landmarks.landmark):
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        points.append((x, y))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # add padding
    pad_x = int(0.08 * (x2 - x1))
    pad_y = int(0.10 * (y2 - y1))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None

    # named landmarks in face crop coordinates
    for name, idx in LANDMARKS.items():
        lm = face_landmarks.landmark[idx]
        px = int(np.clip(lm.x * w, 0, w - 1))
        py = int(np.clip(lm.y * h, 0, h - 1))
        named_points[name] = (px - x1, py - y1)

    bbox = (x1, y1, x2, y2)
    return face, named_points, bbox


##classifier
def prepare_classifier_input(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = infer_transform(pil_img).unsqueeze(0).to(device)
    return tensor


##depth
def get_depth_map(face_bgr):
    input_batch = midas_transform(face_bgr).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    prediction = nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=face_bgr.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    depth_raw = prediction.detach().cpu().numpy()

    depth_norm = cv2.normalize(
        depth_raw, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_raw, depth_norm, depth_color


def sample_patch_mean(depth_map, x, y, patch=4):
    h, w = depth_map.shape
    x1 = max(0, x - patch)
    x2 = min(w, x + patch + 1)
    y1 = max(0, y - patch)
    y2 = min(h, y + patch + 1)

    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return float(np.mean(roi))


def extract_depth_geometry_features(depth_map, landmarks):
    values = {}
    for name, (x, y) in landmarks.items():
        val = sample_patch_mean(depth_map, x, y, patch=4)
        if val is None:
            return None
        values[name] = val

    nose = values["nose_tip"]
    forehead = values["forehead"]
    chin = values["chin"]
    left_cheek = values["left_cheek"]
    right_cheek = values["right_cheek"]
    left_eye = values["left_eye_outer"]
    right_eye = values["right_eye_outer"]
    left_mouth = values["left_mouth"]
    right_mouth = values["right_mouth"]

    cheek_mean = 0.5 * (left_cheek + right_cheek)
    eye_mean = 0.5 * (left_eye + right_eye)
    mouth_mean = 0.5 * (left_mouth + right_mouth)

    # NOTE:
    # MiDaS depth direction can vary visually depending on normalization.
    # So we use absolute prominence for robustness in heuristics.
    features = {
        "nose_vs_cheeks": nose - cheek_mean,
        "nose_vs_forehead": nose - forehead,
        "nose_vs_chin": nose - chin,
        "nose_vs_eyes": nose - eye_mean,
        "nose_vs_mouth": nose - mouth_mean,
        "forehead_vs_chin": forehead - chin,
        "eye_vs_mouth": eye_mean - mouth_mean,
        "cheek_symmetry": abs(left_cheek - right_cheek),
        "eye_symmetry": abs(left_eye - right_eye),
        "mouth_symmetry": abs(left_mouth - right_mouth),
        "anchor_std": float(np.std([
            nose, forehead, chin, left_cheek, right_cheek,
            left_eye, right_eye, left_mouth, right_mouth
        ]))
    }

    return features


def smooth_depth_features(features):
    depth_feature_buffer.append(features)
    out = {}
    for k in features.keys():
        out[k] = float(np.mean([f[k] for f in depth_feature_buffer]))
    return out


class TemporalDepthAnalyzer:
    def __init__(self, maxlen=20):
        self.history = deque(maxlen=maxlen)

    def update(self, feature_dict, face_bbox_area=0.0):
        item = dict(feature_dict)
        item["face_bbox_area"] = float(face_bbox_area)
        self.history.append(item)

    def ready(self, min_frames=8):
        return len(self.history) >= min_frames

    def _series(self, key):
        return np.array([x[key] for x in self.history], dtype=np.float32)

    def compute_temporal_features(self):
        if not self.ready():
            return None

        feats = {}
        keys = [
            "nose_vs_cheeks",
            "nose_vs_forehead",
            "nose_vs_chin",
            "nose_vs_eyes",
            "nose_vs_mouth",
            "cheek_symmetry",
            "eye_symmetry",
            "anchor_std",
            "face_bbox_area",
        ]

        for k in keys:
            s = self._series(k)
            feats[f"{k}_mean"] = float(np.mean(s))
            feats[f"{k}_std"] = float(np.std(s))
            feats[f"{k}_range"] = float(np.max(s) - np.min(s))

            if len(s) >= 2:
                diff = np.diff(s)
                feats[f"{k}_motion_mean"] = float(np.mean(np.abs(diff)))
            else:
                feats[f"{k}_motion_mean"] = 0.0

        return feats


def classify_depth_geometry(features):
    score = 0

    if abs(features["nose_vs_cheeks"]) > 15:
        score += 2
    else:
        score -= 2

    if abs(features["nose_vs_eyes"]) > 5:
        score += 1
    else:
        score -= 1

    if features["anchor_std"] > 12:
        score += 1
    else:
        score -= 1

    if features["cheek_symmetry"] < 12:
        score += 1
    else:
        score -= 1

    if features["eye_symmetry"] < 12:
        score += 1
    else:
        score -= 1

    if score >= 2:
        return "3D FACE", score
    return "LOW DEPTH GEOMETRY", score



def classify_temporal_depth(features):
    score = 0
    
    if abs(features["nose_vs_cheeks_mean"]) > 8:
        score += 2
    else:
        score -= 2

    if abs(features["nose_vs_eyes_mean"]) > 5:
        score += 1
    else:
        score -= 1

    if features["anchor_std_mean"] > 6:
        score += 1
    else:
        score -= 1

    if features["cheek_symmetry_mean"] < 12:
        score += 1
    else:
        score -= 1

    if features["eye_symmetry_mean"] < 12:
        score += 1
    else:
        score -= 1

    if features["nose_vs_cheeks_std"] < 10:
        score += 1
    else:
        score -= 1

    if features["nose_vs_eyes_std"] < 10:
        score += 1
    else:
        score -= 1

    if score >= 3:
        return "TEMPORAL 3D", score
    return "TEMPORAL FLAT", score


# =========================================================
# TEXTURE
# =========================================================
def detect_screen_attack(face):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)

    edge_density = float(np.sum(edges > 0) / edges.size)

    return edge_density > 0.18, edge_density

def detect_spoof_texture(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())

    if variance > 260:
        return "SCREEN/PRINT ARTIFACT", variance
    return "NATURAL SKIN", variance


# =========================================================
# RPPG
# =========================================================

def draw_signal(signal, width=300, height=180):
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


def get_rppg_result(face_bgr, fps):
    h, w, _ = face_bgr.shape

    cheek_roi = face_bgr[int(h * 0.40):int(h * 0.72), int(w * 0.28):int(w * 0.72)]
    if cheek_roi.size == 0:
        return None, None, "INVALID ROI", np.zeros((180, 300, 3), dtype=np.uint8)

    rppg_frame_buffer.append(cheek_roi)

    signal_img = np.zeros((180, 300, 3), dtype=np.uint8)

    min_needed = max(60, int(fps * 2.5))
    if len(rppg_frame_buffer) < min_needed:
        return None, None, "COLLECTING...", signal_img

    signal = extract_rppg(list(rppg_frame_buffer), fs=max(fps, 1))
    signal_img = draw_signal(signal)

    freq = np.fft.rfftfreq(len(signal), d=1.0 / max(fps, 1e-6))
    fft = np.abs(np.fft.rfft(signal))

    valid = (freq >= 0.75) & (freq <= 3.0)  # 45-180 BPM
    if not np.any(valid):
        return None, None, "INVALID SIGNAL", signal_img

    valid_freq = freq[valid]
    valid_fft = fft[valid]

    peak_idx = np.argmax(valid_fft)
    bpm = float(valid_freq[peak_idx] * 60.0)
    peak_strength = float(valid_fft[peak_idx])

    if 45 <= bpm <= 180 and peak_strength > 10:
        pulse_label = "REAL PULSE"
    else:
        pulse_label = "WEAK SIGNAL"

    return bpm, peak_strength, pulse_label, signal_img


# =========================================================
# MOTION / CHALLENGE RESPONSE
# =========================================================
def update_motion_buffers(bbox):
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    area = float((x2 - x1) * (y2 - y1))

    face_center_buffer.append((cx, cy))
    face_area_buffer.append(area)

    return cx, cy, area


def get_motion_status():
    if len(face_center_buffer) < 10:
        return "HOLD STILL / MOVE NATURALLY", 0.0

    xs = [p[0] for p in face_center_buffer]
    ys = [p[1] for p in face_center_buffer]
    areas = list(face_area_buffer)

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    area_range = max(areas) - min(areas) if len(areas) > 0 else 0.0

    motion_score = x_range + y_range + 0.002 * area_range

    if motion_score > 25:
        return "GOOD NATURAL MOTION", motion_score
    return "LOW MOTION", motion_score


# =========================================================
# VISUALIZATION
# =========================================================
def draw_named_landmarks(face_bgr, landmarks):
    out = face_bgr.copy()
    for name, (x, y) in landmarks.items():
        cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(out, name, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return out


# =========================================================
# FUSION
# =========================================================
def normalize(x,min_val,max_val):
    return np.clip((x-min_val)/(max_val-min_val+1e-6),0,1)
def combine_decision(live_prob, spoof_prob, texture_label, texture_val, depth_score,
                     temporal_score, pulse_label, motion_score, screen_attack=False):

    cnn_score = live_prob

    texture_score = 1 - normalize(texture_val, 120, 260)

    raw_depth_score = depth_score
    raw_temporal_score = temporal_score

    depth_score = normalize(raw_depth_score, -5, 5)

    temporal_score = normalize(raw_temporal_score, -5, 5)

    if pulse_label == "REAL PULSE":
        rppg_score = 1.0
    elif pulse_label == "COLLECTING...":
        rppg_score = 0.5
    else:
        rppg_score = 0.0

    motion_score = normalize(motion_score, 0, 50)

    final = (
        0.55 * cnn_score +
        0.12 * texture_score +
        0.12 * depth_score +
        0.08 * temporal_score +
        0.08 * rppg_score +
        0.05 * motion_score
    )

    spoof_votes = 0
    if spoof_prob >= 0.75:
        spoof_votes += 1
    if screen_attack and spoof_prob >= 0.45:
        spoof_votes += 1
    if texture_label == "SCREEN/PRINT ARTIFACT" and texture_val >= 320:
        spoof_votes += 1
    if raw_depth_score <= -4 and spoof_prob >= 0.35:
        spoof_votes += 1
    if raw_temporal_score <= -4 and spoof_prob >= 0.35:
        spoof_votes += 1
    if pulse_label == "WEAK SIGNAL" and spoof_prob >= 0.45:
        spoof_votes += 1

    live_votes = 0
    if spoof_prob <= 0.45:
        live_votes += 2
    if raw_depth_score >= 0:
        live_votes += 1
    if raw_temporal_score >= 0:
        live_votes += 1
    if pulse_label in ("REAL PULSE", "COLLECTING..."):
        live_votes += 1
    if texture_label == "NATURAL SKIN" and not screen_attack:
        live_votes += 1

    if spoof_prob <= 0.45:
        label = "LIVE"
        final = max(final, 0.62)
    elif spoof_prob >= 0.75:
        label = "SPOOF"
        final = min(final, 0.25)
    elif spoof_votes >= 3 and live_votes < 2:
        label = "SPOOF"
        final = min(final, 0.35)
    else:
        label = "LIVE" if final >= 0.45 else "SPOOF"

    if label == "LIVE":
        confidence = max(abs(final - 0.5) * 2.0, 1.0 - spoof_prob)
        color = (0, 255, 0)
    else:
        confidence = max(abs(final - 0.5) * 2.0, spoof_prob)
        color = (0, 0, 255)

    confidence = float(np.clip(confidence, 0.50, 0.99))
    if label == "LIVE" and screen_attack:
        status = "LIVE - EDGE CUE, IMPROVE LIGHTING"
    elif confidence >= 0.75:
        status = "HIGH CONFIDENCE"
    else:
        status = "LOW CONFIDENCE - HOLD STEADY"
    return label, float(final), color, confidence, status


# =========================================================
# MAIN
# =========================================================
def main():
    temporal_analyzer = TemporalDepthAnalyzer(maxlen=20)

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    print("[INFO] Starting webcam... Press ESC to exit.")

    global frame_count,last_depth_data,last_rppg,last_face,last_bbox,last_landmarks

    while True:
        frame_count+=1

        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        instant_fps = 1.0 / dt if dt > 0 else 0.0
        fps_buffer.append(instant_fps)
        fps = sum(fps_buffer) / len(fps_buffer)

        dashboard = np.zeros((720, 1280, 3), dtype=np.uint8)

        face, landmarks, bbox = detect_face_and_landmarks(frame)

        # Title
        cv2.putText(dashboard, "Advanced Multimodal Face Analysis System",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if face is None or landmarks is None or bbox is None:
          if last_face is None:
            cv2.putText(dashboard, "No face detected",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Dashboard", dashboard)
            if cv2.waitKey(1) & 0xFF == 27:
              break
            continue
          else:
           face = last_face
           bbox=last_bbox
           landmarks=last_landmarks
        else:
           last_face = face
           last_bbox=bbox
           last_landmarks=landmarks

        # resize face for stable downstream usage
        face = cv2.resize(face, (224, 224))
        screen_attack, screen_edge_density = detect_screen_attack(face)

        # because face is resized, scale landmarks to resized face
        x1, y1, x2, y2 = bbox
        orig_w = max(1, x2 - x1)
        orig_h = max(1, y2 - y1)

        scaled_landmarks = {}
        for name, (lx, ly) in landmarks.items():
            sx = int(np.clip(lx * 224 / orig_w, 0, 223))
            sy = int(np.clip(ly * 224 / orig_h, 0, 223))
            scaled_landmarks[name] = (sx, sy)

        face_with_points = draw_named_landmarks(face, scaled_landmarks)

        # -------------------------------------------------
        # 1. DEEP CLASSIFIER
        # -------------------------------------------------
        cls_input = prepare_classifier_input(face)
        with torch.no_grad():
            pred, _ = model(cls_input)

        spoof_prob = float(torch.sigmoid(pred).item())
        prob_buffer.append(spoof_prob)
        avg_spoof_prob = float(np.median(prob_buffer))
        live_prob = 1.0 - avg_spoof_prob
        model_score = 1.0 - avg_spoof_prob

        # -------------------------------------------------
        # 2. TEXTURE
        # -------------------------------------------------
        texture_label, texture_value = detect_spoof_texture(face)

        # -------------------------------------------------
        # 3. DEPTH + STATIC GEOMETRY
        # -------------------------------------------------
        if frame_count % 5== 0 or last_depth_data is None:
            last_depth_data=get_depth_map(face)
        depth_raw, depth_norm, depth_color = last_depth_data

        nose = depth_norm[100:120,100:120]
        cheek = depth_norm[100:120,60:80]

        nose_depth = np.mean(nose)
        cheek_depth = np.mean(cheek)
        flat_surface = abs(nose_depth - cheek_depth) < 5

        depth_features = extract_depth_geometry_features(depth_norm, scaled_landmarks)

        if depth_features is not None:
            depth_features = smooth_depth_features(depth_features)
            depth_label, depth_score = classify_depth_geometry(depth_features)
        else:
            depth_label, depth_score = "DEPTH ERROR", 0

        if flat_surface:
            depth_label = "FLAT SURFACE"
            depth_score = min(depth_score, -3)

        # -------------------------------------------------
        # 4. TEMPORAL DEPTH
        # -------------------------------------------------
        _, _, area = update_motion_buffers(bbox)
        if depth_features is not None:
            temporal_analyzer.update(depth_features, face_bbox_area=area)

        if temporal_analyzer.ready(min_frames=8):
            temporal_features = temporal_analyzer.compute_temporal_features()
            temporal_label, temporal_score = classify_temporal_depth(temporal_features)
        else:
            temporal_label, temporal_score = "COLLECTING...", 0

        # -------------------------------------------------
        # 5. MOTION STATUS
        # -------------------------------------------------
        motion_status, motion_score = get_motion_status()

        # -------------------------------------------------
        # 6. RPPG
        # -------------------------------------------------
        if frame_count % 10 == 0:
            last_rppg=get_rppg_result(face,fps)
        bpm, peak_strength, pulse_label, signal_img = last_rppg

        # -------------------------------------------------
        # 7. FUSION
        # -------------------------------------------------
        final_label, final_score, final_color, final_confidence, final_status = combine_decision(
            live_prob,
            avg_spoof_prob,
            texture_label,
            texture_value,
            depth_score,
            temporal_score,
            pulse_label,
            motion_score,
            screen_attack
        )

        # -------------------------------------------------
        # DRAW PANELS
        # -------------------------------------------------
        dashboard[60:284, 20:244] = face_with_points
        dashboard[60:284, 280:504] = depth_color
        dashboard[60:240, 540:840] = signal_img

        # live webcam preview with bbox
        preview = frame.copy()
        fx1, fy1, fx2, fy2 = bbox
        cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (180, 180, 180), 2)
        preview = cv2.resize(preview, (380, 285))
        dashboard[60:345, 870:1250] = preview

        # section titles
        cv2.putText(dashboard, "Face + Key Landmarks", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dashboard, "Depth Map", (280, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dashboard, "rPPG Signal", (540, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dashboard, "Camera Preview", (870, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # metrics block
        y0 = 390
        line_gap = 32

        cv2.putText(dashboard, f"Classifier Score: {model_score:.3f}",
                    (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        screen_text = f"ScreenEdge={screen_edge_density:.3f}"
        cv2.putText(dashboard, f"Texture: {texture_label}   Value={texture_value:.2f}   {screen_text}",
                    (20, y0 + line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(dashboard, f"Static Depth: {depth_label}   Score={depth_score}",
                    (20, y0 + 2 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(dashboard, f"Temporal Depth: {temporal_label}   Score={temporal_score}",
                    (20, y0 + 3 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if bpm is not None and peak_strength is not None:
            cv2.putText(dashboard, f"Pulse: {pulse_label}   BPM={bpm:.1f}   Peak={peak_strength:.2f}",
                        (20, y0 + 4 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(dashboard, f"Pulse: {pulse_label}",
                        (20, y0 + 4 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(dashboard, f"Motion Cue: {motion_status}   Value={motion_score:.2f}",
                    (20, y0 + 5 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.putText(dashboard, f"FPS: {fps:.1f}",
                    (20, y0 + 6 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # challenge prompt
        prompt = "Prompt: Slightly turn your head or move naturally for stronger temporal verification"
        cv2.putText(dashboard, prompt,
                    (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

        # internal fusion still runs, but the visible verdict is hidden.
        cv2.rectangle(dashboard, (870, 390), (1240, 660), (180, 180, 180), 3)
        cv2.putText(dashboard, "ANALYSIS PANEL",
                    (930, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        cv2.putText(dashboard, "Prediction display disabled",
                    (900, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        cv2.putText(dashboard, "Cues are shown for review",
                    (910, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

        # short academic note on dashboard
        cv2.putText(dashboard,
                    "Fusion cues: CNN classifier + monocular depth + facial geometry + temporal consistency + rPPG + texture",
                    (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Professor-Level Liveness Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[FATAL ERROR]")
        print(e)
        traceback.print_exc()
        input("\nPress Enter to exit...")
