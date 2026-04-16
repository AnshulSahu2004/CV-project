import cv2
import torch
from torch.utils.data import Dataset
import mediapipe as mp
from PIL import Image
import numpy as np

mp_face = mp.solutions.face_detection


class FaceCropDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img, label = self.dataset[idx]

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.detections:

            bbox = results.detections[0].location_data.relative_bounding_box

            h, w, _ = img.shape

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            face = img[y1:y2, x1:x2]

            if face.size != 0:
                img = face

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label