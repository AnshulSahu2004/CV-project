import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

from liveliness_classification import LivenessNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# DATA TRANSFORMS
# -------------------------

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomGrayscale(p=0.1),

    transforms.GaussianBlur(kernel_size=3),

    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -------------------------
# LOAD DATASET
# -------------------------

dataset = datasets.ImageFolder("dataset")
from face_dataset import FaceCropDataset

from collections import defaultdict
import random
import os

video_groups = defaultdict(list)

for idx, (path, label) in enumerate(dataset.samples):

    filename = os.path.basename(path)

    # example: 1_1.avi_25_real.jpg
    video_id = filename.split(".avi")[0]

    video_groups[video_id].append(idx)

videos = list(video_groups.keys())
random.shuffle(videos)

split = int(0.8 * len(videos))

train_videos = videos[:split]
val_videos = videos[split:]

train_indices = []
val_indices = []

for v in train_videos:
    train_indices.extend(video_groups[v])

for v in val_videos:
    val_indices.extend(video_groups[v])

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)


print("Train videos:", len(train_videos))
print("Validation videos:", len(val_videos))
print("Train samples:", len(train_indices))
print("Validation samples:", len(val_indices))

# -------------------------
# HANDLE CLASS IMBALANCE
# -------------------------

targets = [dataset.targets[i] for i in train_dataset.indices]

class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dataset = FaceCropDataset(train_dataset, transform=train_transform)
val_dataset = FaceCropDataset(val_dataset, transform=test_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

model = LivenessNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
epochs = 20
best_acc = 0
print("Class mapping:", dataset.class_to_idx)


for epoch in range(epochs):

    model.train()

    train_losses = []

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.float().view(-1,1).to(device)

        pred, depth_map = model(images)
        cls_loss = criterion(pred, labels)
        depth_loss = torch.mean(depth_map**2)
        loss = cls_loss + 0.1 * depth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)


    model.eval()

    preds = []
    gt = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs, _ = model(images)
            outputs = torch.clamp(outputs,-10,10)

            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float().cpu().numpy().flatten()

            preds.extend(predictions)
            gt.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(gt, preds)

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", avg_train_loss)
    print("Validation Accuracy:", acc)


    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pth")
        print("Model saved")

print("\nTraining finished")
print("Best Accuracy:", best_acc)