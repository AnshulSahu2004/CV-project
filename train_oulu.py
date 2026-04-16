import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

from sklearn.metrics import confusion_matrix

def compute_metrics(gt, preds):

    cm = confusion_matrix(gt, preds, labels=[0,1])

    # cm = [[TN, FP],
    #       [FN, TP]]

    tn, fp, fn, tp = cm.ravel()

    apcer = fp / (tn + fp + 1e-8)
    bpcer = fn / (fn + tp + 1e-8)
    acer = 0.5 * (apcer + bpcer)
    acc = (tp + tn) / (tn + fp + fn + tp + 1e-8)

    return acc, apcer, bpcer, acer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05
    ),

    transforms.GaussianBlur(3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -------------------------
# LOAD DATA
# -------------------------
full_dataset = datasets.ImageFolder("C:/Users/hp/Desktop/CV project/Oulu-NPU")

indices = list(range(len(full_dataset)))
np.random.shuffle(indices)

train_size = int(0.8 * len(indices))
val_size = int(0.1 * len(indices))

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

# separate datasets with different transforms
train_data = torch.utils.data.Subset(
    datasets.ImageFolder("C:/Users/hp/Desktop/CV project/Oulu-NPU", transform=train_transform),
    train_idx
)

val_data = torch.utils.data.Subset(
    datasets.ImageFolder("C:/Users/hp/Desktop/CV project/Oulu-NPU", transform=test_transform),
    val_idx
)

test_data = torch.utils.data.Subset(
    datasets.ImageFolder("C:/Users/hp/Desktop/CV project/Oulu-NPU", transform=test_transform),
    test_idx
)

print("Dataset loaded:", len(full_dataset))
print("Class mapping:", full_dataset.class_to_idx)

# -------------------------
# HANDLE CLASS IMBALANCE
# -------------------------
targets = [full_dataset.targets[i] for i in train_data.indices]
class_counts = np.bincount(targets)

class_weights = 1. / class_counts
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# -------------------------
# DATALOADER
# -------------------------
train_loader = DataLoader(train_data, batch_size=32, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
# -------------------------
# MODEL
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = nn.Sequential(
    nn.Linear(512,128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128,1)
)

model = model.to(device)

# -------------------------
# TRAIN SETUP
# -------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 20
best_acc = 0

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(epochs):

    model.train()
    train_loss = []

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().view(-1,1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    preds, gt = [], []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            predictions = (probs > 0.5).cpu().numpy().flatten()

            preds.extend(predictions)
            gt.extend(labels.numpy())

    acc, apcer, bpcer, acer = compute_metrics(gt, preds)

    print(f"Val ACC: {acc:.4f}")
    print(f"APCER: {apcer:.4f}")
    print(f"BPCER: {bpcer:.4f}")
    print(f"ACER: {acer:.4f}")

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", np.mean(train_loss))
    print("Val Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Model Saved!")

# -------------------------
# FINAL TEST
# -------------------------
model.load_state_dict(torch.load("best_model.pth"))

model.eval()
preds, gt = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        predictions = (probs > 0.5).cpu().numpy().flatten()

        preds.extend(predictions)
        gt.extend(labels.numpy())

acc, apcer, bpcer, acer = compute_metrics(gt, preds)

print("\nFINAL TEST RESULTS")
print(f"ACC: {acc:.4f}")
print(f"APCER: {apcer:.4f}")
print(f"BPCER: {bpcer:.4f}")
print(f"ACER: {acer:.4f}")