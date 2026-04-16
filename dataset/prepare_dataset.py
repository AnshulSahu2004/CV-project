import os
import shutil

source = "dataset/CASIA-FASD/train_img/train_img/color"

live_folder = "dataset/live"
spoof_folder = "dataset/spoof"

os.makedirs(live_folder, exist_ok=True)
os.makedirs(spoof_folder, exist_ok=True)

for img in os.listdir(source):

    if not img.endswith(".jpg"):
        continue

    src = os.path.join(source, img)

    if "real" in img:
        dst = os.path.join(live_folder, img)

    elif "fake" in img:
        dst = os.path.join(spoof_folder, img)

    else:
        continue

    shutil.copy(src, dst)

print("Dataset prepared successfully")