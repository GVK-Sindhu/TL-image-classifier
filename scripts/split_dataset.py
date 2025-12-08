import os
import random
import shutil

random.seed(42)

RAW_DIR = 'data/raw'
OUT_DIR = 'data'
TRAIN = 0.7
VAL = 0.15
TEST = 0.15


for split in ['train', 'val', 'test']:
    split_dir = os.path.join(OUT_DIR, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

# Get class names
classes = os.listdir(RAW_DIR)
print("Classes:", classes)

for cls in classes:
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN)
    n_val = int(n * VAL)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        out_dir = os.path.join(OUT_DIR, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(cls_path, f), os.path.join(out_dir, f))

    print(f"{cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Splitting completed.")
