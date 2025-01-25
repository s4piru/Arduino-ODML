# data_split.py
import os
import shutil
import random
from constants import DATA_DIR

def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_source):
            continue

        images = os.listdir(cls_source)
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create target directories
        train_cls_dir = os.path.join(train_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(train_cls_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(test_cls_dir, img))

    print("Completed data split.")

if __name__ == "__main__":
    source_dir = DATA_DIR
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    split_data(source_dir, train_dir, test_dir, split_ratio=0.8)
