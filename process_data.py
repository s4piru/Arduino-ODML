import os
import shutil
import random
from PIL import Image
import pillow_heif
from constants import DATA_DIR, ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR, SPLIT_RATIO, CLASSES, MAX_SIZE

# Setup to open HEIC files
pillow_heif.register_heif_opener()

def is_image_file(filename):
    return filename.lower().endswith(('.heic', '.heif', '.png', '.jpg', '.jpeg'))

def resize_image(image, max_size):
    original_size = image.size
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    print(f"Resized image from {original_size} to {image.size}.")
    return image

def process_images():
    """Convert images from HEIC to JPEG, resize them, and save to a temporary directory."""
    for cls in CLASSES:
        input_dir = os.path.join(ORIGINAL_DATA_DIR, cls)
        processed_dir = os.path.join(PROCESSED_DATA_DIR, cls)
        
        # Create the output directory if it does not exist
        os.makedirs(processed_dir, exist_ok=True)
        print(f"Created or confirmed processed directory: {processed_dir}")
        
        # Recursively traverse the input directory
        for root, _, files in os.walk(input_dir):
            for file in files:
                if is_image_file(file):
                    input_path = os.path.join(root, file)
                    
                    try:
                        # Open HEIC image (supports other formats as well)
                        with Image.open(input_path) as img:
                            # Resize the image
                            resized_img = resize_image(img, MAX_SIZE)
                            
                            # Convert to JPEG format
                            base_name = os.path.splitext(file)[0]
                            output_filename = base_name + '.jpg'
                            output_path = os.path.join(processed_dir, output_filename)
                            
                            # Save the image
                            resized_img.convert('RGB').save(output_path, 'JPEG', quality=85)
                            print(f"Converted and saved: {output_path}")
                    
                    except Exception as e:
                        print(f"Error occurred while processing {input_path}: {e}")

def split_data(train_dir, test_dir, split_ratio=0.8):
    """Split processed images into training and testing sets."""
    for cls in CLASSES:
        processed_cls_dir = os.path.join(PROCESSED_DATA_DIR, cls)
        if not os.path.isdir(processed_cls_dir):
            print(f"Warning: Directory for class '{cls}' does not exist. Skipping.")
            continue

        # List image files
        image_files = [f for f in os.listdir(processed_cls_dir) if is_image_file(f)]
        if not image_files:
            print(f"Warning: No image files found for class '{cls}'.")
            continue

        # Shuffle images
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split_ratio)
        train_images = image_files[:split_idx]
        test_images = image_files[split_idx:]

        # Create target directories
        train_cls_dir = os.path.join(train_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        # Copy images
        for img in train_images:
            src_path = os.path.join(processed_cls_dir, img)
            dst_path = os.path.join(train_cls_dir, img)
            shutil.copy(src_path, dst_path)
        for img in test_images:
            src_path = os.path.join(processed_cls_dir, img)
            dst_path = os.path.join(test_cls_dir, img)
            shutil.copy(src_path, dst_path)

        print(f"Split data for class '{cls}'. Training: {len(train_images)}, Testing: {len(test_images)}")

def clean_directories():
    """Remove existing processed and output directories."""
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
        print(f"Removed existing processed directory: {PROCESSED_DATA_DIR}")
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Removed existing output directory: {DATA_DIR}")

if __name__ == "__main__":
    random.seed(42)
    clean_directories()
    
    print("Starting image processing...")
    process_images()
    print("Image processing completed.\n")
    
    print("Starting data split...")
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')
    split_data(train_dir, test_dir, split_ratio=SPLIT_RATIO)
    print("Data split completed.")
