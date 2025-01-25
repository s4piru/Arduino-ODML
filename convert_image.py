import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

INPUT_ROOT = 'original_data'
OUTPUT_ROOT = 'data'
CLASSES = ['bottle', 'can']
MAX_SIZE = 512  # bigger side of the image will be resized to this size

def resize_image(image, max_size):
    """
    Resize the image so that the bigger side is equal to max_size.
    """
    original_size = image.size
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    print(f"Resized image from {original_size} to {image.size}")
    return image

def process_images():
    for cls in CLASSES:
        input_dir = os.path.join(INPUT_ROOT, cls)
        output_dir = os.path.join(OUTPUT_ROOT, cls)
        
        # If the output directory does not exist, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Traverse the input directory recursively
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.heic', '.heif')):
                    input_path = os.path.join(root, file)
                    
                    try:
                        # open the HEIC image
                        with Image.open(input_path) as img:
                            # resize the image
                            resized_img = resize_image(img, MAX_SIZE)
                            
                            # convert the image to JPEG
                            base_name = os.path.splitext(file)[0]
                            output_filename = base_name + '.jpg'
                            output_path = os.path.join(output_dir, output_filename)
                            
                            # save the image
                            resized_img.convert('RGB').save(output_path, 'JPEG', quality=85)
                            print(f"Converted and saved: {output_path}")
                    
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    process_images()
    print("HEIC to JPEG conversion completed.")
