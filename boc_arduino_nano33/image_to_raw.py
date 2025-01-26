from PIL import Image
import sys

def image_to_raw(image_path, output_path, width, height, array_name="rawImage"):
    """Resizes an image to the specified dimensions and outputs its RGB data as an array."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load the image: {e}")
        sys.exit(1)
    
    img = img.resize((width, height))
    pixels = list(img.getdata())
    
    c_array = f"const uint8_t {array_name}[{width * height * 3}] PROGMEM = {{\n"
    
    line = "  "
    for i, pixel in enumerate(pixels):
        r, g, b = pixel
        line += f"{r}, {g}, {b}, "
        if (i + 1) % 4 == 0:
            c_array += line + "\n"
            line = "  "
    if line.strip():
        c_array += line + "\n"
    
    c_array += "};\n"
    
    try:
        with open(output_path, 'w') as f:
            f.write(c_array)
        print(f"RAW data successfully written to '{output_path}'.")
    except Exception as e:
        print(f"Failed to write: {e}")
        sys.exit(1)

def print_usage():
    print("Run: python image_to_raw.py input_image output_file width height array_name")
    print("input_image : Path to the input image file (input.jpg)")
    print("output_file : Path to the output file (raw_image.h)")
    print("width       : Width after resizing (pixels, integer)")
    print("height      : Height after resizing (pixels, integer)")
    print("array_name  : Name of the output array (default is 'rawImage')")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Invalid arguments.\n")
        print_usage()
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_file = sys.argv[2]
    try:
        width = int(sys.argv[3])
        height = int(sys.argv[4])
    except ValueError:
        print("Width and height must be int.\n")
        print_usage()
        sys.exit(1)
    
    array_name = "rawImage"
    if len(sys.argv) >= 6:
        array_name = sys.argv[5]
    
    image_to_raw(input_image, output_file, width, height, array_name)
