from PIL import Image
import sys

def image_to_raw(image_path, output_path, width, height, array_name="rawImage"):
    """
    画像を指定されたサイズにリサイズし、RGBデータをC言語の配列形式で出力します。

    :param image_path: 入力画像ファイルのパス
    :param output_path: 出力ファイルのパス（.hファイルなど）
    :param width: リサイズ後の幅（ピクセル）
    :param height: リサイズ後の高さ（ピクセル）
    :param array_name: 出力配列の名前（デフォルトは "rawImage"）
    """
    try:
        # 画像を開く
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"画像の読み込みに失敗しました: {e}")
        sys.exit(1)
    
    # 画像をリサイズ
    img = img.resize((width, height))
    pixels = list(img.getdata())
    
    # C配列の開始部分
    c_array = f"const uint8_t {array_name}[{width * height * 3}] PROGMEM = {{\n"
    
    # 各ピクセルのRGB値を配列に追加
    line = "  "
    for i, pixel in enumerate(pixels):
        r, g, b = pixel
        line += f"{r}, {g}, {b}, "
        # 1行に12値（4ピクセル分）ずつ出力
        if (i + 1) % 4 == 0:
            c_array += line + "\n"
            line = "  "
    # 残りのデータを追加
    if line.strip():
        c_array += line + "\n"
    
    # 配列の終了部分
    c_array += "};\n"
    
    # 出力ファイルに書き込み
    try:
        with open(output_path, 'w') as f:
            f.write(c_array)
        print(f"RAWデータが正常に '{output_path}' に出力されました。")
    except Exception as e:
        print(f"出力ファイルへの書き込みに失敗しました: {e}")
        sys.exit(1)

def print_usage():
    print("Usage: python image_to_raw.py <input_image> <output_file> <width> <height> [array_name]")
    print("  <input_image>  : 入力画像ファイルのパス（例: input.jpg）")
    print("  <output_file>  : 出力ファイルのパス（例: raw_image.h）")
    print("  <width>        : リサイズ後の幅（ピクセル、整数）")
    print("  <height>       : リサイズ後の高さ（ピクセル、整数）")
    print("  [array_name]   : 出力配列の名前（オプション、デフォルトは 'rawImage'）")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("引数が不足しています。\n")
        print_usage()
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_file = sys.argv[2]
    try:
        width = int(sys.argv[3])
        height = int(sys.argv[4])
    except ValueError:
        print("幅と高さは整数値で指定してください。\n")
        print_usage()
        sys.exit(1)
    
    array_name = "rawImage"  # デフォルト値
    if len(sys.argv) >= 6:
        array_name = sys.argv[5]
    
    image_to_raw(input_image, output_file, width, height, array_name)
