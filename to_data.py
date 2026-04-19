from PIL import Image
import struct
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else 'out.png'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'out.data'

img = Image.open(input_file).convert('RGBA')

with open(output_file, 'wb') as f:
    f.write(struct.pack('ii', *img.size))
    f.write(img.tobytes())

print(f"Сохранено: {img.size[0]}x{img.size[1]} пикселей в {output_file}")