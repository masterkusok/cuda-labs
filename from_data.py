from PIL import Image
import struct
import ctypes
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else 'out.data'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'out.png'

with open(input_file, 'rb') as fin:
    width, height = struct.unpack('ii', fin.read(8))
    
    buffer = ctypes.create_string_buffer(4 * width * height)
    
    fin.readinto(buffer)

image = Image.new('RGBA', (width, height))
pixels = image.load()

offset = 0
for y in range(height):
    for x in range(width):
        r, g, b, a = struct.unpack_from('cccc', buffer, offset)
        pixels[x, y] = (ord(r), ord(g), ord(b), ord(a))
        offset += 4

image.save(output_file)