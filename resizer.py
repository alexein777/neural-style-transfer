from PIL import Image
from utils import *
import matplotlib.pyplot as plt
import os
import numpy as np


NEW_SIZE = (600, 400)
INPUT_DIR = 'input'
OUTPUT_DIR = 'output/combo'
CONFIG_LIST = [
    {
        'content': os.path.join(INPUT_DIR, 'content', 'bilbo.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'sunny-forrest.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'eiffel.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'monet-twilight-venice.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'hourglass.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'sass.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'jesus-disciples.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'sea-fall.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'knightfall.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'monet-soleil-levant.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'la-casa-de-papel.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'abstract-painting.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'skyrim.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'autumn-valley.jpg'),
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'selfie-rooftop.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'abstract-colorful.jpg'),
    }
]

# Resize content and style images first and save them to OUTPUT_DIR
for config in CONFIG_LIST:
    content_image = Image.open(config['content'])
    style_image = Image.open(config['style'])

    content_resized = resize_image(content_image, NEW_SIZE)
    content_new_name = get_image_name(config['content']) + "=" + str(NEW_SIZE)
    content_out_path = os.path.join(OUTPUT_DIR, content_new_name)
    content_resized.save(content_out_path + '.jpg')

    style_resized = resize_image(style_image, NEW_SIZE)
    style_new_name = get_image_name(config['style']) + "=" + str(NEW_SIZE)
    style_out_path = os.path.join(OUTPUT_DIR, style_new_name)
    style_resized.save(style_out_path + '.jpg')

# Resize generated images and save them to output path
for img_name in os.listdir('output/original'):
    img_path = os.path.join('output/original', img_name)
    image = Image.open(img_path)
    image_resized = resize_image(image, NEW_SIZE)
    image_new_name = get_image_name(img_name) + '=' + str(NEW_SIZE)
    image_new_path = os.path.join(OUTPUT_DIR, image_new_name)
    image_resized.save(image_new_path + '.jpg')

print('Images resized successfully.')
# img = Image.open('output/original/call-of-duty-mw2+monet-grenouillere_a=100000.0,b=4000.0,iter=1000.jpg')
# img = resize_image(img, fx=1.5, fy=1.5)
# plt.imshow(img)
# plt.show()
