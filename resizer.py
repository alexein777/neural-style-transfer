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
        'content': os.path.join(INPUT_DIR, 'content', 'beth-harmon.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'scream.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'beatles.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'colorful.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'call-of-duty-mw2.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'monet-grenouillere.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'captain-jack-sparrow.jpeg'),
        'style': os.path.join(INPUT_DIR, 'style', 'kosovo-maiden.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'dragon.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'storm.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'golden-gate.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'wave.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'jon-and-daenerys.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'monet.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'lion.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'mona-lisa.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'mountain-landscape.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'van-gogh-starry-night.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'neo-trinity.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'gernika.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'optimus-prime.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'frostfire.jpg')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'the-return-of-the-king.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'wheat-field.png')
    },
    {
        'content': os.path.join(INPUT_DIR, 'content', 'witcher.jpg'),
        'style': os.path.join(INPUT_DIR, 'style', 'winter-bridge.jpg')
    },
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

# img = Image.open('output/original/call-of-duty-mw2+monet-grenouillere_a=100000.0,b=4000.0,iter=1000.jpg')
# img = resize_image(img, fx=1.5, fy=1.5)
# plt.imshow(img)
# plt.show()
