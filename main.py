from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import os


if __name__ == '__main__':
    print()
    # model = load_vgg_model(Config.VGG_MODEL_PATH)
    #
    # # model['input'].assign(image)
    # # sess.run(model['conv4_2'])  # activate layer
    # # tf.reset_default_graph()
    # sess = tf.compat.v1.Session()
    #
    # content_image = np.asarray(Image.open('./input/01_content_beth_harmon.jpg'))
    # style_image = np.asarray(Image.open('./input/01_style_chess_fantasy.jpg'))
    #
    # content_image = reshape_and_normalize_image(content_image)
    # style_image = reshape_and_normalize_image(style_image)
    # generated_image = generate_noise_image(content_image, noise_ratio=0.7)