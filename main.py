from utils import *
from neural_style_transfer import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import os


if __name__ == '__main__':
    # model['input'].assign(image)
    # sess.run(model['conv4_2'])  # activate layer
    # tf.reset_default_graph()

    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()

    content_im = Image.open('./input/01_content_beth_harmon.jpg')
    content_im_resized = content_im.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.BILINEAR)
    content_image = np.asarray(content_im_resized)

    style_im = Image.open('./input/01_style_chess_fantasy.jpg')
    style_im_resized = style_im.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.BILINEAR)
    style_image = np.asarray(style_im_resized)

    content_image = reshape_and_normalize_image(content_image)
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image, noise_ratio=0.5)

    # Load VGG
    model = load_vgg_model(Config.VGG_MODEL_PATH)

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_content = sess.run(out)
    a_generated = out
    content_cost = calculate_content_cost(a_content, a_generated)
    #
    sess.run(model['input'].assign(style_image))
    style_cost = calculate_style_cost(sess, model, STYLE_LAYER_WEIGHTS)
    total_cost = calculate_total_cost(content_cost, style_cost)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(total_cost)

    def _model_nn(sess, input_image, num_iter=500):
        sess.run(tf.compat.v1.global_variables_initializer())

        sess.run(model['input'].assign(input_image))

        for i in range(num_iter):
            # Minimize total cost
            sess.run(train_step)

            # Generate image again, as params have changed
            generated_image = sess.run(model['input'])

            if i % 30 == 0:
                J_total, J_content, J_style = sess.run([total_cost, content_cost, style_cost])
                print(f'\nIteration: {i}')
                print('------------------------------------')
                print(f'\t* Total cost: {J_total}')
                print(f'\t* Content cost: {J_content}')
                print(f'\t* Style cost: {J_style}')

        save_image(generated_image, './output/generated_image.jpg')

        return generated_image

    _model_nn(sess, generated_image)
