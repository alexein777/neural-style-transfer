import scipy.io
from PIL import Image
import tensorflow as tf
import numpy as np
import os


class Config:
    IMAGE_HEIGHT = 300  # Height of VGG-19 input (conv1_1 layer)
    IMAGE_WIDTH = 400  # Width of VGG-19 input (conv1_1 layer)
    N_CHANNELS = 3  # Number of channels (RGB)
    NOISE_RATIO = 0.6  # Noise ratio used for generated image
    IMAGE_INPUT_SHAPE = (1, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)  # Input imaged shape for VGG

    # Image means from ImageNet dataset, used for normalization. More info here:
    # https://github.com/tensorflow/models/issues/517
    IMAGENET_MEANS = np.array([123.68, 116.78, 103.94]).reshape((1, 1, 1, 3))

    # Pretrained model path (note: can't be pushed to git repo due to size) and output path
    VGG_MODEL_PATH = os.path.join('.', 'models', 'imagenet-vgg-19.mat')
    OUTPUT_DIR = './output'


def load_vgg_model(path):
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    # Helper functions that create tf tensors for the model
    def _weights(layer, expected_layer_name):
        """
        Return kernel params and bias from VGG for a specified layer.
        Expected shapes (for a given layer) are:
            kernel: (f, f, n_c, n_f)
                * f: filter size (f x f)
                * n_c: number of channels
                * n_f: number of filters
            b (bias): (n_f, 1), where n_f is same as above (1 real value for each filter)
        """

        # Ensure that correct layer is extracted
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name

        # Extract weights and bias vector from corresponding layer
        kb = vgg_layers[0][layer][0][0][2]
        kernel = kb[0][0]
        bias = kb[0][1]

        return kernel, bias

    def _relu(conv2d_layer):
        """Return RELU activation of a tensor conv2d input."""
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """Return conv2d layer using the weights and biases from VGG model at given layer."""
        kernel, bias = _weights(layer, layer_name)
        kernel = tf.constant(kernel)
        bias = tf.constant(np.reshape(bias, bias.size))

        return tf.nn.conv2d(prev_layer, kernel, strides=[1, 1, 1, 1], padding='SAME') + bias

    def _avg_pool(prev_layer):
        """Return the Average pooling layer for a given previous layer (tensor)."""
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _max_pool(prev_layer):
        """Return the Average pooling layer for a given previous layer (tensor)."""
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Construct the model graph, following VGG-19 architecture.
    # Note: Gatys et al. proposes Average pooling instead of Max pooling.
    graph = {}
    graph['input'] = tf.Variable(np.zeros(shape=Config.IMAGE_INPUT_SHAPE), dtype='float32')
    graph['conv1_1'] = _relu(_conv2d(graph['input'], 0, 'conv1_1'))
    graph['conv1_2'] = _relu(_conv2d(graph['conv1_1'], 2, 'conv1_2'))
    graph['avgpool1'] = _avg_pool(graph['conv1_2'])
    graph['conv2_1'] = _relu(_conv2d(graph['avgpool1'], 5, 'conv2_1'))
    graph['conv2_2'] = _relu(_conv2d(graph['conv2_1'], 7, 'conv2_2'))
    graph['avgpool2'] = _avg_pool(graph['conv2_2'])
    graph['conv3_1'] = _relu(_conv2d(graph['avgpool2'], 10, 'conv3_1'))
    graph['conv3_2'] = _relu(_conv2d(graph['conv3_1'], 12, 'conv3_2'))
    graph['conv3_3'] = _relu(_conv2d(graph['conv3_2'], 14, 'conv3_3'))
    graph['conv3_4'] = _relu(_conv2d(graph['conv3_3'], 16, 'conv3_4'))
    graph['avgpool3'] = _avg_pool(graph['conv3_4'])
    graph['conv4_1'] = _relu(_conv2d(graph['avgpool3'], 19, 'conv4_1'))
    graph['conv4_2'] = _relu(_conv2d(graph['conv4_1'], 21, 'conv4_2'))
    graph['conv4_3'] = _relu(_conv2d(graph['conv4_2'], 23, 'conv4_3'))
    graph['conv4_4'] = _relu(_conv2d(graph['conv4_3'], 25, 'conv4_4'))
    graph['avgpool4'] = _avg_pool(graph['conv4_4'])
    graph['conv5_1'] = _relu(_conv2d(graph['avgpool4'], 28, 'conv5_1'))
    graph['conv5_2'] = _relu(_conv2d(graph['conv5_1'], 30, 'conv5_2'))
    graph['conv5_3'] = _relu(_conv2d(graph['conv5_2'], 32, 'conv5_3'))
    graph['conv5_4'] = _relu(_conv2d(graph['conv5_3'], 34, 'conv5_4'))
    graph['avgpool5'] = _avg_pool(graph['conv5_4'])

    return graph


def generate_noise_image(content_image, noise_ratio=Config.NOISE_RATIO):
    """Generate image from content image by adding noise to it."""
    noise_image = np.random.uniform(-20, 20, Config.IMAGE_INPUT_SHAPE).astype('float32')

    # Gerenate image as a weighted average of noise image and content image
    return noise_ratio * noise_image + (1 - noise_ratio) * content_image


def reshape_and_normalize_image(image):
    """Reshape and normalize input image to prepare it for feeding into VGG."""
    reshaped_image = np.reshape(image, ((1,) + image.shape))

    # Subtract means to match the expected input of VGG (which has an ImageNet mean shift)
    return reshaped_image - Config.IMAGENET_MEANS


def save_image(image, path):
    """Save numpy image to a given location."""
    # Unnormalize image before saving for better quality
    image = image + Config.IMAGENET_MEANS

    # Remove unnecessary first dimension and clamp remaining values to [0, 255]
    image = np.clip(image[0], 0, 255).astype('uint8')
    Image.fromarray(image).save(path)
