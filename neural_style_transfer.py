from utils import *
import numpy as np
import tensorflow as tf


STYLE_LAYER_WEIGHTS = {
    'conv1_1': 0.2,
    'conv2_1': 0.2,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}


def content_cost(a_content, a_generated):
    """Calculate content cost from activations of a chosen layer."""

    # Extract dimensions from activation tensor (ignore mini-batch dimension)
    _, n_H, n_W, n_C = a_content.get_shape().as_list()
    norm_expr = 1 / (4 * n_H * n_W * n_C)

    return norm_expr * tf.reduce_sum(tf.square(tf.subtract(a_content, a_generated)))


def gram_matrix(a_unrolled):
    """
    Calculate Gram matrix for activations for a given layer, but unrolled to a matrix instead of tensor.
    :param a_unrolled: Matrix of shape (n_C, n_H * n_W)
    """

    # G = A * A_T
    return tf.matmul(a_unrolled, tf.transpose(a_unrolled))


def layer_style_cost(a_style, a_generated):
    """Calculate style cost for activations of a specific layer."""

    _, n_H, n_W, n_C = a_style.get_shape().as_list()

    # Unroll activation tensors into matrices, but change the order of dimensions so that n_C is first
    a_style_unrolled = tf.reshape(tf.transpose(a_style, perm=[3, 1, 2, 0], shape=[n_C, -1]))
    a_generated_unrolled = tf.reshape(tf.transpose(a_generated, perm=[3, 1, 2, 0], shape=[n_C, -1]))

    gram_style = gram_matrix(a_style_unrolled)
    gram_generated = gram_matrix(a_generated_unrolled)
    norm_expr = 1 / (4 * (n_C * n_H * n_W) ** 2)

    return norm_expr * tf.reduce_sum(tf.square(tf.subtract(gram_style, gram_generated)))


def style_cost(sess, model, style_layer_weights):
    """
    Calculate overall style cost of the model.
    :param model: Dict, keys - names of VGG layers, values - tensors
    :param style_layer_weights: Dict, keys - names of VGG layers, values - weights
    """

    cost = 0

    for layer_name, style_weight in style_layer_weights.items():
        # Extract output tensor from current layer
        out = model[layer_name]
        a_style = sess.run(out)
        a_generated = out  # will be calculated later, after assigning input image

        layer_style_cost_ = layer_style_cost(a_style, a_generated)
        cost += style_weight * layer_style_cost_

    return cost


def total_cost(content_cost, style_cost, alpha=12, beta=30):
    """Calculate total cost function as linear combination of content and style costs."""
    return alpha * content_cost + beta * style_cost