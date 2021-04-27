from utils import *
import tensorflow as tf


def calculate_content_cost(a_content, a_generated):
    """Calculate content cost from activations of a chosen layer."""

    # Extract dimensions from activation tensor (ignore mini-batch dimension)
    _, n_H, n_W, n_C = a_content.shape
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

    _, n_H, n_W, n_C = a_style.shape

    # Unroll activation tensors into matrices, but change the order of dimensions so that n_C is first
    a_style_unrolled = tf.reshape(tf.transpose(a_style, perm=[3, 1, 2, 0]), shape=[n_C, -1])
    a_generated_unrolled = tf.reshape(tf.transpose(a_generated, perm=[3, 1, 2, 0]), shape=[n_C, -1])

    gram_style = gram_matrix(a_style_unrolled)
    gram_generated = gram_matrix(a_generated_unrolled)
    norm_expr = 1 / (4 * (n_C * n_H * n_W) ** 2)

    return norm_expr * tf.reduce_sum(tf.square(tf.subtract(gram_style, gram_generated)))


def calculate_style_cost(sess, model, style_layer_weights):
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

        layer_cost = layer_style_cost(a_style, a_generated)
        cost += style_weight * layer_cost

    return cost


def calculate_total_cost(content_cost, style_cost, alpha=10.0, beta=40.0):
    """Calculate total cost function as linear combination of content and style costs."""
    return alpha * content_cost + beta * style_cost


class NeuralStyleTransfer:
    STYLE_LAYER_WEIGHTS = {
        'conv1_1': 0.15,
        'conv2_1': 0.15,
        'conv3_1': 0.2,
        'conv4_1': 0.25,
        'conv5_1': 0.25
    }

    def __init__(self, content_layer_name='conv4_2'):
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        self.content_layer_name = content_layer_name
        self.sess = tf.compat.v1.InteractiveSession()
        self.model = load_vgg_model(Config.VGG_MODEL_PATH)

    def generate_image(self,
                       content_img_path,
                       style_img_path,
                       output_img_path='default',
                       style_layer_weights=STYLE_LAYER_WEIGHTS,
                       noise_ratio=0.2,
                       learning_rate=1,
                       alpha=1e5,
                       beta=1e3,
                       num_iter=500,
                       print_cost=True):
        # Load images, resize and normalize them to prepare them as inputs to VGG
        content_image = load_and_prepare_image(content_img_path)
        style_image = load_and_prepare_image(style_img_path)
        init_generated_image = generate_noise_image(content_image, noise_ratio=noise_ratio)

        # Pass content image through VGG
        self.sess.run(self.model['input'].assign(content_image))

        # Get activations for a content image from selected layer
        out = self.model[self.content_layer_name]  # This is still a placeholder!
        a_content = self.sess.run(out)  # This is evaluated to get contet activations
        a_generated = out  # Still a placeholder, will be evaluated in training iterations

        # Define content, style and total costs (evaluated in training)
        content_cost = calculate_content_cost(a_content, a_generated)
        self.sess.run(self.model['input'].assign(style_image))
        style_cost = calculate_style_cost(self.sess, self.model, style_layer_weights)
        total_cost = calculate_total_cost(content_cost, style_cost, alpha=alpha, beta=beta)

        # Define optimizer and training step to minimize total cost
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(total_cost)

        # Set generated, noisy image as input to the model and start training
        generated_image = init_generated_image
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.model['input'].assign(generated_image))

        for i in range(num_iter):
            # Minimize total cost
            self.sess.run(train_step)

            # New generated image after param updates
            generated_image = self.sess.run(self.model['input'])

            if print_cost and i % 20 == 0:
                _total_cost, _content_cost, _style_cost = self.sess.run([total_cost, content_cost, style_cost])
                print(f'\nIteration: {i}')
                print('------------------------------------')
                print(f'\t* Total cost: {_total_cost}')
                print(f'\t* Content cost: {_content_cost}')
                print(f'\t* Style cost: {_style_cost}')

        if output_img_path == 'default':
            content_name = get_image_name(content_img_path)
            style_name = get_image_name(style_img_path)
            output_name = content_name + '+' + style_name + f'_a={alpha},b={beta},iter={num_iter}'
            output_path = os.path.join(Config.OUTPUT_DIR, output_name + '.jpg')

            save_image(generated_image, output_path)
        else:
            save_image(generated_image, output_img_path)

        # Print prompt and show image immediately
        print('Image generated successfuly!')

        return generated_image
