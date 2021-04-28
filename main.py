from neural_style_transfer import NeuralStyleTransfer
import matplotlib.pyplot as plt
import os

CONTENT_INPUT_PATH = os.path.join('input', 'content')
STYLE_INPUT_PATH = os.path.join('input', 'style')

if __name__ == '__main__':
    nst = NeuralStyleTransfer()

    config_list = [
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'bilbo.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'sunny-forrest.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 3e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'eiffel.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'monet-twilight-venice.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 4e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'hourglass.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'sass.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 6e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'jesus-disciples.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'sea-fall.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 2e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'knightfall.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'monet-soleil-levant.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 5e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'la-casa-de-papel.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'abstract-painting.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 4e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'skyrim.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'autumn-valley.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 5e3,
            'num_iter': 1000
        },
        {
            'content_img_path': os.path.join(CONTENT_INPUT_PATH, 'selfie-rooftop.jpg'),
            'style_img_path': os.path.join(STYLE_INPUT_PATH, 'abstract-colorful.jpg'),
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 3e3,
            'num_iter': 1000
        }
    ]

    generated_images = nst.generate_images(config_list)
    # Eventually show images im matplotlib (make sure to call unnormalize_image() first)
    # for generated_image in generated_images:
    #     plt.imshow(unnormalize_image(generated_image))
    # plt.show()

