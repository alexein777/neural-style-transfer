from neural_style_transfer import NeuralStyleTransfer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nst = NeuralStyleTransfer()

    config_list = [
        {
            'content_img_path': 'input/beatles.jpg',
            'style_img_path': 'input/rainy-night.jpg',
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 5e4,
            'num_iter': 1000
        },
        {
            'content_img_path': 'input/dragon.jpg',
            'style_img_path': 'input/storm.jpg',
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 3e3,
            'num_iter': 1000
        },
    ]

    generated_images = nst.generate_images(config_list)
    # Eventually show images im matplotlib (make sure to call unnormalize_image() first)
    # for generated_image in generated_images:
    #     plt.imshow(unnormalize_image(generated_image))
    # plt.show()

