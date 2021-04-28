from neural_style_transfer import NeuralStyleTransfer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nst = NeuralStyleTransfer()

    config_list = [
        {
            'content_img_path': 'input/beatles.jpg',
            'style_img_path': 'input/colorful.jpg',
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 5e3,
            'num_iter': 1000
        },
        {
            'content_img_path': 'input/call-of-duty-mw2.jpg',
            'style_img_path': 'input/monet-grenouillere.jpg',
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 4e3,
            'num_iter': 1000
        },
        {
            'content_img_path': 'input/the-return-of-the-king.jpg',
            'style_img_path': 'input/wheat-field.png',
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 2e3,
            'num_iter': 1000
        },
        {
            'content_img_path': 'input/witcher.jpg',
            'style_img_path': 'input/winter-bridge.jpg',
            'noise_ratio': 0,
            'alpha': 1e5,
            'beta': 2e3,
            'num_iter': 1000
        },
    ]

    generated_images = nst.generate_images(config_list)
    # Eventually show images im matplotlib (make sure to call unnormalize_image() first)
    # for generated_image in generated_images:
    #     plt.imshow(unnormalize_image(generated_image))
    # plt.show()

