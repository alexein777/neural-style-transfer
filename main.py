from neural_style_transfer import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nst = NeuralStyleTransfer()
    generated_image = nst.generate_image(content_img_path='input/jon-and-daenerys.jpg',
                                         style_img_path='input/monet.jpg',
                                         noise_ratio=0,
                                         alpha=1e5,
                                         beta=3e3,
                                         num_iter=1000)

    im = unnormalize_image(generated_image)
    plt.imshow(im)
    plt.show()

