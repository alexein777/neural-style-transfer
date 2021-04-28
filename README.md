# Neural Style Transfer

A classic NST project following the paper [Gatys et al. 2015.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
Given a content image and a style image, generate new image with the same content as content image, but having style transfered from style image to it.

## Examples

| Content    | Style      | Generated  |
| ------------- | ------------- | ------------- |
| ![beth-harmon](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/beth-harmon%3D(600%2C%20400).jpg) | ![scream](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/scream%3D(600%2C%20400).jpg) | ![beth-harmon+scream](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/beth-harmon%2Bscream%3D(600%2C%20400).jpg) |
| ![landscape](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/mountain-landscape%3D(600%2C%20400).jpg) | ![starry-night](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/van-gogh-starry-night%3D(600%2C%20400).jpg) | ![landscape+starry-night](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/landscape%2Bstarry-night%3D(600%2C%20400).jpg) |
| ![witcher](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/witcher%3D(600%2C%20400).jpg) | ![winter-bridge](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/winter-bridge%3D(600%2C%20400).jpg) | ![witcher+winter-bridge](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/witcher%2Bwinter-bridge_a%3D100000.0%2Cb%3D2000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![optimus-prime](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/optimus-prime%3D(600%2C%20400).jpg) | ![frostfire](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/frostfire%3D(600%2C%20400).jpg) | ![frostfire-optimus](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/optimus_prime%3D(600%2C%20400).jpg) |
| ![cod-mw2](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/call-of-duty-mw2%3D(600%2C%20400).jpg) | ![grenouillere](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/monet-grenouillere%3D(600%2C%20400).jpg) | ![monet-cod-m2](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/call-of-duty-mw2%2Bmonet-grenouillere_a%3D100000.0%2Cb%3D4000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![dragon](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/dragon%3D(600%2C%20400).jpg) | ![storm](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/storm%3D(600%2C%20400).jpg) | ![storm-dragon](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/dragon%2Bstorm_a%3D100000.0%2Cb%3D3000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![jon-and-daenerys](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/jon-and-daenerys%3D(600%2C%20400).jpg) | ![monet](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/monet%3D(600%2C%20400).jpg) | ![jon-and-daenerys+monet](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/jon-and-daenerys%2Bmonet_a%3D100000.0%2Cb%3D3000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![minas-tirith](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/the-return-of-the-king%3D(600%2C%20400).jpg) | ![wheat-field](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/wheat-field%3D(600%2C%20400).jpg) | ![wheat-minas-tirith](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/the-return-of-the-king%2Bwheat-field_a%3D100000.0%2Cb%3D2000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![jack-sparrow](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/captain-jack-sparrow%3D(600%2C%20400).jpg) | ![kosovo-maiden](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/kosovo-maiden%3D(600%2C%20400).jpg) | ![serbian-jack-sparrow](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/captain-jack-sparrow%2Bkosovo-maiden_a%3D100000.0%2Cb%3D3000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![neo-trinity](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/neo-trinity%3D(600%2C%20400).jpg) | ![guernica](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/gernika%3D(600%2C%20400).jpg) | ![neo-trinity+guernica](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/neo-trinity%2Bgernika_a%3D100000.0%2Cb%3D3000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![golden-gate](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/golden-gate%3D(600%2C%20400).jpg) | ![wave](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/wave%3D(600%2C%20400).jpg) | ![wavy-golden-gate](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/golden_gate%3D(600%2C%20400).jpg) |
| ![lion](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/lion%3D(600%2C%20400).jpg) | ![mona-lisa](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/mona-lisa%3D(600%2C%20400).jpg) | ![lion+mona-lisa](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/lion%2Bmona-lisa_a%3D100000.0%2Cb%3D4000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |
| ![beatles](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/beatles%3D(600%2C%20400).jpg) | ![colorful](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/colorful%3D(600%2C%20400).jpg) | ![colorful-beatles](https://github.com/alexein777/neural-style-transfer/blob/master/output/combo/beatles%2Bcolorful_a%3D100000.0%2Cb%3D5000.0%2Citer%3D1000%3D(600%2C%20400).jpg) |

## Training info

Most generated images used content image as initial generated image (`noise_ratio=0`). All generated images used `\alpha=1e5` (contet loss weight), while `\beta` (style loss weight) varied from `1e2` to `1e5`.
Number of iterations used for all generated images is `1000` (results could probably be a bit better with more iterations, but it takes about ~30min for 1000 iterations on my machine).

## Important notes

Model used for image generation is VGG-19. However, model isn't pushed to repo due to size, so function `load_vgg_model()` won't work. Model can be downloaded [here](https://www.kaggle.com/keras/vgg19) and has to be put in `models/imagenet-vgg-19.mat`, or simply change `load_vgg_model()` function to load model from your local machine.

Code is written using TensorFlow v1.0 and carries some "ugliness" with it. I will migrate to TensorFlow v2.0 or PyTorch as soon as possible.

## Acknowledgements

Check out some other cool repos on Neural Style Transfer, like https://github.com/gordicaleksa/pytorch-neural-style-transfer (hyperparameter values used in this repo are inspired by @gordicaleksa's work).
