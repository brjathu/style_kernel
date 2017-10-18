import vgg
import tensorflow as tf
import numpy as np
from sys import stderr
import matplotlib.pylab as plt
from PIL import Image


tf.logging.set_verbosity(tf.logging.ERROR)

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

# exponential kernal
# sig = 30000


# mattern kernal
v = 2.5


try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
            content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
            learning_rate, beta1, beta2, epsilon, pooling, exp_sigma, mat_sigma, mat_rho, text_to_print,
            print_iterations=None, checkpoint_iterations=None, kernel=3, d=2):

    tf.logging.set_verbosity(tf.logging.INFO)
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]

    0 - dot product kernel
    1 - exponential kernel
    2 - matern kernel
    3 - polynomial kernel

    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))

                # sqr = features.T*features.T
                # dim = features.shape

                if(kernel == 0):
                    gram2 = np.matmul(features.T, features) / features.size
                elif(kernel == 1):
                    gram2 = gramExp_np(features, exp_sigma) / features.size  # exponential kernal
                elif(kernel == 2):
                    gram2 = gramMatten_np(features, mat_sigma, v, mat_rho) / features.size  # Mattern kernal
                elif(kernel == 3):
                    gram2 = gramPoly_np(features, d=d) / features.size

                # print(features.shape,"diamention of feature\n")
                style_features[i][layer] = gram2

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    g = tf.Graph()
    with g.as_default(), g.device('/gpu'):
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                net[content_layer] - content_features[content_layer]) /
                content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)

        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))

                style_gram = style_features[i][style_layer]

                dim = feats.get_shape()
                # print(dim)

                sqr = tf.reduce_sum(tf.transpose(feats) * tf.transpose(feats), axis=1)

                if(kernel == 0):
                    gram = (tf.matmul(tf.transpose(feats), feats)) / size
                elif(kernel == 1):
                    gram = tf.exp(-1 * (tf.transpose(tf.ones([dim[1], dim[1]]) * sqr) + tf.ones([dim[1], dim[1]]) * sqr - 2 *
                                        tf.matmul(tf.transpose(feats), feats)) / 2 / (exp_sigma * exp_sigma)) / size  # exponetial kernal
                elif(kernel == 2):
                    # mattern kernal
                    d2 = tf.nn.relu(tf.transpose(tf.ones([dim[1], dim[1]]) * sqr) + tf.ones([dim[1], dim[1]]) * sqr - 2 * tf.matmul(tf.transpose(feats), feats))
                    if(v == 0.5):
                        gram = mat_sigma**2 * tf.exp(-1 * tf.sqrt(d2) / mat_rho) / size
                    elif(v == 1.5):
                        gram = mat_sigma**2 * (tf.ones([dim[1], dim[1]]) + tf.sqrt(3.0) * tf.sqrt(d2) / mat_rho) * tf.exp(-1 * tf.sqrt(3.0) * tf.sqrt(d2) / mat_rho) / size
                    elif(v == 2.5):
                        gram = mat_sigma**2 * (tf.ones([dim[1], dim[1]]) + tf.sqrt(5.0) * tf.sqrt(d2) / mat_rho + 5 * d2 / 3 / (mat_rho**2)) * tf.exp(-1 * tf.sqrt(5.0) * tf.sqrt(d2) / mat_rho) / size
                elif(kernel == 3):
                    # polynomial kernal
                    gram = (tf.matmul(tf.transpose(feats), feats))**d / size

                style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)

            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])

        tv_loss = tv_weight * 2 * (
            (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
             tv_x_size))

        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress(last_loss):
            new_loss = loss.eval()
            stderr.write('file ===>  %s \n' % text_to_print)
            stderr.write('  content loss: %1.3e \t' % content_loss.eval())
            stderr.write('    style loss: %1.3e \t' % style_loss.eval())
            stderr.write('       tv loss: %1.3e \t' % tv_loss.eval())
            stderr.write('    total loss: %1.3e \t' % new_loss)
            stderr.write('    loss difference: %1.3e \t\n' % (last_loss - new_loss))
            return new_loss

        def save_progress():
            dict = {"content loss": content_loss.eval(), "style loss": style_loss.eval(), "tv loss": tv_loss.eval(), "total loss": loss.eval()}
            return dict

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            new_loss = 0
            # if (print_iterations and print_iterations != 0):
            #     print_progress()
            for i in range(iterations):
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                    new_loss = print_progress(new_loss)

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    dict = save_progress()
                    this_loss = loss.eval()
                    print(this_loss, "loss in each check point")
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    try:
                        img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
                    except:
                        print("uanlabe to result image due to given parameters")
                        img_out = "no  image"

                    if preserve_colors and preserve_colors:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

                    yield (
                        (None if last_step else i),
                        img_out, dict
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def gramExp_np(features, sigma):
    # exponential kernal
    sqr = features.T * features.T
    dim = features.shape
    return np.exp(-1 * ((np.ones((dim[1], dim[1]), dtype=np.int) * np.sum(sqr, axis=1)).T + (np.ones((dim[1], dim[1]),
                                                                                                     dtype=np.int) * np.sum(sqr, axis=1)) - 2 * (np.matmul(features.T, features))) / 2 / sigma / sigma)


def gramPoly_np(features, C=0, d=1):
    # Polynomial kernal
    return (np.matmul(features.T, features) + C)**d


def gramMatten_np(features, sigma, v, mat_rho):
    # matttern kernel
    sqr = features.T * features.T
    dim = features.shape
    d2 = abs((np.ones((dim[1], dim[1]), dtype=np.int) * np.sum(sqr, axis=1)).T + (np.ones((dim[1], dim[1]), dtype=np.int) * np.sum(sqr, axis=1)) - 2 * (np.matmul(features.T, features)))
    if(v == 0.5):
        return sigma**2 * np.exp(-1 * np.sqrt(d2) / mat_rho)
    if(v == 1.5):
        return sigma**2 * (np.ones((dim[1], dim[1]), dtype=np.int) + np.sqrt(3) * np.sqrt(d2) / mat_rho) * np.exp(-1 * np.sqrt(3) * np.sqrt(d2) / mat_rho)
    if(v == 2.5):
        return sigma**2 * (np.ones((dim[1], dim[1]), dtype=np.int) + np.sqrt(5) * np.sqrt(d2) / mat_rho + 5 * d2 / 3 / mat_rho**2) * np.exp(-1 * np.sqrt(5) * np.sqrt(d2) / mat_rho)


# polynomial kernel
