import os
import numpy as np
import scipy.misc
from stylize import stylize
import math
from argparse import ArgumentParser
from PIL import Image
import pickle

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2  # 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 10
PRINT_ITERATIONS = 1
VGG_PATH = 'imagenet-vgg-verydeep-197.mat'
POOLING = 'max'
RANGE_SIGMA = [16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000]
RANGE_SW = [1e16, 1e17, 1e18, 1e19, 1e20, 1e21,1e22, 1e23, 1e24]
# CONTENT_IMAGES = sorted(os.listdir("examples/content"))
STYLE_IMAGES = sorted(os.listdir("examples/style"))




def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
                        dest='content',nargs='+', help='content image',
                        metavar='CONTENT')
    parser.add_argument('--styles',
                        dest='styles',
                        nargs='+', help='one or more style images',
                        metavar='STYLE')
    parser.add_argument('--output',
                        dest='output', help='output path',
                        metavar='OUTPUT')
    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='iterations (default %(default)s)',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='statistics printing frequency',
                        metavar='PRINT_ITERATIONS', default=PRINT_ITERATIONS)
    parser.add_argument('--checkpoint-output',
                        dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
                        metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
                        dest='width', help='output width',
                        metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
                        dest='style_scales',
                        nargs='+', help='one or more style scales',
                        metavar='STYLE_SCALE')
    parser.add_argument('--network',
                        dest='network', help='path to network parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
                        dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
                        metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
                        dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
                        metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
                        dest='style_blend_weights', help='style blending weights',
                        nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
                        dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
                        metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
                        dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
                        metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
                        dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
                        metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
                        dest='initial', help='initial image',
                        metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
                        dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
                        metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
                        dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
                        dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
                        metavar='POOLING', default=POOLING)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)
    CONTENT_IMAGES = options.content
    count = 0
    try:
        os.system("mkdir final")
        os.system("mkdir final/exp")
        os.system("mkdir final/pickle_exp")
    except:
        print("dirctory stucture already exist")

    for c in CONTENT_IMAGES:
        for s in STYLE_IMAGES:
            for sig in RANGE_SIGMA:
                locat = os.listdir("final/exp/")
                if(not(sig in locat)):
                    os.system("mkdir final/exp/"+ str(sig))
                    os.system("mkdir final/pickle_exp/"+str(sig))
                for sw in RANGE_SW:
                    c_image = imread(c)
                    s_image = [imread("examples/style/" + s)]

                    width = options.width
                    if width is not None:
                        new_shape = (int(math.floor(float(c_image.shape[0]) /
                                                    c_image.shape[1] * width)), width)
                        c_image = scipy.misc.imresize(c_image, new_shape)
                    target_shape = c_image.shape
                    for i in range(len(s_image)):
                        style_scale = STYLE_SCALE
                        if options.style_scales is not None:
                            style_scale = options.style_scales[i]
                        s_image[i] = scipy.misc.imresize(s_image[i], style_scale *
                                                         target_shape[1] / s_image[i].shape[1])

                    style_blend_weights = options.style_blend_weights
                    if style_blend_weights is None:
                        # default is equal weights
                        style_blend_weights = [1.0 / 1]
                    else:
                        total_blend_weight = sum(style_blend_weights)
                        style_blend_weights = [weight / total_blend_weight
                                               for weight in style_blend_weights]

                    initial = options.initial
                    if initial is not None:
                        initial = scipy.misc.imresize(imread(initial), c_image.shape[:2])
                        # Initial guess is specified, but not noiseblend - no noise should be blended
                        if options.initial_noiseblend is None:
                            options.initial_noiseblend = 0.0
                    else:
                        # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
                        if options.initial_noiseblend is None:
                            options.initial_noiseblend = 1.0
                        if options.initial_noiseblend < 1.0:
                            initial = c_image

                    if options.checkpoint_output and "%s" not in options.checkpoint_output:
                        parser.error("To save intermediate images, the checkpoint output "
                                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

                    count += 1
                    sname = c.split("/")[-1][0:-4] + "_" + s[0:-4] + "_" + str(sw) + "_" + str(sig) + ".jpg"
                    print("loop ==> " + str(count) + "of" + str(len(CONTENT_IMAGES) * len(STYLE_IMAGES) * len(RANGE_SIGMA) * len(RANGE_SW)))

                    # print("--content " + c + " --styles " + s + " --output final/exp/" + sname + " --iterations 1000 --style-weight " + str(sw))
                    files_in_folder = os.listdir("final/exp/" + str(sig) + "/")
                    if(sname in files_in_folder):
                        print("file already exist")
                        continue

                    text_file = open("Output.txt", "a")
                    text_file.write("image name: " + sname + '\n')
                    text_file.close()
                    for iteration, image, dict in stylize(
                        network=options.network,
                        initial=initial,
                        initial_noiseblend=options.initial_noiseblend,
                        content=c_image,
                        styles=s_image,
                        preserve_colors=options.preserve_colors,
                        iterations=ITERATIONS,
                        content_weight=options.content_weight,
                        content_weight_blend=options.content_weight_blend,
                        style_weight=sw,
                        style_layer_weight_exp=options.style_layer_weight_exp,
                        style_blend_weights=style_blend_weights,
                        tv_weight=options.tv_weight,
                        learning_rate=options.learning_rate,
                        beta1=options.beta1,
                        beta2=options.beta2,
                        epsilon=options.epsilon,
                        pooling=options.pooling,
                        print_iterations=PRINT_ITERATIONS,
                        checkpoint_iterations=options.checkpoint_iterations,
                        exp_sigma=sig,
                        text_to_print= sname
                    ):
                        print(dict)
                        try:
                            combined_rgb = image
                            imsave("final/exp/" + str(sig) + "/" + sname, combined_rgb)
                            with open("final/pickle_exp/"+str(sig)+"/"+sname[0:-4] + ".pkl", 'wb') as f:
                                pickle.dump(dict, f)

                        except:
                            print("============= ERROR =============")


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


if __name__ == '__main__':
    main()
