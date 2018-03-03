import os, argparse, glob
import numpy as np
import tensorflow as tf
from dcgan import DCGAN

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str,
                    help='either train or generate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')
parser.add_argument('--model_dir', default='model', type=str,
                    help='path to save trained model to')


def image_loader(features):
    image = tf.read_file(features)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 128, 128)
    image = tf.image.resize_images(image, [64, 64])
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 256.0)
    return image


def train_input_fn(features, labels, batch_size):
    with tf.name_scope('train_input_fn'):
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.map(image_loader, num_parallel_calls=8)
        dataset = dataset.shuffle(1000).repeat(1).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()


def generate_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.shuffle(1).repeat(1).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def get_celeb_a_features():
    image_paths = []
    for dir, _, filenames in os.walk('data/celebA'):
        for f in filenames:
            image_paths += [os.path.join(dir, f)]
    return image_paths

def write_image(image, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    image_path = os.path.join(path, name, 'jpg')
    print('Writing to ' + image_path)
    with open(image_path, 'wb') as f:
        f.write(image)


def generate(dcgan, args, n_row=4, n_col=8):
    z = np.load('data/z.npy')
    image = dcgan.predict(
        input_fn=lambda: generate_input_fn(z, None, batch_size=args.batch_size))
    image = list(image)[0]
    if not os.path.exists('result'):
        os.makedirs('result')
    path = 'result/{:03}'.format(len(glob.glob('result/*/')))
    write_image(image, path, 'random')

    inter_z = []
    for i in range(n_row):
        for j in range(n_col):
            inter_z += [z[i] * j / n_col + z[i + 1] * (n_col - j) / n_col]
    inter_z = np.array(inter_z)

    image = dcgan.predict(
        input_fn=lambda: generate_input_fn(inter_z, None, batch_size=args.batch_size))
    image = list(image)[0]
    write_image(image, path, 'interpolation')


def train(dcgan, args):
    image_paths = get_celeb_a_features()
    remainder_len = len(image_paths) % args.batch_size
    if remainder_len != 0:
        image_paths = image_paths[0:-remainder_len]
    dcgan.train(
        input_fn=lambda: train_input_fn(
            image_paths, None, args.batch_size),
        steps=args.train_steps
    )

def main(argv):
    args = parser.parse_args(argv[1:])

    dcgan = tf.estimator.Estimator(
        model_fn=DCGAN,
        params={
            'batch_size': args.batch_size,
            's_size': 4,
            'z_dim': 100,
            'g_depths': [1024, 512, 256, 128],
            'd_depths': [64, 128, 256, 512],
            'learning_rate': 0.0002,
            'beta1': 0.5
        },
        model_dir=args.model_dir
    )

    if args.mode == 'generate':
        generate(dcgan, args)
    elif args.mode == 'train':
        train(dcgan, args)
    else:
        print('Unsupported mode')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
