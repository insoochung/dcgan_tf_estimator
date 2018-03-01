import os
import argparse
import tensorflow as tf
from dcgan import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int, help='number of training steps')
parser.add_argument('--model_dir', default='model', type=str, help='path to save trained model to')

def image_loader(features):
    image = tf.read_file(features)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 178, 178)
    image = tf.image.resize_images(image, [64, 64])
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 256.0)
    return image

def train_input_fn(features, labels, batch_size):
    # train_data = tf.convert_to_tensor(tf.constant(features, dtype=tf.string))
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.map(
        image_loader, num_parallel_calls=8)
    dataset = dataset.shuffle(len(features) * 2).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

def get_celeb_a_features():
    image_path_list = []
    for dir, _, filenames in os.walk('data/celebA'):
        for f in filenames:
            image_path_list += [os.path.join(dir, f)]
    return image_path_list

def main(argv):
    args = parser.parse_args(argv[1:])

    train_data = get_celeb_a_features()
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

    dcgan.train(
        input_fn=lambda:train_input_fn(train_data, None, args.batch_size),
        steps=args.train_steps
    )

if __name__ == '__main__':
    tf.app.run(main)
