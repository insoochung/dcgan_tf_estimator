"""
Modification of https://github.com/sugyan/tf-dcgan/blob/master/dcgan.py
"""
import tensorflow as tf


class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(
                    inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(
                    outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            # fractional strided convolutions
            with tf.variable_scope('frac_stride_conv1'):
                outputs = tf.layers.conv2d_transpose(
                    outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('frac_stride_conv2'):
                outputs = tf.layers.conv2d_transpose(
                    outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('frac_stride_conv3'):
                outputs = tf.layers.conv2d_transpose(
                    outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('frac_stride_conv4'):
                outputs = tf.layers.conv2d_transpose(
                    outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')

        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs


class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [3] + depths

    def __call__(self, inputs, training=False, name=''):
        outputs = tf.convert_to_tensor(inputs)
        with tf.name_scope('d_' + name), tf.variable_scope('d', reuse=tf.AUTO_REUSE):
            # convolution x 4
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(
                    outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(
                    outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(
                    outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(
                    outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(
                        outputs, training=training), name='outputs')
            with tf.variable_scope('classify'):
                reshape = tf.reshape(outputs, [-1, 4 * 4 * self.depths[-1]])
                outputs = tf.layers.dense(reshape, 2, name='outputs')

        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs


def DCGAN(features, labels, mode, params):
    s_size = params['s_size']
    z_dim = params['z_dim']
    g_depths = params['g_depths']
    d_depths = params['d_depths']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    beta1 = params['beta1']

    generator = Generator(depths=g_depths, s_size=s_size)

    if mode == tf.estimator.ModeKeys.PREDICT:
        images = generator(features, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0),
                                              dtype=tf.uint8)
        images = [image for image in tf.split(images, batch_size, axis=0)]

        n_row = 4
        n_col = 8
        rows = []
        for i in range(n_row):
            rows.append(tf.concat(images[n_col * i: n_col * (i + 1)], 2))

        image = tf.concat(rows, 1)
        image = tf.map_fn(tf.image.encode_jpeg, image, dtype=tf.string)
        return tf.estimator.EstimatorSpec(mode, predictions=image)

    assert mode == tf.estimator.ModeKeys.TRAIN

    z = tf.random_uniform([batch_size, z_dim], minval=-1.0, maxval=1.0, name='z')
    discriminator = Discriminator(depths=d_depths)

    generated = generator(z, training=True)
    g_outputs = discriminator(generated, training=True, name='generated')
    t_outputs = discriminator(features, training=True, name='true')

    with tf.name_scope('g_loss'):
        g_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([batch_size], dtype=tf.int64),
                logits=g_outputs))
    with tf.name_scope('d_loss'):
        with tf.name_scope('true'):
            d_loss_true = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([batch_size], dtype=tf.int64),
                    logits=t_outputs))
        with tf.name_scope('generated'):
            d_loss_generated = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([batch_size], dtype=tf.int64),
                    logits=g_outputs))
        d_loss = d_loss_true + d_loss_generated

    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss_true', d_loss_true)
    tf.summary.scalar('d_loss_generated', d_loss_generated)
    tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('optimizer'):
        g_opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(
            g_loss,
            var_list=generator.variables,
            global_step=tf.train.get_global_step())
        d_opt_op = d_opt.minimize(d_loss, var_list=discriminator.variables)
        train_op = tf.group(g_opt_op, d_opt_op, name='train')

    return tf.estimator.EstimatorSpec(
        mode, loss=g_loss + d_loss, train_op=train_op)
