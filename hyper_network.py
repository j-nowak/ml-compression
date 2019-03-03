import tensorflow as tf
import numpy as np

from utils import *

@tf.custom_gradient
def round_with_soft_grad(x):
    def grad(dy):
        return dy * (1 - 0.5 * tf.cos(2 * np.pi * x))
    return tf.round(x), grad

@tf.custom_gradient
def round_with_id_grad(x):
    def grad(dy):
        return dy
    return tf.round(x), grad

@tf.custom_gradient
def round_with_const_grad(x):
    def grad(dy):
        return 1 + 0 * dy
    return tf.round(x), grad

def _phase_shift(I, r):
       # Helper function with main phase shift operation
       _, a, b, c = I.get_shape().as_list()
       X = tf.reshape(I, (-1, a, b, r, r))
       X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
       X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
       X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
       X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
       return tf.reshape(X, (-1, a*r, b*r, 1))

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code
  if color:
    Xc = tf.split(3, 3, X)
    X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  else:
    X = _phase_shift(X, r)
  return X

def prelu(_x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def downsample(x, filters_num):
    x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(4, 4), strides=(2, 2), padding='same')
    x = tf.layers.batch_normalization(x)
    return x

def upsample(x, filters_num):
    x = tf.layers.conv2d_transpose(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x)
    return x

def residual(x, filters_num):
    ry = x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
    x = x + ry
    return x

class HyperNetwork:
    def __init__(self, x, hparams, alpha):
        self.hparams = hparams
        self.latent_loss = 0.0

        with tf.name_scope('aec'):
            # self.build_net(x)
            # self.build_net_comparison(x)
            self.build_net_contquant(x, alpha)

    def quntize(self, encoded):
        if self.hparams.quant_method == 1:
            q, qmin, qmax= tf.quantize(encoded, 0.0, 1.0, tf.qint8)
            dq = tf.dequantize(q, 0.0, 1.0)
        elif self.hparams.quant_method == 2:
            q = encoded * self.hparams.quant_size
            q = round_with_soft_grad(q)
            dq = q / self.hparams.quant_size
        elif self.hparams.quant_method == 3:
            q = encoded * self.hparams.quant_size
            q = round_with_id_grad(q)
            dq = q / self.hparams.quant_size
        elif self.hparams.quant_method == 4:
            q = encoded * self.hparams.quant_size
            q = round_with_const_grad(q)
            dq = q / self.hparams.quant_size
        elif self.hparams.quant_method == 5:
            q = encoded * self.hparams.quant_size
            q = tf.stop_gradient(tf.round(q))
            dq = q / self.hparams.quant_size
        else:
            q, dq = encoded, encoded

        # latent_hist = tf.summary.histogram('raw_latent', tf.reshape(encoded, [-1]))
        # latent_hist = tf.summary.histogram('discret_latent', tf.reshape(tf.cast(q, tf.int8), [-1]))

        return dq

    def encode(self, x, alpha=1.0):
        # 64x64x64
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 64x64x64
        x = residual(x, 64)
        x = residual(x, 64)

        # 32x32x128
        x = downsample(x, 128)

        # 32x32x128
        x = residual(x, 128)
        x = residual(x, 128)

        # 16x16x128
        x = downsample(x, 128)

        # 16x16x128
        x = residual(x, 128)
        x = residual(x, 128)

        # 8x8x128
        x = downsample(x, 128)

        # 8x8x128
        x = residual(x, 128)
        x = residual(x, 128)

        # 8x8x16
        encoded = x
        encoded = tf.layers.conv2d(inputs=encoded, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')

        encoded = encoded * alpha

        encoded = tf.nn.sigmoid(encoded)

        return encoded

    def decode(self, encoded):
        x = encoded

        # 8x8x64
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 8x8x64
        x = residual(x, 64)
        x = residual(x, 64)

        # 16x16x128
        x = upsample(x, 128)

        # 16x16x128
        x = residual(x, 128)
        x = residual(x, 128)

        # 32x32x128
        x = upsample(x, 128)

        # 32x32x128
        x = residual(x, 128)
        x = residual(x, 128)
        x = residual(x, 128)
        x = residual(x, 128)

        # 64x64x128
        x = upsample(x, 128)

        # 64x64x128
        x = residual(x, 128)
        x = residual(x, 128)

        # 64x64x16
        x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 64x64x3
        x = tf.layers.conv2d(inputs=x, filters=self.hparams.channels, kernel_size=(5, 5), strides=(1, 1), padding='same')
        x = tf.nn.sigmoid(x)

        # 64x64x3
        decoded = x
        return decoded

    def build_net(self, x):
        with tf.variable_scope("ENCODER", reuse=False) as scope:
            encoded = self.encode(x)

        quantized = self.quntize(encoded)

        with tf.variable_scope("DECODER", reuse=False) as scope:
            decoded = self.decode(quantized)
        
        print('Latent:', encoded.shape)
        print('Decoded:', decoded.shape)
        self.encoded = encoded
        self.quantized = quantized
        self.decoded = decoded
        self.quant_decoded = decoded

    def simple_quant(self, encoded):
        q = encoded * self.hparams.quant_size
        q = tf.stop_gradient(tf.round(q))
        dq = q / self.hparams.quant_size
        return dq

    def build_net_comparison(self, x):
        with tf.variable_scope("ENCODER", reuse=False) as scope:
            encoded = self.encode(x)
        with tf.variable_scope("DECODER", reuse=False) as scope:
            decoded = self.decode(encoded)

        quantized = self.simple_quant(encoded)
        with tf.variable_scope("DECODER", reuse=True) as scope:
            quant_decoded = self.decode(quantized)
        
        self.encoded = encoded
        self.decoded = decoded
        self.quantized = quantized
        self.quant_decoded = quant_decoded
        
        print('Latent:', encoded.shape)
        print('Decoded:', decoded.shape)

    def build_net_contquant(self, x, alpha):
        with tf.variable_scope("ENCODER", reuse=False) as scope:
            encoded = self.encode(x, 1.0)
        with tf.variable_scope("DECODER", reuse=False) as scope:
            decoded = self.decode(encoded)

        with tf.variable_scope("ENCODER", reuse=True) as scope:
            cont_encoded = self.encode(x, alpha)
        with tf.variable_scope("DECODER", reuse=True) as scope:
            cont_decoded = self.decode(cont_encoded)

        quantized = self.simple_quant(cont_encoded)
        with tf.variable_scope("DECODER", reuse=True) as scope:
            quant_decoded = self.decode(quantized)
        
        self.encoded = encoded
        self.decoded = decoded
        self.cont_decoded = cont_decoded
        self.quant_decoded = quant_decoded

        print('Latent:', encoded.shape)
        print('Decoded:', decoded.shape)

