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
def floor_with_id_grad(x):
    def grad(dy):
        return dy
    return tf.floor(x), grad

@tf.custom_gradient
def round_with_const_grad(x):
    def grad(dy):
        return 1 + 0 * dy
    return tf.round(x), grad

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

def left_half(in_x, n):
    return tf.pow(2.0 * in_x, n) / 2.0
def right_half(in_x, n):
    return -tf.pow(2.0 * (1 - in_x), n) / 2.0 + 1.0

def staircase(x, alpha):
    return tf.where(tf.less(x, 0.5), left_half(x, alpha), right_half(x, alpha))

class HyperNetwork:
    def __init__(self, x, hparams, alpha):
        self.hparams = hparams
        self.latent_loss = 0.0

        with tf.name_scope('aec'):
            # self.build_net(x)
            # self.build_net_comparison(x)
            # self.build_net_contquant(x, alpha)
            self.build_net_contdiscretequant(x, alpha)

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

    def transform(self, x, alhpa):
        res = x - alhpa * (tf.math.sin(2 * np.pi * x) / (2 * np.pi))
        res = tf.math.maximum(0.0, res) 
        res = tf.math.minimum(self.hparams.quant_size, res)
        return res

    def transform_v2(self, x, alpha):
        # alpha = tf.Print(alpha, [alpha], message="alpha:")
        x = tf.math.maximum(0.0, x) 
        x = tf.math.minimum(self.hparams.quant_size, x)
        # x = tf.Print(x, [x], message="xxx:")
      
        floor_x = floor_with_id_grad(x)
        mantis = x - floor_x
        # mantis = tf.Print(mantis, [mantis], message="mantis:")
        
        stair = staircase(mantis, alpha)
        # stair = tf.Print(stair, [stair], message="stair:")

        res = floor_x + stair

        return res

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

        # encoded = tf.nn.sigmoid(encoded)
        # encoded = encoded * self.hparams.quant_size
        encoded = self.transform_v2(encoded, alpha)
        encoded = encoded / self.hparams.quant_size

        return encoded

    def decode(self, encoded):
        # encoded in [0, 1]
        x = 2 * (encoded - 0.5)

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
        # x = tf.nn.sigmoid(x)

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

        quantized = self.simple_quant(encoded)
        with tf.variable_scope("DECODER", reuse=True) as scope:
            quant_decoded = self.decode(quantized)
        
        self.encoded = encoded
        self.decoded = decoded
        self.cont_decoded = cont_decoded
        self.quant_decoded = quant_decoded

        tf.summary.histogram('encoded', tf.reshape(encoded, [-1]))
        tf.summary.histogram('cont_encoded', tf.reshape(cont_encoded, [-1]))

        print('Latent:', encoded.shape)
        print('Decoded:', decoded.shape)

    def build_net_contdiscretequant(self, x, alpha):
        with tf.variable_scope("ENCODER", reuse=False) as scope:
            encoded = self.encode(x, alpha)
        with tf.variable_scope("DECODER", reuse=False) as scope:
            decoded = self.decode(encoded)

        quantized = self.simple_quant(encoded)
        with tf.variable_scope("DECODER", reuse=True) as scope:
            quant_decoded = self.decode(quantized)
        
        self.encoded = encoded
        self.decoded = decoded
        self.quant_decoded = quant_decoded

        tf.summary.histogram('encoded', tf.reshape(encoded, [-1]))

        print('Latent:', encoded.shape)
        print('Decoded:', decoded.shape)

