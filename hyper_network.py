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
    # return tf.nn.relu(_x)
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

class HyperNetwork:
    def __init__(self, x, hparams, layers, compute_bias):
        self.hparams = hparams
        self.latent_loss = 0.0

        with tf.name_scope('hyper_network'):
            if self.hparams.resnet_type == 1:
                self.__new_resnet(x, hparams, layers, compute_bias)
            elif self.hparams.resnet_type == 2:
                self.__old_resnet(x, hparams, layers, compute_bias)
            elif self.hparams.resnet_type == 3:
                self.prelu_resnet(x, layers, compute_bias)
            else:
                self.__resnet(x, hparams, layers, compute_bias)

    def __quntize(self, encoded):
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

    def residual_unit(self, x, filters_num):
        ry = x
        x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.layers.batch_normalization(x)
        x = prelu(x)
        x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.layers.batch_normalization(x)
        x = x + ry
        # x = prelu(x)
        return x

    def prelu_encode(self, x):
        # 64x64x64
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')
        x = prelu(x)

        # 64x64x128
        x = self.residual_unit(x, 128)
        x = self.residual_unit(x, 128)

        # 32x32x128
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')

        # 32x32x128
        x = self.residual_unit(x, 128)
        x = self.residual_unit(x, 128)

        # 16x16x128
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')

        # 16x16x128
        x = self.residual_unit(x, 128)
        x = self.residual_unit(x, 128)

        # 8x8x128
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')

        # 8x8x128
        x = self.residual_unit(x, 128)
        x = self.residual_unit(x, 128)

        # 8x8x64
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
        x = tf.nn.sigmoid(x)
        return x

    def prelu_decode(self, x):
        # # 8x8x64
        # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
        # x = prelu(x)

        # # 8x8x64
        # x = self.residual_unit(x, 64)
        # x = self.residual_unit(x, 64)

        # # # 64x64x32
        # x = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=(4, 4), strides=2, padding='same')
        # print('xxxxx', x.shape)
        # # x = PS(x, 8)
        # # x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        # x = prelu(x)

        # # 64x64x32
        # x = self.residual_unit(x, 32)
        # x = self.residual_unit(x, 32)

        # # 64x64x64
        # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same')
        # x = prelu(x)

        # # 64x64x64
        # x = self.residual_unit(x, 64)
        # x = self.residual_unit(x, 64)

        # # # 16x16x64
        # # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')
        # # x = prelu(x)

        # # 16x16x64
        # # x = self.residual_unit(x, 64)
        # # x = self.residual_unit(x, 64)

        # # 16x16x64
        # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')

        # 8x8x32
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')
        x = prelu(x)

        # 8x8x32
        x = self.residual_unit(x, 32)
        x = self.residual_unit(x, 32)

        # 16x16x64
        x = tf.layers.conv2d_transpose(inputs=x, filters=32, kernel_size=(4, 4), strides=2, padding='same')
        x = prelu(x)

        # 16x16x64
        x = self.residual_unit(x, 32)
        x = self.residual_unit(x, 32)

        # 32x32x128
        x = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=(4, 4), strides=2, padding='same')
        x = prelu(x)

        # 32x32x128
        x = self.residual_unit(x, 64)
        x = self.residual_unit(x, 64)

        # 64x64x128
        x = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=(4, 4), strides=2, padding='same')
        x = prelu(x)

        # 64x64x128
        x = self.residual_unit(x, 64)
        x = self.residual_unit(x, 64)

        # 64x64x64
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')

        # 512x512x1
        # x = PS(x, 8)
        x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

        return x

    def prelu_build_weights(self, decoded, layers, compute_bias):
        print('DDDD', decoded.shape)

        total_size = 0
        all_matrices, all_biases = [], []

        for i in range(len(layers)):
            out_size = np.prod(layers[i])
            if compute_bias:
                bias_size = layers[i][-1]
            else:
                bias_size = 0

            w = decoded[:, total_size:total_size + out_size]
            total_size = total_size + out_size

            b = decoded[:, total_size:total_size + bias_size]
            total_size = total_size + bias_size

            w = tf.reshape(w, [-1] + layers[i])

            print('www', w.shape)
            print('total_size', total_size)

            all_matrices.append(w)
            if bias_size > 0:
                b = tf.reshape(b, [-1, layers[i][-1]])
                all_biases.append(b)
        return all_matrices, all_biases


    def prelu_resnet(self, x, layers, compute_bias):
        if self.hparams.in_img_width != self.hparams.in_img_height:
            raise Exception("This model doesn't support different in width and height")
        x = tf.reshape(x, shape=[-1, self.hparams.in_img_width, self.hparams.in_img_height, self.hparams.channels])

        with tf.variable_scope("ENCODER", reuse=False) as scope:
            encoded = self.prelu_encode(x)

        quantized = self.__quntize(encoded)

        with tf.variable_scope("DECODER", reuse=False) as scope:
            decoded = self.prelu_decode(quantized)
        with tf.variable_scope("WEIGHT_BUILDER", reuse=False) as scope:
            self.matrices, self.bsss = self.prelu_build_weights(decoded, layers, compute_bias)
        
        print('Latent:', encoded.shape)

    def downsample(self, x, filters_num):
        x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(4, 4), strides=(2, 2), padding='same')
        # x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)
        return x

    def encode(self, x):
        # 64x64x64
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 64x64x64
        x = self.standard_residual(x, 64)
        x = self.standard_residual(x, 64)

        # 32x32x128
        x = self.downsample(x, 128)

        # 32x32x128
        x = self.standard_residual(x, 128)
        x = self.standard_residual(x, 128)

        # 16x16x128
        x = self.downsample(x, 128)

        # 16x16x128
        x = self.standard_residual(x, 128)
        x = self.standard_residual(x, 128)

        # 8x8x128
        x = self.downsample(x, 128)

        # 8x8x128
        x = self.standard_residual(x, 128)
        x = self.standard_residual(x, 128)

        # 8x8x64
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 8x8x64
        x = self.standard_residual(x, 64)
        x = self.standard_residual(x, 64)     

        # 8x8x16
        encoded = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.sigmoid)
        return encoded

    # def standard_residual(self, x, filters_num):
    #     ry = x
    #     x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     x = tf.layers.batch_normalization(x)
    #     x = tf.nn.relu(x)
    #     x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     x = tf.layers.batch_normalization(x)
    #     x = x + ry
    #     x = tf.nn.relu(x)
    #     return x

    # full pre-activation
    def standard_residual(self, x, filters_num):
        ry = x
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x, filters=filters_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = x + ry
        return x

    def decode_other(self, encoded):
        x = encoded

        # 32x32x16
        x = PS(x, 4)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')

        # 8x8x16
        x = self.standard_residual(x, 32)
        x = self.standard_residual(x, 32)
        # x = self.standard_residual(x, 16)

        # 32x32x64
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 32x32x64
        x = self.standard_residual(x, 64)
        x = self.standard_residual(x, 64)
        # x = self.standard_residual(x, 16)

        # 16x16x64
        x = self.downsample(x, 64)

        # 16x16x64
        x = self.standard_residual(x, 64)
        x = self.standard_residual(x, 64)

        # 8x8x128
        x = self.downsample(x, 128)

        # 8x8x128
        x = self.standard_residual(x, 128)
        x = self.standard_residual(x, 128)

        # 8x8x256
        x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)

        # 8x8x256
        x = self.standard_residual(x, 256)
        x = self.standard_residual(x, 256)

        # 8x8x256
        decoded = x
        # decoded = tf.layers.conv2d(inputs=decoded, filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same')
        # decoded = tf.nn.tanh(decoded)
        # decoded = tf.layers.batch_normalization(decoded)
        print('DECODED', decoded.shape)
        return decoded


    # def decode_other(self, encoded):
    #     x = tf.layers.batch_normalization(encoded)

    #     # 8x8x16
    #     x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')
    #     x = tf.nn.relu(x)
    #     x = tf.layers.batch_normalization(x)

    #     # 8x8x16
    #     x = self.standard_residual(x, 16)
    #     x = self.standard_residual(x, 16)
    #     # x = self.standard_residual(x, 16)

    #     # 32x32x1
    #     # x = PS(x, 4)

    #     # 32x32x16
    #     # x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')
    #     # x = tf.nn.relu(x)
    #     # x = tf.layers.batch_normalization(x)

    #     # 32x32x16
    #     # x = self.standard_residual(x, 16)
    #     # x = self.standard_residual(x, 16)
    #     # x = self.standard_residual(x, 16)

    #     # 128x128x1
    #     # x = PS(x, 4)

    #     # 64x64x32
    #     x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     # x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')
    #     x = tf.nn.relu(x)
    #     x = tf.layers.batch_normalization(x)

    #     # 64x64x32
    #     x = self.standard_residual(x, 32)
    #     x = self.standard_residual(x, 32)
    #     # x = self.standard_residual(x, 32)

    #     # 32x32x64
    #     x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')
    #     x = tf.nn.relu(x)
    #     x = tf.layers.batch_normalization(x)

    #     # 32x32x64
    #     x = self.standard_residual(x, 64)
    #     x = self.standard_residual(x, 64)

    #     # 16x16x64
    #     x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')
    #     x = tf.nn.relu(x)
    #     x = tf.layers.batch_normalization(x)

    #     # 16x16x64
    #     x = self.standard_residual(x, 64)
    #     x = self.standard_residual(x, 64)

    #     # 8x8x64
    #     x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     # x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')
    #     x = tf.nn.relu(x)
    #     x = tf.layers.batch_normalization(x)

    #     # 8x8x128
    #     x = self.standard_residual(x, 128)
    #     x = self.standard_residual(x, 128)

    #     # 8x8x256
    #     x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')
    #     # x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')
    #     x = tf.nn.relu(x)
    #     x = tf.layers.batch_normalization(x)

    #     # 8x8x256
    #     x = self.standard_residual(x, 256)
    #     x = self.standard_residual(x, 256)

    #     # 8x8x256
    #     decoded = x
    #     # decoded = tf.layers.conv2d(inputs=decoded, filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same')
    #     # decoded = tf.nn.tanh(decoded)
    #     # decoded = tf.layers.batch_normalization(decoded)
    #     print('DECODED', decoded.shape)
    #     return decoded

    def build_weights(self, decoded, layers, compute_bias):
        # print('ddddd', decoded.shape)
        y = decoded
        # prev_filters = 256
        # small_filters = prev_filters
        # for i in range(int(prev_filters / 64)):
        #     y = next_layer(y, small_filters, int(small_filters/2), 100)
        #     y = tf.layers.batch_normalization(y)
        #     y = tf.nn.relu(y)
        #     small_filters = int(small_filters/2)

        small_weights_out = y
        small_weights_out = tf.reshape(small_weights_out, [-1, np.prod(small_weights_out.get_shape().as_list()[1:])])
        current_out_dim = np.prod(small_weights_out.get_shape().as_list()[1:])
        all_matrices = []
        all_biases = []

        for i in range(len(layers)):
            if compute_bias:
                bias_size = layers[i][-1]
            else:
                bias_size = 0

            out_size = np.prod(layers[i])

            if out_size <= 8*8*64:
                w,b = self.handle_small_weights(out_size, bias_size, small_weights_out)
            # elif out_size <= current_out_dim*2:
            #     w,b = self.handle_medium_weights(out_size, bias_size, decoded)
            else:
                w,b = self.handle_big_weights(out_size, bias_size, decoded)

            w = tf.reshape(w, [-1] + layers[i])
            all_matrices.append(w)
            if bias_size > 0:
                b = tf.reshape(b, [-1, layers[i][-1]])
                all_biases.append(b)
        return all_matrices, all_biases


    def __new_resnet(self, x, hparams, layers, compute_bias):
        if self.hparams.in_img_width != self.hparams.in_img_height:
            raise Exception("This model doesn't support different in width and height")
        x = tf.reshape(x, shape=[-1, self.hparams.in_img_width, self.hparams.in_img_height, hparams.channels])

        with tf.variable_scope("ENCODER", reuse=False) as scope:
            encoded = self.encode(x)

        quantized = self.__quntize(encoded)

        with tf.variable_scope("DECODER", reuse=False) as scope:
            # decoded = self.decode(encoded)
            decoded = self.decode_other(quantized)
        with tf.variable_scope("WEIGHT_BUILDER", reuse=False) as scope:
            self.matrices, self.bsss = self.build_weights(decoded, layers, compute_bias)
        
        print('Latent:', encoded.shape)

        # q_encoded = tf.stop_gradient(self.__quntize(encoded))
        # self.latent_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(q_encoded - encoded), [1, 2, 3]))
        # with tf.variable_scope("DECODER", reuse=True) as scope:
        #     quant_decoded = self.decode(q_encoded)
        # with tf.variable_scope("WEIGHT_BUILDER", reuse=True) as scope:
        #     self.quant_matrices, self.quant_biases = self.build_weights(quant_decoded, layers, compute_bias)

    def __resnet(self, x, hparams, layers, compute_bias):
        if self.hparams.in_img_width != self.hparams.in_img_height:
            raise Exception("This model doesn't support different in width and height")
        
        def conv_layer(x, conv_size, filters_in, filters_out, name, strides = (1,1)):
            with tf.name_scope(name):
                filter_a = default_tf_variable([conv_size, 1, filters_in, filters_out])
                filter_b = default_tf_variable([1, conv_size, filters_out, filters_out])
            x = tf.nn.conv2d(x, filter_a, strides=[1, strides[0], strides[1], 1], padding='SAME')
            x = tf.nn.conv2d(x, filter_b, strides=[1, strides[0], strides[1], 1], padding='SAME')
            return x
        
        def bias(x, bias_size, name):
            with tf.name_scope(name):
                bias = default_tf_variable([bias_size])
            x = tf.nn.bias_add(x, bias)
            return x

        def next_layer(x, prev_filters, filters_out, i, layer_size = 3, strides=(1,1)):
            x = conv_layer(x, layer_size, prev_filters, filters_out, 'resnet_layer' + str(i), strides)
            x = bias(x, filters_out, 'resnet_bias' + str(i))
            return x
                                           
        x = tf.reshape(x, shape=[-1, self.hparams.in_img_width, self.hparams.in_img_height, hparams.channels])
        pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        five_five = conv_layer(x, 5, hparams.channels, 10, 'inception_conv_1')
        three_three = conv_layer(x, 3, hparams.channels, 10, 'inception_conv_2')
        one_one = conv_layer(x, 1, hparams.channels, 9, 'inception_conv_3')
        x = tf.concat([one_one, three_three, five_five, pooling], axis=3)
        x = bias(x, 32, 'inception_bias')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        if self.hparams.in_img_width % 8 != 0:
            raise Exception('Size must be diviable by 8')
        iterations = self.hparams.in_img_width / 8
        prev_filters = 32
        j = 0
        for i in range(int(np.log2(iterations))):
            ry = x
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             ry1 = x + ry
#             x = ry1
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             ry2 = x + ry1
#             x = ry2
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
            x = x + ry
#             print(x.shape)
            x = next_layer(x, prev_filters, int(prev_filters*2), j)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
            prev_filters = int(prev_filters*2)
        
        y = x
        small_filters = prev_filters
        for i in range(int(prev_filters / 64)):
            y = next_layer(y, small_filters, int(small_filters/2), j)
            y = tf.layers.batch_normalization(y)

            if i < prev_filters / 64 - 1:
                y = tf.nn.relu(y)
            else:
                y = tf.nn.tanh(y)
            small_filters = int(small_filters/2)

        encoded = self.__quntize(y)

        small_weights_out = encoded
        small_weights_out = tf.reshape(small_weights_out, [-1, np.prod(small_weights_out.get_shape().as_list()[1:])])
        current_out_dim = np.prod(small_weights_out.get_shape().as_list()[1:])
        self.matrices = []
        self.bsss = []
        for i in range(len(layers)):
            if compute_bias:
                bias_size = layers[i][-1]
            else:
                bias_size = 0

            out_size = np.prod(layers[i])

            if out_size < 8*8*64:
                w,b = self.handle_small_weights(out_size, bias_size, small_weights_out)
            elif out_size <= current_out_dim*2:
                w,b = self.handle_medium_weights(out_size, bias_size, encoded)
            else:
                w,b = self.handle_big_weights(out_size, bias_size, encoded)

            w = tf.reshape(w, [-1] + layers[i])
            self.matrices.append(w)
            if bias_size > 0:
                b = tf.reshape(b, [-1, layers[i][-1]])
                self.bsss.append(b)

    def __old_resnet(self, x, hparams, layers, compute_bias):
        if self.hparams.in_img_width != self.hparams.in_img_height:
            raise Exception("This model doesn't support different in width and height")
        
        def conv_layer(x, conv_size, filters_in, filters_out, name, strides = (1,1)):
            with tf.name_scope(name):
                filter_a = default_tf_variable([conv_size, 1, filters_in, filters_out])
                filter_b = default_tf_variable([1, conv_size, filters_out, filters_out])
            x = tf.nn.conv2d(x, filter_a, strides=[1, strides[0], strides[1], 1], padding='SAME')
            x = tf.nn.conv2d(x, filter_b, strides=[1, strides[0], strides[1], 1], padding='SAME')
            return x
        
        def bias(x, bias_size, name):
            with tf.name_scope(name):
                bias = default_tf_variable([bias_size])
            x = tf.nn.bias_add(x, bias)
            return x

        def next_layer(x, prev_filters, filters_out, i, layer_size = 3, strides=(1,1)):
            x = conv_layer(x, layer_size, prev_filters, filters_out, 'resnet_layer' + str(i), strides)
            x = bias(x, filters_out, 'resnet_bias' + str(i))
            return x
                                           
        x = tf.reshape(x, shape=[-1, self.hparams.in_img_width, self.hparams.in_img_height, hparams.channels])
        pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        five_five = conv_layer(x, 5, hparams.channels, 10, 'inception_conv_1')
        three_three = conv_layer(x, 3, hparams.channels, 10, 'inception_conv_2')
        one_one = conv_layer(x, 1, hparams.channels, 9, 'inception_conv_3')
        x = tf.concat([one_one, three_three, five_five, pooling], axis=3)
        x = bias(x, 32, 'inception_bias')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        
        if self.hparams.in_img_width % 8 != 0:
            raise Exception('Size must be diviable by 8')
        iterations = self.hparams.in_img_width / 8
        prev_filters = 32
        j = 0
        for i in range(int(np.log2(iterations))):
            ry = x
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             ry1 = x + ry
#             x = ry1
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             ry2 = x + ry1
#             x = ry2
            x = next_layer(x, prev_filters, prev_filters, j)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
            x = x + ry
#             print(x.shape)
            x = next_layer(x, prev_filters, int(prev_filters*2), j)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            j += 1
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
            prev_filters = int(prev_filters*2)
        
        y = x
        small_filters = prev_filters
        for i in range(int(prev_filters / 64)):
            y = next_layer(y, small_filters, int(small_filters/2), j)
            y = tf.layers.batch_normalization(y)
            y = tf.nn.relu(y)
            small_filters = int(small_filters/2)

        small_weights_out = y
        small_weights_out = tf.reshape(small_weights_out, [-1, np.prod(small_weights_out.get_shape().as_list()[1:])])
        current_out_dim = np.prod(small_weights_out.get_shape().as_list()[1:])
        self.matrices = []
        self.bsss = []
        for i in range(len(layers)):
            if compute_bias:
                bias_size = layers[i][-1]
            else:
                bias_size = 0

            out_size = np.prod(layers[i])
            bias_size = layers[i][-1]
            if out_size <= 8*8*128:
                w,b = self.handle_small_weights(out_size, bias_size, small_weights_out)
            elif out_size <=  8*8*256:
                w,b = self.handle_medium_weights(out_size, bias_size, x)
            else:
                w,b = self.handle_big_weights(out_size, bias_size, x)

            w = tf.reshape(w, [-1] + layers[i])
            self.matrices.append(w)
            if bias_size > 0:
                b = tf.reshape(b, [-1, layers[i][-1]])
                self.bsss.append(b)
        
    def handle_big_weights(self, out_size, bias_size, x):
        out_weights_prod = np.prod(x.get_shape().as_list()[1:])
        f_out_size = x.get_shape().as_list()[-1]
        i = 0
        b = x
        def output_layers(x, f_in, f_out_size, j):
            x = conv_layer(x, 3, f_in, f_out_size, 'huge_handle_layer' + str(j))
            x = bias(x, f_out_size, 'huge_handle_bias' + str(j))
            return x
        
        x = output_layers(x, f_out_size, f_out_size, i)
        iterations = int(out_size / out_weights_prod)
        print(iterations, out_size, out_weights_prod)
        for i in range(int(np.log2(iterations))):
            x = output_layers(x, f_out_size, f_out_size*2 ,i+1)
            f_out_size *= 2
        logits = tf.reshape(x, [-1, out_size])

        if bias_size <= 0:
            return logits, None

        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_huge_layer_biases' + str(0), strides=(2,2))
        b = bias(b, int(filters/2), 'last_huge_layer_biases_biases' + str(0))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        
        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_huge_layer_biases' + str(1))
        b = bias(b, int(filters/2), 'last_huge_layer_biases_biases' + str(1))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        b = tf.reshape(b, [-1, np.prod(b.get_shape().as_list()[1:])])
        out_weights_prod = b.get_shape().as_list()[-1]
        
        with tf.name_scope('last_huge_layer_biases'):
            weights_fc_for_bias = default_tf_variable([out_weights_prod, bias_size], 'weights_fc_huge_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_huhge_for_bias')
        fc1_for_bias = tf.add(tf.matmul(b, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias) # maybe remove?
        biases = fc1_for_bias

        return logits, biases
    
    def handle_medium_weights(self, out_size, bias_size, x):
        out_weights_prod = np.prod(x.get_shape().as_list()[1:])
        print('out_weights_prod', out_weights_prod)
        print('out_size', out_size)
        f_out_size = x.get_shape().as_list()[-1]
        i = 0
        b = x
        def output_layers(x, f_in, f_out_size, j):
            x = conv_layer(x, 3, f_in, f_out_size, 'medium_handle_layer' + str(j))
            x = bias(x, f_out_size, 'medium_handle_bias' + str(j))
#             x = tf.layers.batch_normalization(x)
#             x = tf.nn.relu(x)
            return x
        
        x = output_layers(x, f_out_size, f_out_size, i)
        iterations = int(out_weights_prod / out_size)
        print('iterations', iterations)
        for i in range(int(np.log2(iterations))):
            x = output_layers(x, f_out_size, int(f_out_size/2) ,i+1)
            f_out_size = int(f_out_size/2)
            
        logits = tf.reshape(x, [-1, out_size])

        if bias_size <= 0:
            return logits, None

        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_medium_layer_biases' + str(0), strides=(2,2))
        b = bias(b, int(filters/2), 'last_medium_layer_biases_biases' + str(0))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        
        filters = b.get_shape().as_list()[-1]
        b = conv_layer(b, 3, filters, int(filters/2), 'last_medium_layer_biases' + str(1))
        b = bias(b, int(filters/2), 'last_medium_layer_biases_biases' + str(1))
        b = tf.layers.batch_normalization(b)
        b = tf.nn.relu(b)
        b = tf.reshape(b, [-1, np.prod(b.get_shape().as_list()[1:])])
        out_weights_prod = b.get_shape().as_list()[-1]
        
        
        with tf.name_scope('last_medium_layer_biases'):
            weights_fc_for_bias = default_tf_variable([out_weights_prod, bias_size], 'weights_fc_medium_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_medium_for_bias')
        fc1_for_bias = tf.add(tf.matmul(b, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias) # maybe remove?
        biases = fc1_for_bias

        return logits, biases
    
    def handle_small_weights(self, out_size, bias_size, small_weights_out):
        out_weights_prod = np.prod(small_weights_out.get_shape().as_list()[1:])
        with tf.name_scope('fc_layer_weights'):
            weights_fc_for_weights = default_tf_variable([out_weights_prod, out_size], 'weights_fc_for_weights')
            bias_fc_for_weights = default_tf_variable([out_size], 'bias_fc_for_weights')
        fc = tf.add(tf.matmul(small_weights_out, weights_fc_for_weights), bias_fc_for_weights)
        fc = tf.layers.batch_normalization(fc)
        logits = fc
        logits = tf.reshape(logits, [-1, out_size])

        if bias_size <= 0:
            return logits, None

        with tf.name_scope('fc_layer_biases'):
            weights_fc_for_bias = default_tf_variable([out_weights_prod, bias_size], 'weights_fc_for_bias')
            bias_fc_for_bias = default_tf_variable([bias_size], 'bias_fc_for_bias')
        fc1_for_bias = tf.add(tf.matmul(small_weights_out, weights_fc_for_bias), bias_fc_for_bias)
        fc1_for_bias = tf.layers.batch_normalization(fc1_for_bias) # maybe remove?
        biases = fc1_for_bias

        return logits, biases