import tensorflow as tf
import numpy as np

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

def encode_8x8x16(x):
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
    return tf.layers.conv2d(inputs=x, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')

def decode_8x8x16(encoded, channels_num=3):
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
    decoded = tf.layers.conv2d(inputs=x, filters=channels_num, kernel_size=(5, 5), strides=(1, 1), padding='same')
    # decoded = tf.nn.sigmoid(decoded)
    return decoded

def hard_round(x, quant_size):
    rounded = tf.stop_gradient(tf.round(x))
    return rounded / quant_size

# --- QUANTIZATION ---
def quant_dequant(x, quant_func, quant_size, qparams):
    quant = quant_func(x, quant_size, qparams)
    dequant = quant / quant_size
    return quant, dequant

@tf.custom_gradient
def __round_with_sin_grad(x):
    def grad(dy):
        return dy * (1 - 0.5 * tf.cos(2 * np.pi * x))
    return tf.round(x), grad

@tf.custom_gradient
def __round_with_id_grad(x):
    def grad(dy):
        return dy
    return tf.round(x), grad

def gradient_override_quant(x, quant_size, qparams):
    quant = x
    quant = tf.nn.sigmoid(quant)
    quant = quant * quant_size

    if qparams['override_func'] == 'sinus':
        quant = __round_with_sin_grad(quant)
    elif qparams['override_func'] == 'id':
        quant = __round_with_id_grad(quant)
    else:
        raise Exception("Unknown override func: " + qparams.override_func)

    return quant

# ---- STAIRCASE POWER QUANT ----

@tf.custom_gradient
def __floor_with_id_grad(x):
    def grad(dy):
        return dy
    return tf.floor(x), grad

def __left_half(in_x, n):
    return tf.pow(2.0 * in_x, n) / 2.0
def __right_half(in_x, n):
    return -tf.pow(2.0 * (1 - in_x), n) / 2.0 + 1.0

def __staircase(x, alpha):
    return tf.where(tf.less(x, 0.5), __left_half(x, alpha), __right_half(x, alpha))

def staircase_power_quant(x, quant_size, qparams):
    x = tf.math.maximum(0.0, x) 
    x = tf.math.minimum(quant_size, x)
    
    floor_x = __floor_with_id_grad(x)
    mantis = x - floor_x        
    stair = __staircase(mantis, qparams['alpha'])

    return floor_x + stair

# --- STAIRCASE SINUS QUANT ----

def staircase_sinus_quant(x, quant_size, qparams):
    res = x - qparams['alpha'] * (tf.math.sin(2 * np.pi * x) / (2 * np.pi))
    res = tf.math.maximum(0.0, res)
    return tf.math.minimum(quant_size, res)