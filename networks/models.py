import tensorflow as tf
import numpy as np

from .building_blocks import *

def baseline_model_NO_QUANT(x, target):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_8x8x16(x)
    encode = tf.nn.sigmoid(encoded)
    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_8x8x16(encoded)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    tf.summary.scalar('PSNR_train', PSNR_train)

    return distortion_loss, encoded, decoded

def baseline_model_SINUS_GRAD(x, target, hparams):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_8x8x16(x)

    qparams = { 'override_func': 'sinus' }
    _, dequantized = quant_dequant(
        encoded, 
        gradient_override_quant, 
        hparams.quant_size, 
        qparams)

    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_8x8x16(dequantized)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    tf.summary.scalar('PSNR_train', PSNR_train)

    return distortion_loss, encoded, decoded

def baseline_model_ID_GRAD(x, target, hparams):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_8x8x16(x)

    qparams = { 'override_func': 'id' }
    _, dequantized = quant_dequant(
        encoded, 
        gradient_override_quant, 
        hparams.quant_size, 
        qparams)

    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_8x8x16(dequantized)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    tf.summary.scalar('PSNR_train', PSNR_train)

    return distortion_loss, encoded, decoded

def __compute_alpha(step_num, max_alpha, alpha_div):
        return 1 + tf.math.minimum(max_alpha, step_num / alpha_div)

def binary_8x8x16_continous(x, target, step_num, hparams):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_8x8x16(x)

    alpha = __compute_alpha(step_num, hparams.max_alpha, hparams.alpha_div)
    cont_encoded = tf.nn.sigmoid(alpha * encoded)
    
    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_8x8x16(cont_encoded)

    quantized = tf.stop_gradient(tf.round(cont_encoded))
    with tf.variable_scope("DECODER", reuse=True) as scope:
        quant_decoded = decode_8x8x16(quantized)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)
    distortion_loss_quantized = tf.losses.mean_squared_error(target, quant_decoded)
    total_loss = distortion_loss + distortion_loss_quantized

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    PSNR_train_quant = tf.reduce_mean(tf.image.psnr(target, quant_decoded, max_val=1.0))

    tf.summary.scalar('alpha', alpha)

    tf.summary.scalar('PSNR_train', PSNR_train)
    tf.summary.scalar('PSNR_train_quant', PSNR_train_quant)

    tf.summary.scalar('distortion_loss_quantized', distortion_loss_quantized)

    tf.summary.histogram('cont_encoded', tf.reshape(cont_encoded, [-1]))

    return total_loss, cont_encoded, decoded, quant_decoded

def binary_16x16x4_continous(x, target, step_num, hparams):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_16x16x4(x)

    alpha = __compute_alpha(step_num, hparams.max_alpha, hparams.alpha_div)
    cont_encoded = tf.nn.sigmoid(alpha * encoded)
    
    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_16x16x4(cont_encoded)

    quantized = tf.stop_gradient(tf.round(cont_encoded))
    with tf.variable_scope("DECODER", reuse=True) as scope:
        quant_decoded = decode_16x16x4(quantized)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)
    distortion_loss_quantized = tf.losses.mean_squared_error(target, quant_decoded)
    total_loss = distortion_loss + distortion_loss_quantized

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    PSNR_train_quant = tf.reduce_mean(tf.image.psnr(target, quant_decoded, max_val=1.0))

    tf.summary.scalar('alpha', alpha)

    tf.summary.scalar('PSNR_train', PSNR_train)
    tf.summary.scalar('PSNR_train_quant', PSNR_train_quant)

    tf.summary.scalar('distortion_loss_quantized', distortion_loss_quantized)

    tf.summary.histogram('cont_encoded', tf.reshape(cont_encoded, [-1]))

    return total_loss, cont_encoded, decoded, quant_decoded

def __compute_alpha_sinus(step_num, max_alpha, alpha_div):
    return tf.math.maximum(max_alpha, step_num / alpha_div)

def cont_quant_sinus(x, target, step_num, hparams):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_8x8x16(x)
    
    alpha = __compute_alpha_sinus(step_num, hparams.max_alpha, hparams.alpha_div)
    qparams = { 'alpha': alpha }
    cont_encoded, dequantized = quant_dequant(
        encoded, 
        staircase_sinus_quant, 
        hparams.quant_size, 
        qparams)
    
    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_8x8x16(dequantized)

    hard_quantized = hard_round(cont_encoded, hparams.quant_size)
    with tf.variable_scope("DECODER", reuse=True) as scope:
        quant_decoded = decode_8x8x16(hard_quantized)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)
    distortion_loss_quantized = tf.losses.mean_squared_error(target, quant_decoded)
    total_loss = distortion_loss + distortion_loss_quantized

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    PSNR_train_quant = tf.reduce_mean(tf.image.psnr(target, quant_decoded, max_val=1.0))
    
    tf.summary.scalar('alpha', alpha)

    tf.summary.scalar('PSNR_train', PSNR_train)
    tf.summary.scalar('PSNR_train_quant', PSNR_train_quant)

    tf.summary.scalar('distortion_loss_quantized', distortion_loss_quantized)

    tf.summary.histogram('cont_encoded', tf.reshape(cont_encoded, [-1]))

    return total_loss, cont_encoded, decoded, quant_decoded

def cont_quant_power(x, target, step_num, hparams):
    with tf.variable_scope("ENCODER", reuse=False) as scope:
        encoded = encode_8x8x16(x)
    
    alpha = __compute_alpha(step_num, hparams.max_alpha, hparams.alpha_div)
    qparams = { 'alpha': alpha }
    cont_encoded, dequantized = quant_dequant(
        encoded, 
        staircase_power_quant, 
        hparams.quant_size, 
        qparams)
    
    with tf.variable_scope("DECODER", reuse=False) as scope:
        decoded = decode_8x8x16(dequantized)

    hard_quantized = hard_round(cont_encoded, hparams.quant_size)
    with tf.variable_scope("DECODER", reuse=True) as scope:
        quant_decoded = decode_8x8x16(hard_quantized)

    distortion_loss = tf.losses.mean_squared_error(target, decoded)
    distortion_loss_quantized = tf.losses.mean_squared_error(target, quant_decoded)
    total_loss = distortion_loss + distortion_loss_quantized

    tf.summary.scalar('alpha', alpha)

    PSNR_train = tf.reduce_mean(tf.image.psnr(target, decoded, max_val=1.0))
    PSNR_train_quant = tf.reduce_mean(tf.image.psnr(target, quant_decoded, max_val=1.0))

    tf.summary.scalar('PSNR_train', PSNR_train)
    tf.summary.scalar('PSNR_train_quant', PSNR_train_quant)

    tf.summary.scalar('distortion_loss_quantized', distortion_loss_quantized)

    tf.summary.histogram('cont_encoded', tf.reshape(cont_encoded, [-1]))

    return total_loss, cont_encoded, decoded, quant_decoded