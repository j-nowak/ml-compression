import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL

from PIL import Image
from tensorflow.contrib.training import HParams
from tqdm import tqdm

class CompressNetwork:
    
    def __fc_layers(self, x, weights_t, biases_t, batch_trainables):
        with tf.variable_scope('compress_network'):
            for i in range(len(weights_t)):
                winit, wtrian = weights_t[i]
                binit, btrian = biases_t[i]
                
                with tf.variable_scope('layer_{}'.format(i)):
                    W = tf.squeeze(tf.get_variable(
                        'weights', 
                        initializer=tf.constant(winit),
                        trainable=wtrian))
                    
                    b = tf.squeeze(tf.get_variable(
                        'bias', 
                        initializer=tf.constant(binit),
                        trainable=btrian))
                    
                    self.weigths.append(W)
                    self.biases.append(b)
                    
                    x = tf.matmul(x, W) + b
                    if (i < len(weights_t) - 1):
                        x = tf.cos(x)
                        x = tf.layers.batch_normalization(x, trainable=batch_trainables[i])
        return tf.nn.sigmoid(x)

    def for_man(self, hparams, weights_t, biases_t, batch_trainables, out_img):
        self.weigths = []
        self.biases = []
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            global_step = tf.Variable(0, trainable=False)
            self.pixels = tf.placeholder(tf.float32, shape=(None, 2))
            
            self.logits = self.__fc_layers(self.pixels, weights_t, biases_t, batch_trainables)
            self.comp_img = tf.reshape(self.logits, [hparams.in_img_width, hparams.in_img_height, hparams.channels])

            self.loss_op = tf.losses.mean_squared_error(out_img, self.comp_img)
            self.optimizer = tf.train.AdamOptimizer()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this trains batch normalziation
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss_op)
        return self

    def __convert_const_ws(self, ws_size, batch_size, i):
        a, b = ws_size
        with tf.variable_scope('layer_{}'.format(i)):
            return tf.Variable(tf.random_uniform([a, b], minval=-0.5, maxval=0.5), name='common_weights')

    def __get_bias(self, bs_size, i):
        with tf.variable_scope('layer_{}'.format(i)):
            return tf.Variable(tf.random_uniform([bs_size], minval=-0.5, maxval=0.5), name='bias')

    def __compute_size(self, tvars, hparams):
        def compute_shape_size(shape):
            return np.prod([x for x in shape if x is not None])
        self.vars_count = np.sum([compute_shape_size(v.get_shape().as_list()) for v in tvars])
        self.bytes_count = self.vars_count * 32 # we use float32 as vars
        self.bpp = self.bytes_count / (hparams.in_img_width * hparams.in_img_height)
        print('NET BPP: ', self.bpp)

    def for_hyper(self, hparams, pixels, trainable_ws, const_ws, trainables_mask):
        self.__compute_size(trainable_ws, hparams)

        x = pixels
        trainables_index, const_index = 0, 0
        
        with tf.variable_scope('compress_network'):
            for i in range(0, len(trainables_mask)):
                if trainables_mask[i]:
                    W = trainable_ws[trainables_index]
                    trainables_index += 1
                else:
                    W = self.__convert_const_ws(const_ws[const_index], hparams.batch_size, i)
                    const_index += 1
                b = self.__get_bias(W.get_shape().as_list()[-1], i)

                if (len(x.get_shape()) == 3):
                    x = tf.matmul(x, W) + tf.expand_dims(b, 0)
                elif (len(W.get_shape()) == 3):
                    x = tf.map_fn(lambda u: tf.matmul(x, u) + b, W)
                else:
                    x = tf.matmul(x, W) + b
                        
                if (i < len(trainables_mask) - 1):
                    x = tf.cos(x)
                    x = tf.layers.batch_normalization(x, trainable=False)

            x = tf.nn.sigmoid(x)

        self.logits = x
        self.comp_img = tf.reshape(self.logits, [-1, hparams.in_img_width, hparams.in_img_height, hparams.channels])

        return self

    def __init__(self):
        pass


def raw_compress_net(hparams, layers, img, steps=30000):
    init_weights = []
    init_biases = []
    init_batch_trains = []

    for i in range(1, len(layers)):
        w = np.random.uniform(-0.5, 0.5, size=(layers[i - 1], layers[i])).astype('float32')
        b = np.random.uniform(-1, 1, layers[i]).astype('float32')
        
        init_weights.append((w, True))
        init_biases.append((b, True))
        init_batch_trains.append(True)

    comp_net = CompressNetwork().for_man(hparams, init_weights, init_biases, init_batch_trains, img)

    with tf.Session(graph = comp_net.graph) as sess:
        sess.run(tf.global_variables_initializer())

        pixels = [[j, i] for i in range(hparams.in_img_height) for j in range(hparams.in_img_width)]
        fd = {comp_net.pixels: pixels}

        def overfit_compress(step_num):
            _ = sess.run(comp_net.train_op, feed_dict=fd)

        for i in tqdm(range(steps)):
            overfit_compress(i)
            
        final_img, ws, bs = sess.run([comp_net.comp_img, comp_net.weigths, comp_net.biases], feed_dict=fd)

    return comp_net, ws, bs, final_img

class TargetNetwork:
    def __fc_layers(self, x, weights, biases):
        mul = lambda u: tf.layers.batch_normalization(tf.cos(tf.matmul(x, u[0]) + u[1]), trainable=False)
        x = tf.map_fn(mul, (weights[0], biases[0]), dtype=tf.float32)
        
        with tf.name_scope('target_network'):
            for i in range(1, len(weights)):
                with tf.name_scope('weights_{}'.format(i)):                        
                    if(i==2):
                        self.later_x = x
                        
                    x = tf.matmul(x, weights[i]) + tf.expand_dims(biases[i],1)
                    
                    if(i < len(weights) - 1):
                        x = tf.cos(x)
                        x = tf.layers.batch_normalization(x, trainable=False)
        return x
    
    def __logits(self, x):
        self.logits = tf.nn.sigmoid(x)

    def __init__(self, hparams, pixels, weights, biases):
        self.hparams = hparams
        x = self.__fc_layers(pixels, weights, biases)
        self.__logits(x)
        