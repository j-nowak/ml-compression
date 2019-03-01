import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import io
import keras

from glob import glob
from PIL import Image
from utils import *
from tqdm import tqdm
from hyper_network import HyperNetwork
from skimage.measure import compare_psnr

class HyperPictureFramework:
    
    def __init__(self, hparams, data_generator, model_name, saved_models_dir):
        self.hparams = hparams

        self.model_name = model_name
        self.saved_models_dir = saved_models_dir

        self.data_generator = data_generator
        
        self.__datasets_inputs()
        self.build_network()
        self.__build_stats()

        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.hparams.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.distortion_loss = tf.losses.mean_squared_error(self.Y, self.aec_network.decoded)
        # self.distortion_loss_quantized = tf.losses.mean_squared_error(self.Y, self.aec_network.quant_decoded)
        self.distortion_loss_cont = tf.losses.mean_squared_error(self.Y, self.aec_network.cont_decoded)
        
        # self.loss_op = self.distortion_loss + self.distortion_loss_quantized
        self.loss_op = self.distortion_loss + self.distortion_loss_cont

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this trains batch normalziation
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss_op)

        self.PSNR_train = tf.reduce_mean(tf.image.psnr(self.Y, self.aec_network.decoded, max_val=1.0))
        self.SSIM_train = tf.reduce_mean(tf.image.ssim(self.Y, self.aec_network.decoded, max_val=1.0))
        
        self.PSNR_train_quant = tf.reduce_mean(tf.image.psnr(self.Y, self.aec_network.quant_decoded, max_val=1.0))
        self.PSNR_train_cont = tf.reduce_mean(tf.image.psnr(self.Y, self.aec_network.cont_decoded, max_val=1.0))

        tf.summary.scalar('total_loss', self.loss_op)
        # tf.summary.scalar('vgg_loss', self.vgg_loss)
        tf.summary.scalar('distortion_loss', self.distortion_loss)
        # tf.summary.scalar('distortion_loss_quantized', self.distortion_loss_quantized)
        tf.summary.scalar('distortion_loss_cont', self.distortion_loss_cont)

        tf.summary.scalar('PSNR_train', self.PSNR_train)
        tf.summary.scalar('SSIM_train', self.SSIM_train)

        tf.summary.scalar('PSNR_train_quant', self.PSNR_train_quant)
        tf.summary.scalar('PSNR_train_cont', self.PSNR_train_cont)
        tf.summary.scalar('alpha', self.alpha)
        # tf.summary.scalar('SSIM_quant', self.SSIM_quant)

        self.merged = tf.summary.merge_all()

        self.__initialize_tf_session()

    def __initialize_tf_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
    
    def save(self, step_num):
        self.saver.save(self.sess, self.saved_models_dir, step_num)
        
    def restore(self, checkpoint, metagraph):
        new_saver = tf.train.import_meta_graph(metagraph)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint))

    def add_vggnet(self):
        def vgg_features(tf_image_input):
            vgg_input = tf.reshape(tf_image_input, [-1, 64, 64, 3]) - 0.5
            vgg_input = keras.layers.Input(batch_shape=(16, 64, 64, 3), tensor=vgg_input)        
            vgg_model = keras.applications.vgg16.VGG16(
                include_top=False, 
                weights='imagenet',
                input_tensor=vgg_input,
                input_shape=(self.hparams.in_img_width, self.hparams.in_img_height, self.hparams.channels), 
                pooling=None)
            vgg_model.layers.pop()
            vgg_features = vgg_model.layers[-1].output
            return vgg_features

        truth_features = vgg_features(self.Y)
        aec_features = vgg_features(self.target_network.logits)
        self.vgg_loss = self.hparams.vgg_loss_lambda * tf.losses.mean_squared_error(truth_features, aec_features)

    def __build_stats(self):
        self.truth_img = tf.placeholder(tf.float32, shape=(None, None, 3))
        self.result_img = tf.placeholder(tf.float32, shape=(None, None, 3))

        truth_yuv = tf.image.rgb_to_yuv(self.truth_img)
        result_yuv = tf.image.rgb_to_yuv(self.result_img)
        
        self.test_ms_ssim_yuv = tf.image.ssim_multiscale(truth_yuv, result_yuv, 1.0)
        self.test_psnr_yuv = tf.image.psnr(truth_yuv, result_yuv, 1.0)
        self.test_ssim_yuv = tf.image.ssim(truth_yuv, result_yuv, 1.0)

        self.test_ms_ssim_rgb = tf.image.ssim_multiscale(self.truth_img, self.result_img, 1.0)
        self.test_psnr_rgb = tf.image.psnr(self.truth_img, self.result_img, 1.0)
        self.test_ssim_rgb = tf.image.ssim(self.truth_img, self.result_img, 1.0)
        
    def build_network(self):
        self.alpha = tf.placeholder(tf.float32, shape=())
        self.aec_network = HyperNetwork(self.X, self.hparams, self.alpha)

    def __datasets_inputs(self):
        input_dataset = self.data_generator.train_dataset
        self.handle = tf.placeholder(tf.string, shape = [])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, input_dataset.output_types, input_dataset.output_shapes)
        el = self.iterator.get_next()
        in_img_tensor, out_img_tensor = el[0], el[1]

        in_width = self.hparams.in_img_width
        in_height = self.hparams.in_img_width
        in_depth = self.hparams.channels

        self.X = tf.reshape(in_img_tensor, shape=[-1, in_width, in_height, in_depth])
        self.Y = tf.reshape(out_img_tensor, shape=[-1, in_width, in_height, in_depth])
        
    def run_test_mode(self, sess, step_num, test_writer):
        test_images_paths = glob(self.hparams.test_dataset_path + '/*')
        ms_ssim_yuv, psnr_yuv, ssim_yuv, ms_ssim_rgb, psnr_rgb, ssim_rgb = run_tests(test_images_paths, self, self.hparams.in_img_width, self.hparams.in_img_width - 1)

        test_summary = tf.Summary(value=[
            tf.Summary.Value(tag='PSNR_YUV', simple_value=psnr_yuv),
            tf.Summary.Value(tag='SSIM_YUV', simple_value=ssim_yuv),
            tf.Summary.Value(tag='MS_SSIM_YUV', simple_value=ms_ssim_yuv),

            tf.Summary.Value(tag='PSNR_RGB', simple_value=psnr_rgb),
            tf.Summary.Value(tag='SSIM_RGB', simple_value=ssim_rgb),
            tf.Summary.Value(tag='MS_SSIM_RGB', simple_value=ms_ssim_rgb),
        ])

        test_writer.add_summary(test_summary, step_num)
        test_writer.flush()

    def copute_alpha_param(self, step_num, max_val=50, when_mid=100000):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        return max_val * sigmoid(step_num + when_mid)
    
    def train(self, sess, train_writer, test_writer):
        train_dataset = self.data_generator.train_dataset
        train_iterator = train_dataset.make_one_shot_iterator()

        self.sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())

        def run_train(step_num):
            tensors = [self.merged, self.loss_op, self.train_op]

            # alpha_param = (step_num // 1000) + 1
            alpha_param = self.copute_alpha_param(step_num)

            fd = {self.handle: train_handle, self.alpha: alpha_param}

            summary, _, _ = sess.run(tensors, feed_dict=fd)
            train_writer.add_summary(summary, step_num)

            if (step_num % self.hparams.test_per_iterations == 0) and step_num > 0:
                self.save(step_num)
                self.run_test_mode(sess, step_num, test_writer)
        
        if self.hparams.steps <= 0:
            i = 0
            while True:
                run_train(i)
                i += 1
        else:
            for i in tqdm(range(self.hparams.steps)):
                run_train(i)