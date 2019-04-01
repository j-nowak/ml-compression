import tensorflow as tf
import numpy as np
import scipy
import math
import io
import keras

from glob import glob
from test_utils import *
from tqdm import tqdm

from networks.models import *

class HyperPictureFramework:
    
    def __init__(self, hparams, data_generator, model_name, saved_models_dir, image_test_func, power_factors=False):
        self.hparams = hparams

        self.model_name = model_name
        self.saved_models_dir = saved_models_dir

        self.image_test_func = image_test_func

        self.data_generator = data_generator
        
        self.prepare_inputs()
        self.build_stats(power_factors)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.hparams.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.build_network()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this trains batch normalziation
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss_op)

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

    def build_stats(self, override_power_factors):
        self.truth_img = tf.placeholder(tf.float32, shape=(None, None, 3))
        self.result_img = tf.placeholder(tf.float32, shape=(None, None, 3))

        if override_power_factors:
            self.test_ms_ssim_rgb = tf.image.ssim_multiscale(self.truth_img, self.result_img, 1.0, power_factors=(0.0448, 0.2856, 0.3001))
        else:
            self.test_ms_ssim_rgb = tf.image.ssim_multiscale(self.truth_img, self.result_img, 1.0)
        
        self.test_psnr_rgb = tf.image.psnr(self.truth_img, self.result_img, 1.0)
        self.test_ssim_rgb = tf.image.ssim(self.truth_img, self.result_img, 1.0)
        
    def build_network(self):
        self.step_num = tf.placeholder(tf.float32, shape=())

        if self.hparams.model_type == 'sin_grad':
            total_loss, encoded, decoded = baseline_model_SINUS_GRAD(self.X, self.Y, self.hparams)
            quant_decoded = decoded
        elif self.hparams.model_type == 'id_grad':
            total_loss, encoded, decoded = baseline_model_ID_GRAD(self.X, self.Y, self.hparams)
            quant_decoded = decoded
        elif self.hparams.model_type == 'no_quant':
            total_loss, encoded, decoded = baseline_model_NO_QUANT(self.X, self.Y)
            quant_decoded = decoded
        elif self.hparams.model_type == 'bin_8x8x16_cont':
            total_loss, encoded, decoded, quant_decoded = binary_8x8x16_continous(self.X, self.Y, self.step_num, self.hparams)
        elif self.hparams.model_type == 'binary_16x16x4_continous':
            total_loss, encoded, decoded, quant_decoded = binary_16x16x4_continous(self.X, self.Y, self.step_num, self.hparams)
        elif self.hparams.model_type == 'cont_quant_sinus':
            total_loss, encoded, decoded, quant_decoded = cont_quant_sinus(self.X, self.Y, self.step_num, self.hparams)
        elif self.hparams.model_type == 'cont_quant_power':
            total_loss, encoded, decoded, quant_decoded = cont_quant_power(self.X, self.Y, self.step_num, self.hparams)
        else:
            raise Exception('Unkonw model type: ' + self.hparams.model_type)

        self.loss_op = total_loss
        self.encoded = encoded
        self.decoded = decoded
        self.quant_decoded = quant_decoded

        tf.summary.scalar('total_loss', self.loss_op)

    def prepare_inputs(self):
        input_dataset = self.data_generator.train_dataset
        self.handle = tf.placeholder(tf.string, shape = [])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, input_dataset.output_types, input_dataset.output_shapes)
        in_img_tensor = self.iterator.get_next()

        in_width = self.hparams.in_img_width
        in_height = self.hparams.in_img_width
        in_depth = self.hparams.channels

        self.X = tf.reshape(in_img_tensor, shape=[-1, in_width, in_height, in_depth])
        self.Y = tf.reshape(in_img_tensor, shape=[-1, in_width, in_height, in_depth])
        
    def run_test_mode(self, sess, step_num, test_writer):
        test_images_paths = glob(self.hparams.test_dataset_path + '/*')
        ms_ssim_rgb, psnr_rgb, ssim_rgb = run_tests(test_images_paths, self, self.hparams.in_img_width, self.hparams.in_img_width - 1)

        test_summary = tf.Summary(value=[
            tf.Summary.Value(tag='PSNR_RGB', simple_value=psnr_rgb),
            tf.Summary.Value(tag='SSIM_RGB', simple_value=ssim_rgb),
            tf.Summary.Value(tag='MS_SSIM_RGB', simple_value=ms_ssim_rgb),
        ])

        test_writer.add_summary(test_summary, step_num)
        test_writer.flush()
    
    def train(self, sess, train_writer, test_writer):
        train_dataset = self.data_generator.train_dataset
        train_iterator = train_dataset.make_one_shot_iterator()

        self.sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())

        def run_train(step_num):
            tensors = [self.merged, self.loss_op, self.train_op]
            fd = {self.handle: train_handle, self.step_num: step_num}

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