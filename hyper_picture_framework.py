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
    
    def __init__(self, hparams, data_generator, model_name, saved_models_dir, target_fn):
        self.hparams = hparams

        self.model_name = model_name
        self.saved_models_dir = saved_models_dir

        self.data_generator = data_generator
        
        self.target_fn = target_fn

        self.__datasets_inputs()
        self.__define_target_architecture()
        self.__build_stats()

        self.add_vggnet()
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.hparams.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   self.hparams.decay_steps, self.hparams.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.distortion_loss = tf.losses.mean_squared_error(self.Y, self.target_network.logits)

        self.loss_op = self.distortion_loss + self.vgg_loss
        # self.loss_op = tf.losses.mean_squared_error(self.Y, self.target_network.logits)# + 0.05 * self.latent_loss
        # self.loss_op = tf.losses.mean_squared_error(self.Y, self.target_network.logits) + tf.losses.mean_squared_error(self.Y, self.quant_target_network.logits)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this trains batch normalziation
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss_op)

        self.PSNR = tf.image.psnr(self.target_network.logits, self.Y, max_val=1.0)
        self.SSIM = tf.image.ssim(self.target_network.logits, self.Y, max_val=1.0)

        # self.PSNR_quant = tf.image.psnr(self.quant_target_network.logits, self.Y, max_val=1.0)
        # self.SSIM_quant = tf.image.ssim(self.quant_target_network.logits, self.Y, max_val=1.0)
        # self.MS_SSIM = tf.image.ssim_multiscale(self.target_network.logits, self.Y, max_val=1.0, power_factors=self.power_factors)

        tf.summary.scalar('total_loss', self.loss_op)
        tf.summary.scalar('vgg_loss', self.vgg_loss)
        tf.summary.scalar('distortion_loss', self.distortion_loss)

        tf.summary.scalar('PSNR', self.PSNR)
        tf.summary.scalar('SSIM', self.SSIM)

        # tf.summary.scalar('PSNR_quant', self.PSNR_quant)
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
        
    def __define_target_architecture(self):
        layers = self.hparams.target_layers

        hypernet = HyperNetwork(self.X, self.hparams, layers, True)
        self.hypernet = hypernet
        self.latent_loss = hypernet.latent_loss

        matrices = hypernet.matrices
        biases = hypernet.bsss
        params = {
            'hparams': self.hparams,
            'pixels': self.pixels,
            'weights': matrices,    
            'biases': biases
        }
        self.target_network = self.target_fn(**params)

        # quant_params = {
        #     'hparams': self.hparams,
        #     'pixels': self.pixels,
        #     'weights': hypernet.quant_matrices,    
        #     'biases': hypernet.quant_biases,
        # }
        # self.quant_target_network = self.target_fn(**quant_params)
        
    def __set_pixel_matrix(self):
        in_width = self.hparams.in_img_width
        in_height = self.hparams.in_img_width

        self.scale = 1.0
        
        pixels_x = tf.linspace(0.0, tf.cast(in_width - 1, tf.float32), tf.cast(in_width, tf.int32))
        pixels_y = (tf.linspace(0.0, tf.cast(in_height - 1, tf.float32), tf.cast(in_height, tf.int32))) 

        a, b = pixels_x[ None, :, None ], pixels_y[ :, None, None ]
        cartesian_product = tf.concat([a + tf.zeros_like(b), tf.zeros_like(a) + b], axis=2)

        self.pixels = tf.reshape(cartesian_product, shape=[tf.cast(in_width * in_height, tf.int32), 2])
        
    def __find_power_factors(self, img_size):
        powers = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        if (img_size < 64):
            return powers[:1]
        elif (img_size < 128):
            return powers[:2]
        else:
            return powers

    def __datasets_inputs(self):
        self.__set_pixel_matrix()

        self.power_factors = self.__find_power_factors(self.hparams.in_img_width)

        input_dataset = self.data_generator.train_dataset
        self.handle = tf.placeholder(tf.string, shape = [])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, input_dataset.output_types, input_dataset.output_shapes)
        el = self.iterator.get_next()
        in_img_tensor, out_img_tensor = el[0], el[1]

        in_width = self.hparams.in_img_width
        in_height = self.hparams.in_img_width
        in_depth = self.hparams.channels

        self.X = in_img_tensor
        self.Y = tf.reshape(out_img_tensor, shape=[-1, tf.cast(in_width * in_height, tf.int32), in_depth])
        
    def run_test_mode(self, sess, step_num, test_writer):
        # self.sess.run(validation_iterator.initializer)

        # ssims, psnrs, losses, ms_ssims = [], [], [], []
        # for i in tqdm(range(self.data_generator.test_size() // self.hparams.batch_size)):
        #     tensors = [self.PSNR, self.SSIM, self.loss_op]
        #     fd = {self.handle: validation_handle}
        #     psnr, ssim, loss = sess.run(tensors, feed_dict=fd)
        #     psnrs.append(psnr)
        #     ssims.append(ssim)
        #     losses.append(loss)
        #     # ms_ssims.append(ms_ssim)

        # test_psnr = np.mean(psnrs)
        # test_ssim = np.mean(ssims)
        # test_loss = np.mean(losses)
        # test_ms_ssim = np.mean(ms_ssims)

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
    
    def train(self, sess, train_writer, test_writer):
        train_dataset = self.data_generator.train_dataset
        train_iterator = train_dataset.make_one_shot_iterator()

        # validation_dataset = self.data_generator.test_dataset
        # validation_iterator = validation_dataset.make_initializable_iterator()

        self.sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())
        # validation_handle = sess.run(validation_iterator.string_handle())

        def run_train(step_num):
            tensors = [self.merged, self.target_network.logits, self.loss_op, self.train_op]
            fd = {self.handle: train_handle}

            summary, _, _, _ = sess.run(tensors, feed_dict=fd)
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