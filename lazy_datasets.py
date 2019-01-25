from glob import glob
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
import scipy.misc
import numpy as np
import random

from keras.datasets import cifar10

class CifarGenerator:
    def __init__(self):
        self.__prepare_datasets()

    def __normalize_colors(self, x):
        return x / 255.0

    def __generator(self, all_imgs):
        p = np.random.permutation(len(all_imgs))
        imgs = all_imgs[p]
        for img in imgs:
            yield img

    def __prepare_datasets(self):
        print('Initializing dataset...')
        (x_train, _), (x_test, _) = cifar10.load_data()

        self.x_train = self.__normalize_colors(x_train)
        self.x_test = self.__normalize_colors(x_test)

    def train_data_generator(self):
        return self.__generator(self.x_train)

    def test_data_generator(self):
        return self.__generator(self.x_test)

    def get_image(self, num):
        return self.x_train[num]

def img_to_np(img):
    np_img = np.array(img)
    return normalize_colors(np_img)

def normalize_colors(x):
    return x / 255.0

def divide_image_in_areas(img, crop_size, step):
    all_areas = []
    width, height = img.size
    for start_top in range(0, height, step):
        bottom = min(start_top + crop_size, height)
        top = bottom - crop_size
        for start_left in range(0, width, step):
            right = min(start_left + crop_size, width)
            left = right - crop_size
            all_areas.append((left, top, right, bottom))
    return all_areas

class HeavyImageGenerator:
    def __init__(self, path, crop_size, batch_size, to_yuv):
        self.img_list = glob(path + '/*')
        self.crop_size = crop_size
        self.crops_num = batch_size
        self.to_yuv = to_yuv

        val_size = int(len(self.img_list) * 0.1)
        self.x_train = self.img_list[val_size:]
        self.x_test = self.img_list[:val_size]

    def __read_img(self, img_path, crop_size, crops_num):
        img = Image.open(img_path)

        if (self.to_yuv):
            img = img.convert('YCbCr')
        
        def get_random_crop():        
            width, height = img.size
            left = np.random.randint(width - crop_size)
            upper = np.random.randint(height - crop_size)
            right = left + crop_size
            lower = upper + crop_size
            return img_to_np(img.crop((left, upper, right, lower)))

        return divide_image_in_areas(img, self.crop_size, self.crop_size - 1)
        # return [get_random_crop() for i in range(crops_num)]

    def __generator(self, all_imgs):
        img_paths = all_imgs[:]
        random.shuffle(img_paths)
        for img_path in img_paths:
            crops = self.__read_img(img_path, self.crop_size, self.crops_num)
            for i in range(len(crops)):
                yield crops[i]

    def train_data_generator(self):
        return self.__generator(self.x_train)

    def test_data_generator(self):
        return self.__generator(self.x_test)

    def get_image(self, num):
        return self.__read_img(self.img_list[num], self.crop_size, 1)[0]

def add_noise(img, noise_level, mean=0.0):
    std = noise_level / 255.0
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img = np.clip(noisy_img * 255, 0, 255)
    return noisy_img / 255.0

def crop_img(img, crops):
    left, top, right, bottom = crops
    img_crop = img.crop((left, top, right, bottom))
    return img_to_np(img_crop)

def get_random_crop(img_size, crop_size):   
    width, height = img_size     
    left = np.random.randint(width - crop_size)
    upper = np.random.randint(height - crop_size)
    right = left + crop_size
    lower = upper + crop_size
    return (left, upper, right, lower)

class NoisyImageGenerator:
    def __init__(self, path, crop_size, batch_size, noise_level, in_func):
        self.img_list = glob(path + '/*')
        self.crop_size = crop_size
        self.crops_num = batch_size
        self.noise_level = noise_level
        self.in_func = in_func

    def __generator(self, all_imgs):
        img_paths = all_imgs[:]
        random.shuffle(img_paths)
        for img_path in img_paths:
            target_img = Image.open(img_path)
            
            for i in range(self.crops_num):
                crop = get_random_crop(target_img.size, self.crop_size)
                target_crop = crop_img(target_img, crop)

                if self.in_func == 'noise':
                    noisy_crop = add_noise(target_crop, self.noise_level)
                else:
                    noisy_crop = target_crop

                yield noisy_crop, target_crop

    def train_data_generator(self):
        return self.__generator(self.img_list)

class Dataset:
    def __init__(self, hparams):
        # if hparams.in_func == 'noise':
        self.data_generator = NoisyImageGenerator(hparams.train_dataset_path, hparams.in_img_width, hparams.batch_size, hparams.noise_level, hparams.in_func)
        # else:
        #     self.data_generator = HeavyImageGenerator(hparams.train_dataset_path, hparams.in_img_width, hparams.batch_size, hparams.noise_level)

        self.train_dataset = self.__prepare_train_dataset(hparams)

    def __prepare_train_dataset(self, hparams):
        return tf.data.Dataset() \
            .from_generator(self.data_generator.train_data_generator, output_types=(tf.float32, tf.float32)) \
            .batch(hparams.batch_size, drop_remainder=True) \
            .prefetch(hparams.queue_capacity) \
            .repeat()

    def get_image(self, num):
        return self.data_generator.get_image(num)

    def test_size(self):
        return len(self.data_generator.x_test)