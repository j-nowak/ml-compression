from glob import glob
from tqdm import tqdm
from PIL import Image
import PIL

import tensorflow as tf
import scipy.misc
import numpy as np
import random

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

class BigImageGenerator:
    def __init__(self, path, crop_size, batch_size):
        self.img_list = glob(path + '/*')
        self.crop_size = crop_size
        self.crops_num = batch_size

    def __generator(self, all_imgs):
        img_paths = all_imgs[:]
        random.shuffle(img_paths)
        for img_path in img_paths:
            target_img = Image.open(img_path)
            
            for i in range(self.crops_num):
                crop = get_random_crop(target_img.size, self.crop_size)
                target_crop = crop_img(target_img, crop)
                yield target_crop

    def train_data_generator(self):
        return self.__generator(self.img_list)

class Dataset:
    def __init__(self, hparams):
        self.data_generator = BigImageGenerator(hparams.train_dataset_path, 64, hparams.batch_size)
        self.train_dataset = self.__prepare_train_dataset(hparams)

    def __prepare_train_dataset(self, hparams):
        return tf.data.Dataset() \
            .from_generator(self.data_generator.train_data_generator, output_types=(tf.float32)) \
            .batch(hparams.batch_size, drop_remainder=True) \
            .prefetch(hparams.queue_capacity) \
            .repeat()

    def get_image(self, num):
        return self.data_generator.get_image(num)

    def test_size(self):
        return len(self.data_generator.x_test)

def celeba_crop(celeb_img, crop_size):
    crop_box = (25, 50, 25 + 2 * crop_size, 50 + 2 * crop_size)
    cropped = celeb_img.crop(crop_box)
    resized = cropped.resize((crop_size, crop_size), PIL.Image.NEAREST)
    return img_to_np(resized)

class CelebaImageGenerator:
    def __init__(self, path, crop_size):
        self.img_list = glob(path + '/*')
        self.crop_size = crop_size

    def __generator(self, all_imgs):
        img_paths = all_imgs[:]
        random.shuffle(img_paths)
        for img_path in img_paths:
            target_img = Image.open(img_path)
            prepared_img = celeba_crop(target_img, self.crop_size)
            yield prepared_img, prepared_img

    def train_data_generator(self):
        return self.__generator(self.img_list)

class CelebaDataset:
    def __init__(self, hparams):
        self.data_generator = CelebaImageGenerator(hparams.train_dataset_path, 64)
        self.train_dataset = self.__prepare_train_dataset(hparams)

    def __prepare_train_dataset(self, hparams):
        return tf.data.Dataset() \
            .from_generator(self.data_generator.train_data_generator, output_types=(tf.float32)) \
            .batch(hparams.batch_size, drop_remainder=True) \
            .prefetch(hparams.queue_capacity) \
            .repeat()

    def get_image(self, num):
        return self.data_generator.get_image(num)

    def test_size(self):
        return len(self.data_generator.x_test)