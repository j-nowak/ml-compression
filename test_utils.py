import tensorflow as tf
import numpy as np
import scipy
import math

from glob import glob
from PIL import Image
from datasets import *
from tqdm import tqdm

def run_through_net(model, img, crop_size):
    single_img = img.reshape(1, crop_size, crop_size, 3)
    fd = {model.X: single_img, model.step_num: model.hparams.alpha_div}
    out_img = model.sess.run([model.quant_decoded], feed_dict=fd)[0]
    return np.reshape(out_img[0], (crop_size, crop_size, 3))

def compute_weight_mask(img, all_crops):
    width, height = img.size
    res = np.zeros((height, width, 3))
        
    for crop in all_crops:
        left, top, right, bottom = crop
        res[top:bottom, left:right] += 1
        
    return res

def process_img(model, img, crop_size, step, batch_size):
    width, height = img.size
    res = np.zeros((height, width, 3))
    
    all_crops = divide_image_in_areas(img, crop_size, step)
    weight_mask = compute_weight_mask(img, all_crops)
    
    for crop_area in all_crops:
        input_crop = img_to_np(img.crop(crop_area))
        output_crop = run_through_net(model, input_crop, crop_size)
        
        left, top, right, bottom = crop_area
        res[top:bottom, left:right] += output_crop

    return res / weight_mask

def compute_stats(truth_img, result_img, model):
    fd = {model.truth_img: truth_img, model.result_img: result_img}
    ms_rgb, ps_rgb, ss_rgb = model.sess.run([model.test_ms_ssim_rgb, model.test_psnr_rgb, model.test_ssim_rgb], feed_dict=fd)
    return ms_rgb, ps_rgb, ss_rgb

def test_single_image(model, img_path, batch_size, crop_size, step):
    truth_image = img_to_np(Image.open(img_path))
    input_image = Image.fromarray(np.uint8(truth_image * 255))

    result_image = process_img(model, input_image, crop_size, step, batch_size)
    
    ms_rgb, ps_rgb, ss_rgb = compute_stats(truth_image, result_image, model)
    return ms_rgb, ps_rgb, ss_rgb, (truth_image, result_image)

def test_single_image_celeb(model, img_path, batch_size, crop_size, step):
    input_image = Image.open(img_path)
    truth_image = celeba_crop(input_image, 64)

    result_image = run_through_net(model, truth_image, crop_size)
    
    ms_rgb, ps_rgb, ss_rgb = compute_stats(truth_image, result_image, model)
    return ms_rgb, ps_rgb, ss_rgb, (truth_image, result_image)
    
def run_tests(all_img_paths, model, crop_size, step, batch_size=16):
    ms_ssims_rgb, psnrs_rgb, ssims_rgb = [], [], []
    for img_path in tqdm(all_img_paths):
        ms_rgb, ps_rgb, ss_rgb, _ = model.image_test_func(model, img_path, batch_size, crop_size, step)

        ms_ssims_rgb.append(ms_rgb)
        psnrs_rgb.append(ps_rgb)
        ssims_rgb.append(ss_rgb)
        
    return np.mean(ms_ssims_rgb), np.mean(psnrs_rgb), np.mean(ssims_rgb)