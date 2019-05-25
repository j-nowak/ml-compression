import argparse
import tensorflow as tf
import datetime
import pathlib

from training_framework import TrainingFramework
from tensorflow.contrib.training import HParams
from datasets import *
import test_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_type")

parser.add_argument("--dataset",default="data/DIV2K_train_HR")
parser.add_argument('--test_dataset', default='data/kodak')

parser.add_argument("--train_dir",default="train")
parser.add_argument("--tensorboard",default="tensorboard")
parser.add_argument("--savedir",default="saved_models")

parser.add_argument("--model_name",default="i_will_destroy_humans")
parser.add_argument("--checkpoint", default="") 
parser.add_argument("--metagraph", default="") 

parser.add_argument("--learning_rate",default=7e-05, type=float)
parser.add_argument("--decay_steps",default=100000,type=int)
parser.add_argument("--decay_rate",default=0.92,type=float)

parser.add_argument("--img_x",default=39,type=int)
parser.add_argument("--img_y",default=39,type=int)
parser.add_argument("--channels",default=3,type=int)

parser.add_argument("--quant_method",default=0,type=int)
parser.add_argument("--quant_size",default=128.0,type=float)

parser.add_argument("--batch_size",default=16,type=int)
parser.add_argument("--steps",default=100000,type=int)
parser.add_argument("--test_per_iterations",default=500, type=int)

parser.add_argument("--queue_capacity",default=32, type=int) 

parser.add_argument("--max_alpha",default=50.0,type=float)
parser.add_argument("--alpha_div",default=100000,type=float)

args = parser.parse_args()

hparams = HParams()
hyper_parameters = {
    'model_type': args.model_type,
    
    'train_dataset_path' : args.dataset,
    'test_dataset_path' : args.test_dataset,

    'checkpoint':args.checkpoint,
    'metagraph':args.metagraph,
    
    'in_img_width': args.img_x,
    'in_img_height': args.img_y,
    'channels': args.channels,

    'quant_method': args.quant_method,
    'quant_size': args.quant_size,

    'learning_rate': args.learning_rate,
    'decay_steps':args.decay_steps,
    'decay_rate':args.decay_rate,
    
    'batch_size': args.batch_size,
    'steps': args.steps,
    'test_per_iterations': args.test_per_iterations,
    'queue_capacity':args.queue_capacity,

    'max_alpha': args.max_alpha,
    'alpha_div': args.alpha_div,
}
for a,b in hyper_parameters.items():
    hparams.add_hparam(a, b)

# Load data
data_generator = Dataset(hparams)

# Prepare train dirs
model_name = args.model_name + '_' + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print('Running model: ' + model_name)
print('Hyperparameters: ')
for a,b in hyper_parameters.items():
    print(str(a) + ":" + str(b))


pathlib.Path(args.train_dir + '/' + args.savedir).mkdir(parents=True, exist_ok=True) 
saved_models_dir = args.train_dir + '/' + args.savedir + '/' + model_name + '/' + 'tmp'

pathlib.Path(args.train_dir + '/' + args.tensorboard).mkdir(parents=True, exist_ok=True) 
tensorboard_train_dir = args.train_dir + '/' + args.tensorboard + '/' + model_name + '/train'
tensorboard_test_dir = args.train_dir + '/' + args.tensorboard + '/' + model_name + '/test'

# Build net
network = TrainingFramework(hparams, data_generator, model_name, saved_models_dir, test_utils.test_single_image)
if hparams.checkpoint != '': 
    print('To restore graph you need to put metagraph path and directorty that contains checkpoint')
    if hparams.metagraph == '':
        raise Exception("Put metagraph path!!!")
    print('restoring checkpoint')
    network.restore(hparams.checkpoint, hparams.metagraph)
    print('restored')

# Train net
train_writer = tf.summary.FileWriter(tensorboard_train_dir, network.sess.graph)
test_writer = tf.summary.FileWriter(tensorboard_test_dir)
network.train(network.sess, train_writer, test_writer)

