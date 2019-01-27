import argparse
import tensorflow as tf
import datetime
import pathlib

from hyper_picture_framework import HyperPictureFramework
from tensorflow.contrib.training import HParams
from lazy_datasets import *
from compress_target_network import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/DIV2K_train_HR")
parser.add_argument('--test_dataset', default='/home/jakub/Studia/mgr/networks-do-networks/data/kodak')

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
parser.add_argument("--in_func",default="none")

parser.add_argument("--to_yuv",default=False,type=bool)
parser.add_argument("--quant_method",default=0,type=int)
parser.add_argument("--quant_size",default=128.0,type=float)
parser.add_argument("--resnet_type",default=0,type=int)

parser.add_argument("--batch_size",default=16,type=int)
parser.add_argument("--steps",default=100000,type=int)
parser.add_argument("--test_per_iterations",default=500, type=int)

parser.add_argument("--queue_capacity",default=32, type=int) 

parser.add_argument('--target_layers', default=[2, 32, 64, 256, 64, 3], nargs='+', type=int)
# parser.add_argument('--target_layers', default=[2, 32, 128, 256, 128, 64, 3], nargs='+', type=int)
parser.add_argument('--trainables_mask', default='T,T,T,T,T')

parser.add_argument("--noise_level",default=15, type=int)

parser.add_argument("--vgg_loss_lambda",default=0.005,type=float)

args = parser.parse_args()

def parse_mask(mask_str):
    def get_mask(str):
        if str == 'T':
            return True
        else:
            return False
    return [get_mask(s) for s in mask_str.split(',')]

target_layers = args.target_layers

trainables_mask = parse_mask(args.trainables_mask)
trainable_layers = [[target_layers[i], target_layers[i + 1]] for i in range(len(trainables_mask)) if trainables_mask[i]]
const_layers = [(target_layers[i], target_layers[i + 1]) for i in range(len(trainables_mask)) if not trainables_mask[i]]

hparams = HParams()
hyper_parameters = {
    'train_dataset_path' : args.dataset,
    'test_dataset_path' : args.test_dataset,

    'checkpoint':args.checkpoint,
    'metagraph':args.metagraph,
    
    'in_img_width': args.img_x,
    'in_img_height': args.img_y,
    'channels': args.channels,
    'in_func': args.in_func,

    'to_yuv': args.to_yuv,
    'quant_method': args.quant_method,
    'quant_size': args.quant_size,
    'resnet_type': args.resnet_type,

    'learning_rate': args.learning_rate,
    'decay_steps':args.decay_steps,
    'decay_rate':args.decay_rate,
    
    'batch_size': args.batch_size,
    'steps': args.steps,
    'test_per_iterations': args.test_per_iterations,
    'queue_capacity':args.queue_capacity,

    'target_layers': trainable_layers,
    'trainables_mask': trainables_mask,

    'noise_level': args.noise_level,

    'vgg_loss_lambda': args.vgg_loss_lambda,
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
network = HyperPictureFramework(hparams, data_generator, model_name, saved_models_dir)
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

