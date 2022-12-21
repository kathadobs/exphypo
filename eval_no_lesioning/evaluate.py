import os
import tqdm
import torch
import torchvision
import numpy as np
import shutil
import matplotlib.pyplot as plt
import copy
from utils import helper_new as helper
from utils import tools_new as tools
import lesion
import transfer

# from utils import helper
# from utils import tools
from scipy.stats import spearmanr
import argparse

# --------------- FLAGS 
parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--config_file',     default=None,  type=str,  help='path to config file')
parser.add_argument('--ngpus',           default=1,     type=int,  help='number of gpus to use')
parser.add_argument('--workers',         default=4,     type=int,  help='read and write workers')
parser.add_argument('--task_index',      default=0,     type=int,  help='the index for the task i.e. 0 or 1')
parser.add_argument('--batch_size',      default=128,   type=int,  help='batch size lol')
parser.add_argument('--per_categ',       default=None,  type=int,  help='number of imaages per category in task_index')
parser.add_argument('--shuffle',         default="False", type=str,  help='shuffle data in batches')
parser.add_argument('--max_batches',     default=None,     type=int,  help='max bacthes we read from the data_loader')
parser.add_argument('--notebook',        default="False",  type=str, help='True if using a notebook')
parser.add_argument('--reduce_loss',     default="False",   type=str, help='')
parser.add_argument('--topk',            default=1,      type=int, help='')
parser.add_argument('--write_loss',      default="False",  type=str, help='if True saves to npy file')
parser.add_argument('--write_predictions',default="False",  type=str, help='if True saves to npy file')
parser.add_argument('--data_split',      default='test', type=str,  help='train, valid, or test')
parser.add_argument('--iterator_seed',   default=0, type=int,  help='predict iterator')
parser.add_argument('--maxout',          default="False", type=str,  help='read all data first before shuffling and truncating')
parser.add_argument('--read_seed',       default=None, type=int,  help="seed for validator reading in data")
parser.add_argument('--restore_epoch',   default=-1, type=int,  help="epoch to restore from")


FLAGS, FIRE_FLAGS = parser.parse_known_args()
    
# converts strings to corresponding boolean values
FLAGS.shuffle = True if (FLAGS.shuffle == 'True') else False
FLAGS.notebook = True if (FLAGS.notebook == 'True') else False
FLAGS.reduce_loss = True if (FLAGS.reduce_loss == 'True') else False
FLAGS.write_loss = True if (FLAGS.write_loss == 'True') else False
FLAGS.write_predictions = True if (FLAGS.write_predictions == 'True') else False
    
helper.printArguments(FLAGS=FLAGS)


if FLAGS.ngpus > 0:
    torch.backends.cudnn.benchmark = True

def write_predictions(directory, y_true, y_pred):
    predictions_filename = os.path.join(directory, 'predictions.jsonl')
    print("predictions_filename",predictions_filename)
    writer_method = 'a'
    if not os.path.exists(directory):
        writer_method = 'w'
        os.makedirs(directory)
    keys = ['y_true', 'y_pred']
    values = [y_true.tolist(), y_pred.tolist()]
    lesion.write_to_json(filename=predictions_filename, writer_method=writer_method, keys=keys, values=values)
    return None
    
# Get Model
config = helper.Config(config_file=FLAGS.config_file)
model, ckpt_data = config.get_model(pretrained=True, ngpus=FLAGS.ngpus, dataParallel=True, epoch=FLAGS.restore_epoch)

basename = os.path.basename(config.data_dir[FLAGS.task_index])
print(basename)
data_dir = []
for i, directory in enumerate(config.data_dir):
    new_dir = os.path.join(directory, FLAGS.data_split)
    data_dir.append(new_dir)

max_valid_samples = {}
for key in config.max_valid_samples.keys():
    if FLAGS.per_categ is None:
        max_valid_samples[key] = config.max_valid_samples[key]
    else:
        if key == basename:
            max_valid_samples[key] = FLAGS.per_categ
        else:
            max_valid_samples[key] = 0
data_subdir = ''

config.data_dir = data_dir        
config.max_valid_samples = max_valid_samples 


# validator
validator = transfer.Validator(name='valid',
                             model=model, 
                             batch_size=FLAGS.batch_size,
                             data_dir=config.data_dir, 
                             data_subdir=data_subdir,
                             max_samples=config.max_valid_samples,
                             maxout=FLAGS.maxout,
                             read_seed=FLAGS.read_seed,
                             ngpus=FLAGS.ngpus,
                             shuffle=FLAGS.shuffle,
                             includePaths=True,
                             workers=FLAGS.workers)

y_true, y_pred,  y_prob, paths, loss, _ = helper.predict(model=model, 
                                                         data_loader=validator.data_loader, 
                                                         ngpus=FLAGS.ngpus, 
                                                         topk=FLAGS.topk, 
                                                         notebook=FLAGS.notebook, 
                                                         max_batches=FLAGS.max_batches, 
                                                         reduce_loss=FLAGS.reduce_loss,
                                                         seed=FLAGS.iterator_seed) 
y_pred = np.squeeze(y_pred)

if FLAGS.write_loss:
    network_dir = os.path.basename(os.path.dirname(FLAGS.config_file))
    loss_dir = os.path.join('./evaluations', network_dir, config.name)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    loss_filename = os.path.join(loss_dir, "losses")
    np.save(loss_filename, loss)

if FLAGS.write_predictions:
    network_dir = os.path.basename(os.path.dirname(FLAGS.config_file))
    predictions_directory = os.path.join('./evaluations', network_dir, config.name)
    write_predictions(predictions_directory, y_true, y_pred)
    
    
    