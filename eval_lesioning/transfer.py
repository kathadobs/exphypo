import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision
from utils.helper import Config
import utils
import os
from utils.metrics import precision
import numpy as np
import copy

# image preprocessing steps     
IMAGE_RESIZE=256
IMAGE_SIZE=224
GRAYSCALE_PROBABILITY = 0.2
resize_transform      = torchvision.transforms.Resize(IMAGE_RESIZE)
random_crop_transform = torchvision.transforms.RandomCrop(IMAGE_SIZE)
center_crop_transform = torchvision.transforms.CenterCrop(IMAGE_SIZE)
grayscale_transform   = torchvision.transforms.RandomGrayscale(p=GRAYSCALE_PROBABILITY)
normalize             = torchvision.transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)
# Resnet baby model:
# normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class Trainer(object):

    def __init__(self, name, model, batch_size, learning_rate, step_size, weight_decay, data_dir, ngpus, workers, use_scheduler, 
                 max_samples=None, shuffle=True, data_subdir='train', includePaths=False, optim='adam', 
                 momentum=None, maxout=True, read_seed=None):
        self.name = name
        self.model = model
        self.learning_rate = learning_rate
        self.max_samples = max_samples
        self.maxout = maxout
        self.read_seed=read_seed
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.ngpus = ngpus
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle
        self.includePaths = includePaths
        self.dataset, self.data_loader = self.data()
        self.use_scheduler = use_scheduler
        
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                              lr=self.learning_rate, 
                                              betas=(0.9, 0.999), 
                                              eps=1e-08, 
                                              weight_decay=weight_decay, 
                                              amsgrad=False)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=momentum,
                                             weight_decay=weight_decay)
            
        if self.use_scheduler:
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.ngpus > 0:
            self.criterion = self.criterion.cuda()
   
    def data(self):
        if type(self.data_dir) is list:
            ImageFolder = utils.folder_list.ImageFolder
            train_data_dir = []
            for i in range(len(self.data_dir)):
                train_data_dir.append(os.path.join(self.data_dir[i], self.data_subdir))
        else:
            ImageFolder = utils.folder.ImageFolder
            train_data_dir = os.path.join(self.data_dir, self.data_subdir)
        
        #transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
        #                                           torchvision.transforms.RandomHorizontalFlip(),
        #                                           torchvision.transforms.ToTensor(),
        #                                           normalize,
        #                                          ])
        
        transform = torchvision.transforms.Compose([resize_transform, 
                                                    random_crop_transform, 
                                                    grayscale_transform, 
                                                    torchvision.transforms.ToTensor(),
                                                    normalize,
                                                   ])
        dataset = ImageFolder(root=train_data_dir, # [path/to/task1, path/to/task2]
                              max_samples=self.max_samples, # {"task1": 50, "task2": 50}
                              maxout=self.maxout,
                              read_seed=self.read_seed,
                              transform=transform,
                              includePaths=self.includePaths)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.shuffle,
                                                  num_workers=self.workers,
                                                  pin_memory=True)
        return dataset, data_loader

    def __call__(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        if self.ngpus > 0:
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)
        output = self.model(x=x)
        prec_1, prec_5 = precision(output=output, target=y, topk=(1,5))
        prec_1 /= len(output)
        prec_5 /= len(output)             
        loss = self.criterion(output, y)
        
        loss.backward() # compute gradients
        self.optimizer.step() # update weights 
        return loss.item(), prec_1, prec_5, output
    
    
class Validator(object):
    def __init__(self, name, model, batch_size, data_dir, ngpus, workers, 
                 max_samples=None, maxout=True, read_seed=None, 
                 shuffle=False, data_subdir='test', includePaths=False):
        self.name = name
        self.model = model
        self.max_samples = max_samples
        self.maxout=maxout
        self.read_seed=read_seed
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.ngpus = ngpus
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle
        self.includePaths = includePaths
        self.dataset, self.data_loader = self.data()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        if self.ngpus > 0:
            self.criterion = self.criterion.cuda()
 
    def data(self):
        if type(self.data_dir) is list:
            ImageFolder = utils.folder_list.ImageFolder
            test_data_dir = []
            for i in range(len(self.data_dir)):
                test_data_dir.append(os.path.join(self.data_dir[i], self.data_subdir))
        else:
            ImageFolder = utils.folder.ImageFolder
            test_data_dir = os.path.join(self.data_dir, self.data_subdir)
        
        #normalize = torchvision.transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)
        #transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
        #                                torchvision.transforms.CenterCrop(224),
        #                                torchvision.transforms.ToTensor(),
        #                                normalize,])
        
        transform = torchvision.transforms.Compose([resize_transform, 
                                                    center_crop_transform, 
                                                    torchvision.transforms.ToTensor(),
                                                    normalize,
                                                   ])
        
        dataset = ImageFolder(root=test_data_dir, 
                              max_samples=self.max_samples,
                              maxout=self.maxout,
                              read_seed=self.read_seed,
                              transform=transform,
                              includePaths=self.includePaths)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.shuffle,
                                                  num_workers=self.workers,
                                                  pin_memory=True)
        return dataset, data_loader
    
    def __call__(self, x, y):
        if self.ngpus > 0:
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)
        output = self.model(x=x)
        prec_1, prec_5 = precision(output=output, target=y, topk=(1,5))
        prec_1 /= len(output)
        prec_5 /= len(output)
        loss = self.criterion(output,y)
        return loss.item(), prec_1, prec_5, output

def train(trainer,validator,num_epochs,checkpoints_dir,log_dir,
          start_epoch=0, valid_freq=0.5,save_freq=1,notebook=False):
    
    # progress bars
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange

    else:
        from tqdm import tqdm as tqdm
        from tqdm import trange as trange
    
    writer = SummaryWriter(log_dir=log_dir)

    start_time = time.time()
    train_num_steps = len(trainer.data_loader)
    valid_num_steps = len(validator.data_loader)
    valid_step_freq = int(train_num_steps*valid_freq) 
    stop_epoch = start_epoch + num_epochs
    walltime = 0.0
    
    for epoch in trange(start_epoch, stop_epoch+1, initial=start_epoch, desc='epoch'):
        for train_step, train_batch in enumerate(tqdm(trainer.data_loader, desc='train')):
            global_step = (epoch) * train_num_steps + train_step
            if ((train_step) % valid_step_freq) == 0:
                validator.model.eval()
                with torch.no_grad():
                    avg_valid_loss = []
                    avg_valid_prec_1 = []
                    avg_valid_prec_5 = []
                    for valid_step, valid_batch in enumerate(tqdm(validator.data_loader, desc='valid')):
                        x_valid, y_valid = valid_batch
                        valid_start_time = time.time()
                        valid_loss, valid_prec_1, valid_prec_5, _ = validator(x=x_valid,y=y_valid)
                        valid_sec_per_iter = time.time() - valid_start_time
                        avg_valid_loss.append(valid_loss)
                        avg_valid_prec_1.append(valid_prec_1)
                        avg_valid_prec_5.append(valid_prec_5)
                    avg_valid_loss = np.mean(avg_valid_loss)
                    avg_valid_prec_1 = np.mean(avg_valid_prec_1)
                    avg_valid_prec_5 = np.mean(avg_valid_prec_5)

                    writer.add_scalar('valid/loss',         avg_valid_loss,   global_step)
                    writer.add_scalar('valid/precision1', avg_valid_prec_1, global_step)
                    writer.add_scalar('valid/precision5', avg_valid_prec_5, global_step)
                    
                    print('\n---------VALID EPOCH---------')
                    print("{0:<30}: {1:}".format('epoch',            epoch))
                    print("{0:<30}: {1:}".format('global_step',      global_step))
                    print("{0:<30}: {1:}".format('avg_valid_loss',   avg_valid_loss))
                    print("{0:<30}: {1:}".format('avg_valid_prec_1', avg_valid_prec_1))
                    print("{0:<30}: {1:}".format('avg_valid_prec_5', avg_valid_prec_5))
                    print('-----------------------------\n')
   
            # train step
            x_train, y_train = train_batch
            train_start_time = time.time()
            train_loss, train_prec_1, train_prec_5, _ = trainer(x=x_train, y=y_train)
            train_sec_per_iter = time.time() - train_start_time
            walltime += (time.time() - start_time)/(60.0**2)
            start_time = time.time()
            progress = (train_step+1.0)/train_num_steps
            writer.add_scalar('train/loss',         train_loss,   global_step)
            writer.add_scalar('train/precision1', train_prec_1, global_step)
            writer.add_scalar('train/precision5', train_prec_5, global_step)
            if trainer.use_scheduler:
                writer.add_scalar('meta/learning_rate',  trainer.optimizer.param_groups[0]['lr'], global_step) 
            else:
                writer.add_scalar('meta/learning_rate',  trainer.learning_rate, global_step) 
            writer.add_scalar('meta/progress',  progress, global_step) 
            writer.add_scalar('meta/train-sec-per-iter',  train_sec_per_iter, global_step) 
            writer.add_scalar('meta/valid-sec-per-iter',  valid_sec_per_iter, global_step) 
            writer.add_scalar('meta/walltime', walltime, global_step) 
            
            print('\n---------TRAIN STEP---------')
            print("{0:<30}: {1:}".format('epoch',         epoch))
            print("{0:<30}: {1:}".format('global_step',   global_step))
            print("{0:<30}: {1:}".format('progress',      progress))
            print("{0:<30}: {1:}".format('train_loss',    train_loss))
            print("{0:<30}: {1:}".format('train_prec_1',  train_prec_1))
            print("{0:<30}: {1:}".format('train_prec_5',  train_prec_5))
            if trainer.use_scheduler:
                print("{0:<30}: {1:}".format('learning_rate', trainer.optimizer.param_groups[0]['lr']))
            else:
                print("{0:<30}: {1:}".format('learning_rate', trainer.learning_rate))
            print('----------------------------\n')
        # Save Model
        if ((epoch % save_freq) == 0):
            ckpt_data = {}
            ckpt_data['batch_size']     = config.batch_size
            ckpt_data['learning_rate']  = trainer.learning_rate
            ckpt_data['momentum']       = config.momentum
            ckpt_data['step_size']      = config.step_size
            ckpt_data['weight_decay']   = config.weight_decay
            ckpt_data['walltime']       = walltime
            ckpt_data['epoch']          = epoch
            ckpt_data['state_dict']     = model.state_dict()
            ckpt_data['optimizer']      = trainer.optimizer.state_dict()
            torch.save(ckpt_data, os.path.join(config.checkpoints_dir,f'epoch_{epoch:02d}.pth.tar'))
        
        if trainer.use_scheduler:
            trainer.scheduler.step(metrics=avg_valid_loss,epoch=epoch)
        writer.add_scalar('meta/epoch', epoch, global_step)  
    writer.close()

if __name__ == "__main__":
    # --------------- FLAGS 
    '''
    Config file outlines the directories that the transfer model will use
        also defines the number of classes 
        hyperparameters like momentum, batch_size, and optimizer
        checkpoint dir for new transfer model
        log dir for new transfer model

    basemodel is a path to the checkpoint of model we perform transfer learning on

    '''

    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--config_file',   default=None,  type=str,  help='path to config file')
    parser.add_argument('--ngpus',         default=0,     type=int,  help='dafult is cpu (0 gpus)')
    parser.add_argument('--num_epochs',    default=50,    type=int,  help='num of epochs to train for')
    parser.add_argument('--valid_freq',    default=0.5,   type=float, help='how often to validate as percentage of epoch')
    parser.add_argument('--save_freq',     default=5,     type=int,     help='how often to save, dafault ever 5 epochs ')
    parser.add_argument('--workers',       default=4,     type=int,  help='number of workers for read and write of data')
    parser.add_argument('--maxout',        default=False, type=bool, help='read all data and then shuffle')
    parser.add_argument('--read_seed',     default=None,  type=int,  help='seed type')
    parser.add_argument('--learning_rate', default=None,  type=float,  help='initial learning rate')
    parser.add_argument('--basemodel',     default=None, type=str, help='path to base model')
    parser.add_argument('--restore',       default=False, type=bool, help='if True will restore from restore_epoch')
    parser.add_argument('--restore_epoch', default=-1, type=int, help='epoch to restore from, ignored if restore is false')
    FLAGS, FIRE_FLAGS = parser.parse_known_args()

    if FLAGS.ngpus > 0:
            torch.backends.cudnn.benchmark = True
    np.random.seed(0)
    torch.manual_seed(0)
    config = Config(config_file=FLAGS.config_file)
    config.printAttributes()

    if not os.path.exists(config.checkpoints_dir):
        os.makedirs(config.checkpoints_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # IMPORT MODEL Using Custom
    n_in = 2048
    n_out = 6269
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = torch.nn.Linear(in_features=n_in, out_features=n_out, bias=True)
    #model = torch.nn.DataParallel(model).cuda()
    ckpt_data = torch.load(FLAGS.basemodel)
    model_state_dict = {}
    for key, value in ckpt_data['model_state_dict'].items():
        key_list = np.array(key.split('.'))
        separator = '.'
        new_key = separator.join(key_list[1:])
        model_state_dict[new_key] = value
    ckpt_data['model_state_dict']=model_state_dict
    model.load_state_dict(ckpt_data['model_state_dict'])

    # DEFINE trainer and validator
    validator = Validator(name='valid',
                                   model=model, 
                                   batch_size=config.batch_size,
                                   data_dir=config.data_dir, 
                                   ngpus=FLAGS.ngpus, 
                                   workers=FLAGS.workers,
                                   max_samples=config.max_valid_samples,
                                   maxout=FLAGS.maxout,
                                   read_seed=FLAGS.read_seed)             
    trainer = Trainer(name='train',
                               model=model, 
                               batch_size=config.batch_size,
                               learning_rate=FLAGS.learning_rate,
                               optim=config.optimizer,
                               momentum=config.momentum,
                               step_size=config.step_size,
                               weight_decay=config.weight_decay,
                               data_dir=config.data_dir,  
                               ngpus=FLAGS.ngpus, 
                               workers=FLAGS.workers,
                               max_samples=config.max_train_samples,
                               maxout=FLAGS.maxout,
                               read_seed=FLAGS.read_seed,
                               use_scheduler=True)

    # MODIFY CLASSIFIER
    for param in model.parameters():
        param.requires_grad = False
    num_classes = config.num_classes
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model = torch.nn.DataParallel(model)
    if FLAGS.ngpus > 0:
        model = model.cuda()
        
    # RESTORE
    if FLAGS.restore:
        restore_path = utils.tools.get_checkpoint(epoch=FLAGS.restore_epoch, checkpoints_dir=config.checkpoints_dir) 
        if FLAGS.ngpus > 0:
            ckpt_data = torch.load(restore_path)
        else:
            print('Loading model onto cpu...')
            ckpt_data = torch.load(restore_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt_data['state_dict'])
        print('Restored from: ' + os.path.relpath(restore_path))
    
    if config.optimizer == 'adam':
        trainer.optimizer = torch.optim.Adam(params=model.module.fc.parameters(), 
                                             lr=FLAGS.learning_rate, 
                                             betas=(0.9, 0.999), 
                                             eps=1e-08, 
                                             weight_decay=config.weight_decay, 
                                             amsgrad=False)
    elif config.optimizer == 'sgd':
        trainer.optimizer = torch.optim.SGD(params=model.module.fc.parameters(),
                                                 lr=FLAGS.learning_rate,
                                                 momentum=config.momentum,
                                                 weight_decay=config.weight_decay)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
    
    print(model,flush=True)
              
    # Train Network
    train(trainer         = trainer,
          validator       = validator, 
          num_epochs      = FLAGS.num_epochs,
          checkpoints_dir = config.checkpoints_dir,
          log_dir         = config.log_dir,
          notebook        = False)