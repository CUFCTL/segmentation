# Installion:
# conda create --name swiftnet python=3.8 numpy pillow matplotlib tqdm numpy opencv
# pip install torch
# pip install torchvision

# Parameters that need to be changed for each dataset:
#   1. Maybe the mean/std in rn18_pyramid.py
#   2. Target size in rn18_pyramid.py
#   3. Evaluating variable in rn18_pyramid.py
"""train.py
Main training script.

To train the model on a train set:
    - In configs/rn18_pyramid.py set evaluting=False
    - In configs/rn18_pyramid.py set live_video=False 
    - In configs/rn18_pyramid.py set the target_size and target_size_feats to the shape of the image 
      (usually the full size of the image)
    - In configs/rn18_pyramid.py set the root variable to the dataset root directory
    - Set the desired batch size and number epochs (default epochs:250)
    
Usage: bash eval.sh
"""
import argparse 
import os 
from pathlib import Path
import torch
import torch.onnx
import importlib.util
import datetime
import sys
from shutil import copy
import pickle
from time import perf_counter
import shutil
from contextlib import redirect_stdout

from evaluation import evaluate_semseg


def import_module(path):
    """Load Python Module given the path"""
    spec = importlib.util.spec_from_file_location('module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store(model, store_path, name):
    with open(store_path.format(name), 'wb') as f:
        torch.save(model.state_dict(), f)


class Logger2(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


class Logger(object):
    """Create log file that saves the prints to stdout"""
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 


class Trainer:
    def __init__(self, conf, args, name):
        self.conf = conf
        using_hparams = hasattr(conf, 'hyperparams')
        print(f'Using hparams: {using_hparams}')
        self.hyperparams = self.conf
        self.args = args
        self.name = name
        self.model = self.conf.model
        self.optimizer = self.conf.optimizer

        self.dataset_train = self.conf.dataset_train
        self.dataset_val = self.conf.dataset_val
        self.loader_train = self.conf.loader_train
        self.loader_val = self.conf.loader_val

    def __enter__(self):
        """__enter__ is a Python special method that is called when this class is wrapped in a 'with' statement"""
        self.best_iou = -1
        self.best_iou_epoch = -1
        self.validation_ious = []
        self.experiment_start = datetime.datetime.now()

        if self.args.resume:
            self.experiment_dir = Path(self.args.resume)
            print(f'Resuming experiment from {args.resume}')
        else:
            self.experiment_dir = Path(self.args.store_dir) / (
                    self.experiment_start.strftime('%Y_%m_%d_%H_%M_%S_') + self.name)

        self.checkpoint_dir = self.experiment_dir / 'stored'
        self.store_path = str(self.checkpoint_dir / '{}.pt')

        if not self.args.dry and not self.args.resume:
            os.makedirs(str(self.experiment_dir), exist_ok=True)
            os.makedirs(str(self.checkpoint_dir), exist_ok=True)
            copy(self.args.config, str(self.experiment_dir / 'config.py'))
        
        # This code block was causing further print statements to not work so I am leaving it out
        #if self.args.log and not self.args.dry:
            #f = (self.experiment_dir / 'log.txt').open(mode='a')
        #    pass
        self.model.cuda()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__ is a Python special method that is called when this class exits a 'with' statement.
        When training is done, this function stores the best model weights.
        """
        if not self.args.dry:
            store(self.model, self.store_path, 'model')
        if not self.args.dry:
            with open(f'{self.experiment_dir}/val_ious.pkl', 'wb') as f:
                pickle.dump(self.validation_ious, f)
            dir_iou = Path(self.args.store_dir) / (f'{self.best_iou:.2f}_'.replace('.', '-') + self.name)
            if os.path.isdir(dir_iou):
                shutil.rmtree(dir_iou)
            os.rename(self.experiment_dir, dir_iou)

    def train(self):
        """Main training loop for model, hyperparameters are grabbed from the config file (rn18_pyramid.py)"""
        num_epochs = self.hyperparams.epochs
        start_epoch = self.hyperparams.start_epoch if hasattr(self.hyperparams, 'start_epoch') else 0
        print(num_epochs)
        sys.stdout = Logger(self.experiment_dir / 'log.txt')
        for epoch in range(start_epoch, num_epochs):
            if hasattr(self.conf, 'epoch'): # False
                self.conf.epoch.value = epoch
                print(self.conf.epoch)
            self.model.train()
            try:
                #self.conf.lr_scheduler.step() # Placing this after optimizer due to warning
                print(f'Elapsed time: {datetime.datetime.now() - self.experiment_start}')
                for group in self.optimizer.param_groups:
                    print('LR: {:.4e}'.format(group['lr']))
                # Evaluate at every 4 epochs and the last epoch
                eval_epoch = ((epoch % self.conf.eval_each == 0) or (epoch == num_epochs - 1))  # and (epoch > 0)
                self.model.criterion.step_counter = 0
                print(f'Epoch: {epoch} / {num_epochs - 1}')
                if eval_epoch and not self.args.dry:
                    print("Experiment dir: %s" % self.experiment_dir)
                # Creates enumerate object to get the batch index - same as using enumerate in the next for loop
                batch_iterator = iter(enumerate(self.loader_train))
                start_t = perf_counter()
                for step, batch in batch_iterator:
                    self.optimizer.zero_grad()
                    loss = self.model.loss(batch) # model.loss() handles transfering data to gpu, propagating data through model, calculating the loss 
                    loss.backward()
                    self.optimizer.step()
                    self.conf.lr_scheduler.step()
                    if step % 80 == 0 and step > 0:
                        curr_t = perf_counter()
                        # step * batch_size = # of images that have been processed so far
                        print(f'{(step * self.conf.batch_size) / (curr_t - start_t):.2f}fps')  
                if not self.args.dry:
                    store(self.model, self.store_path, 'model')
                    store(self.optimizer, self.store_path, 'optimizer')
                print(self.args.eval)
                print(self.args.eval_train)
                # Evaluate model every 4 epochs and the last epoch using the validation set
                if eval_epoch and self.args.eval:
                    print('Evaluating model')
                    iou, per_class_iou = evaluate_semseg(self.model, self.loader_val, self.dataset_val.class_info)
                    self.validation_ious += [iou]
                    # Runs evaluation code with the training set - Normally not used
                    if self.args.eval_train:
                        print('Evaluating train')
                        evaluate_semseg(self.model, self.loader_train, self.dataset_train.class_info)
                    # Save the best model w/ its iou and epoch number
                    if iou > self.best_iou:
                        self.best_iou = iou
                        self.best_iou_epoch = epoch
                        if not self.args.dry:
                            copy(self.store_path.format('model'), self.store_path.format('model_best'))
                    print(f'Best mIoU: {self.best_iou:.2f}% (epoch {self.best_iou_epoch})')

            except KeyboardInterrupt:
                break


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--store_dir', default='saves/', type=str, help='Path to experiments directory')
parser.add_argument('--resume', default=None, type=str, help='Path to existing experiment dir')
parser.add_argument('--no-log', dest='log', action='store_false', help='Turn off logging')
parser.add_argument('--log', dest='log', action='store_true', help='Turn on train evaluation')
parser.add_argument('--no-eval-train', dest='eval_train', action='store_false', help='Turn off train evaluation')
parser.add_argument('--eval-train', dest='eval_train', action='store_true', help='Turn on train evaluation')
parser.add_argument('--no-eval', dest='eval', action='store_false', help='Turn off evaluation')
parser.add_argument('--eval', dest='eval', action='store_true', help='Turn on evaluation')
parser.add_argument('--dry-run', dest='dry', action='store_true', help='Don\'t store')
parser.set_defaults(log=True)
parser.set_defaults(eval_train=False)
parser.set_defaults(eval=True)

if __name__ == '__main__':
    """Load config file (rn18_pyramid.py), """
    
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)
    

    with Trainer(conf, args, conf_path.stem) as trainer:
        trainer.train()
