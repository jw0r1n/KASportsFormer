import json
import random

import torch
import time
import numpy as np
import yaml
import os
from easydict import EasyDict
import logging
import copy



def same_seed_fix(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream) -> None:
        """Initialise Loader."""
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node):
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def yaml_config_reader(yaml_config_path: str):
    yaml.add_constructor('!include', construct_include, yaml.Loader)
    with open(yaml_config_path, 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config = EasyDict(config)
    _, config_filename = os.path.split(yaml_config_path)
    config_name, _  = os.path.splitext(config_filename)
    config.config_name = config_name
    return config






def get_logger(dir_path: str, file_name: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    emojilist = ['❤️,', '✅', '✨', '😊', '⭐', '🛒', '😭', '😂', '🎉', '💀', '🎄', '🔥', '👍', '🙏', '🎁', '👉', '🚀', '📍', '➡️', '👇', '💙', '👀', '❌', '😉', '🤩', '😍', '🤔']
    selectemoji = random.choice(emojilist)


    myformatter = logging.Formatter(fmt=("[%(asctime)s|%(filename)s|%(levelname)s]" + selectemoji + "%(message)s"), datefmt="%a %b %d %H:%M:%S %Y")
    shandler = logging.StreamHandler()
    shandler.setFormatter(myformatter)
    logger.addHandler(shandler)

    time_str = str(time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fhandler = logging.FileHandler(os.path.join(dir_path, time_str + file_name), mode='w')
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(myformatter)
    logger.addHandler(fhandler)

    return logger


class AverageMetering(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def checkpoint_save(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'learning_rate': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def decay_learning_rate_exponentially(learning_rate, learning_rate_decay, optimizer):
    learning_rate *= learning_rate_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= learning_rate_decay
    return learning_rate


def joint_flip(joint_data, left_joints_index=[1, 2, 3, 14, 15, 16], right_joints_index=[4, 5, 6, 11, 12, 13], deep_copy=True):
    if deep_copy:
        flipped_data = copy.deepcopy(joint_data)
    else:
        flipped_data = joint_data
    flipped_data[..., 0] *= -1 # flip x of all joints
    flipped_data[..., left_joints_index + right_joints_index, :] = flipped_data[..., right_joints_index + left_joints_index, :] # Change orders
    return flipped_data

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    test_config = yaml_config_reader(r'../config/sportspose.yaml')

