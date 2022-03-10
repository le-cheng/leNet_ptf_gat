import logging
import os
import shutil
import socket

import hydra
import torch
import torch.distributed as dist


def is_main_process():
    return get_rank() == 0  

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value = value/world_size
        return value

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_logger(filename='main.log'):
    logger_name = "train.log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    hfile = logging.FileHandler(filename)    #创建一个文件记录日志的handler,设置级别为info
    hfile.setLevel(logging.INFO)

    fmt = "[%(asctime)s %(levelname)s %(filename)s (line:%(lineno)d) %(process)d] %(message)s"
    formatter = logging.Formatter(fmt)

    sh.setFormatter(formatter)
    hfile.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(hfile) #把对象加到logger里
    return logger

def copy_(in_, out_, absolute = True):
    if absolute:
        shutil.copy(hydra.utils.to_absolute_path(in_), out_)
    else:
        out_ = hydra.utils.to_absolute_path(out_)
        if os.path.exists(out_):
           os.remove(out_)
        shutil.copy(hydra.utils.to_absolute_path(in_), out_)

def copytree_(in_, out_, absolute = True):
    if absolute:
        shutil.copytree(hydra.utils.to_absolute_path(in_), out_)
    else:
        out_ = hydra.utils.to_absolute_path(out_)
        if os.path.exists(out_):
           shutil.rmtree(out_)
        shutil.copytree(in_, out_)


import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
