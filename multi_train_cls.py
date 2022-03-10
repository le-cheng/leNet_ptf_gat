import os
from importlib import import_module
from time import time

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from numpy import random
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import ModelNetDataLoader
from train_test import *
from utils import *


@hydra.main(config_path='config', config_name='cls')
def mmmm(cfg: DictConfig):
    # OmegaConf.set_struct(cfg, False)# Using OmegaConf.set_struct, it is possible to prevent the creation of fields that do not exist
    # print(OmegaConf.to_yaml(cfg))
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    cfg.DATA_PATH = hydra.utils.to_absolute_path('data/modelnet40_normal_resampled/')
    copy_('models/{}/model_bn3.py'.format(cfg.model.name), '.')
    copy_('models/{}/model_bn3.py'.format(cfg.model.name), 'models/{}.py'.format(cfg.model_copy_name), absolute = False)
    copy_(__file__, '.')
    copy_(cfg.cfg_path, '.')

    '''CUDA Configuration'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    '''Set Seed'''
    if cfg.manual_seed:
        random.seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed_all(cfg.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    '''set DDP'''
    port = find_free_port()
    cfg.dist_url = f"tcp://localhost:{port}"
    cfg.ngpus_per_node = torch.cuda.device_count()
    cfg.world_size = cfg.ngpus_per_node * cfg.nodes    

    mp.spawn(main, nprocs=cfg.ngpus_per_node, args=(cfg,))   
    # PyTorch提供了mp.spawn来在一个节点启动该节点所有进程，每个进程运行train(i, args)，
    # 其中i从0到args.gpus - 1。

    copytree_('.', cfg.lastresult_path, absolute = False)


def main(gpu, cfg):
    rank = cfg.nr * cfg.ngpus_per_node + gpu	             
    dist.init_process_group(
        backend='nccl', init_method=cfg.dist_url, world_size=cfg.world_size, rank=rank)                                                           
    torch.cuda.set_device(gpu)

    if is_main_process():
        global logger, writer
        writer = SummaryWriter()
        logger =  get_logger('multitrain.log')

        # 起始打印信息
        logger.info("=> start code ---------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Using {device}, GPU is [{str(cfg.gpu)}] device')
        logger.info(f'World_size:{str(get_world_size())}')
        logger.info(f'Dist_url:{str(cfg.dist_url)}')
    
    '''DATA LOADING'''
    if is_main_process(): logger.info('Load dataset ...')
    # cfg.batch_size = int(cfg.batch_size / cfg.ngpus_per_node)
    TRAIN_DATASET = ModelNetDataLoader(
        root=cfg.DATA_PATH, npoint=cfg.num_point, 
        split='train', normal_channel=cfg.normal, 
        classes=cfg.num_class, uniform=cfg.uniform)
    TEST_DATASET = ModelNetDataLoader(
        root=cfg.DATA_PATH, npoint=cfg.num_point, 
        split='test', normal_channel=cfg.normal, 
        classes=cfg.num_class, uniform=cfg.uniform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET)
    test_sampler = torch.utils.data.distributed.DistributedSampler(TEST_DATASET)
    Train_DataLoader = DataLoader(
        dataset=TRAIN_DATASET, batch_size=cfg.batch_size, pin_memory=True,
        shuffle=(train_sampler is None), num_workers=cfg.num_workers, sampler=train_sampler)
    Test_DataLoader =  DataLoader(
        dataset=TEST_DATASET , batch_size=cfg.batch_size, pin_memory=True,
        shuffle=(test_sampler is None), num_workers=cfg.num_workers, sampler=test_sampler)

    '''MODEL LOADING'''
    if is_main_process(): logger.info('Load MODEL ...')
    model = getattr(import_module('models.{}'.format(cfg.model_copy_name)), 'PointTransformerCls')(cfg).cuda()
    if cfg.print_model:
        if is_main_process(): logger.info(model)

    # if gpu == 0:
    #     dummy_input = (torch.zeros(16, 1024, 6),)
    #     writer.add_graph(model, dummy_input, True)
    #     writer.close()
        # raise NotImplementedError()
    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        if is_main_process():
            logger.info('No existing model, starting training from scratch...')
        start_epoch = 0
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    
    '''Hyper Parameter超参数'''
    # Scale learning rate based on global batch size
    cfg.learning_rate = cfg.learning_rate*float(cfg.batch_size*cfg.world_size)/16.
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=cfg.learning_rate, 
            momentum=0.9,
            weight_decay=0.0001
            )
    lossfn = torch.nn.CrossEntropyLoss().cuda()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = MultiStepLR(
        optimizer, 
        milestones = [cfg.epoch*0.6, cfg.epoch*0.9], 
        gamma=cfg.scheduler_gamma)

    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0

    '''TRANING'''
    if is_main_process(): 
        logger.info('Start training...')
        t1 = time()
    for epoch in range(start_epoch,cfg.epoch):
        ###################################
        train_sampler.set_epoch(epoch)
        #############################
        if is_main_process(): logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, cfg.epoch))

        train_instance_acc,epoch_loss = train(model, Train_DataLoader, optimizer, epoch, lossfn)
        
        if is_main_process():
            logger.info('Train Instance Accuracy: %f' % train_instance_acc)
            logger.info('Train Instance Loss: %f' % epoch_loss)
            writer.add_scalar('train_Acc', train_instance_acc, epoch)
            writer.add_scalar('train_Loss', epoch_loss, epoch)
        scheduler.step()
        if cfg.test:
            instance_acc, class_acc = test(model, Test_DataLoader, cfg.num_class)
            if is_main_process():
                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc

                if (instance_acc > best_instance_acc):
                    best_instance_acc = instance_acc
                logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
                logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
                if (instance_acc >= best_instance_acc):
                    best_epoch = epoch + 1
                    logger.info('Save model...')
                    savepath = 'best_model.pth'
                    logger.info('Saving at %s'% savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                writer.add_scalar('Test_Acc', instance_acc, epoch)
                writer.add_scalar('Best_Acc', best_instance_acc, epoch)
                writer.add_scalar('ClassAcc', class_acc, epoch)
                writer.add_scalar('Best_ClassAcc', best_class_acc, epoch)
        global_epoch += 1

    if is_main_process():
        logger.info('End of training...')
        t2 = time()
        logger.info('trian and eval model time is %.4f h'%((t2-t1)/3600))
        writer.close()
    return 0

if __name__ == '__main__':
    import gc  # 加载gc模块
    gc.collect()  # 垃圾回收gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    mmmm()


