import gc
import importlib
import os
from time import time

import hydra
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ModelNet40

from dataset import ModelNetDataLoader
from train_test import data_prefetcher
from utils import *

@hydra.main(config_path='config', config_name='cls')
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)# Using OmegaConf.set_struct, it is possible to prevent the creation of fields that do not exist
    global logger
    logger =  get_logger('simgletrain.log') # name

    print("\n")
    logger.info("=> start code ---------------")
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    logger.info("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))
    logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device}, GPU is NO.{str(cfg.gpu)} device')
    logger.info(f'World_size:{str(get_world_size())}')

    copy_('models/{}/model_bn4.py'.format(cfg.model.name), '.')
    copy_('models/{}/model_bn4.py'.format(cfg.model.name), 'models/{}.py'.format(cfg.model_copy_name), absolute = False)
    copy_(__file__, '.')
    copy_(cfg.cfg_path, '.')

    if cfg.manual_seed:
        random.seed(cfg.manual_seed)
        # np.random.seed(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed_all(cfg.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    '''DATA LOADING'''
 
    logger.info('Load dataset ---------------')
    # DATA_PATH = hydra.utils.to_absolute_path('data/modelnet40_normal_resampled/')
    DATA_PATH = hydra.utils.to_absolute_path('data/')
    # TRAIN_DATASET = ModelNetDataLoader(
    #     root=DATA_PATH, npoint=cfg.num_point, 
    #     split='train' , normal_channel=cfg.normal, 
    #     classes=cfg.num_class, uniform=cfg.uniform)
    # TEST_DATASET = ModelNetDataLoader(
    #     root=DATA_PATH, npoint=cfg.num_point, 
    #     split='test'  , normal_channel=cfg.normal, 
    #     classes=cfg.num_class, uniform=cfg.uniform)
    # Train_DataLoader = DataLoader(
    #     dataset=TRAIN_DATASET , batch_size=cfg.batch_size, 
    #     shuffle=True  , num_workers=cfg.num_workers)
    # Test_DataLoader =  DataLoader(
    #     dataset=TEST_DATASET  , batch_size=cfg.batch_size, 
    #     shuffle=False , num_workers=cfg.num_workers)

    Train_DataLoader = DataLoader(ModelNet40(root=DATA_PATH, partition='train', num_points=cfg.num_point), num_workers=8,
                              batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    Test_DataLoader = DataLoader(ModelNet40(root=DATA_PATH, partition='test', num_points=cfg.num_point), num_workers=8,
                             batch_size=cfg.test_batch_size, shuffle=False, drop_last=False)

    
    '''MODEL LOADING'''
    logger.info('Load MODEL ---------------')
    model = getattr(importlib.import_module('models.{}'.format(cfg.model_copy_name)), 'PointTransformerCls')(cfg).cuda()
    if cfg.print_model:
        logger.info(model)
    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


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
    # lossfn = torch.nn.CrossEntropyLoss().cuda()
    lossfn = cal_loss
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    """scheduler = MultiStepLR(
        optimizer, 
        milestones = [cfg.epoch*0.6,cfg.epoch*0.95], 
        gamma=cfg.scheduler_gamma
        )"""
    scheduler = CosineAnnealingLR(optimizer, cfg.epoch, eta_min=1e-3)

    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    '''TRANING'''
    logger.info('Start training...\n')
    global writer
    writer = SummaryWriter()
    t1 = time()
    for epoch in range(start_epoch,cfg.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, cfg.epoch))

        train_instance_acc,epoch_loss = train(model, Train_DataLoader, optimizer, epoch, lossfn)
        logger.info('Train Instance Accuracy: %f , Train Instance Loss: %f' % (train_instance_acc,epoch_loss))
        # logger.info('Train Instance Loss: %f' % epoch_loss)
        writer.add_scalar('train_Acc', train_instance_acc, epoch)
        writer.add_scalar('train_Loss', epoch_loss, epoch)
        scheduler.step()
        instance_acc, class_acc = test(model, Test_DataLoader, cfg.num_class)

        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        if (instance_acc >= best_instance_acc):
            best_instance_acc = instance_acc
        logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
        logger.info('Best Instance Accuracy: [%f], Class Accuracy: [%f]'% (best_instance_acc, best_class_acc))   
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

    logger.info('End of training...')
    t2 = time()
    logger.info('trian and eval model time is %.4f h'%((t2-t1)/3600))
    writer.close()
    return 0

def train(model, Train_DataLoader, optimizer, epoch, lossfn):
    model.train()
    correct = 0
    epoch_loss = 0 
    num_len = len(Train_DataLoader.dataset)
    Train_DataLoader = tqdm(Train_DataLoader)

    # prefetcher = data_prefetcher(Train_DataLoader)
    # points, target = prefetcher.next()
    # while points is not None:
    for points, label in Train_DataLoader:
        points, label = points.cuda(non_blocking=True), label.squeeze(-1).cuda(non_blocking=True)
        pred = model(points)
        loss = lossfn(pred, label.long())
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # 计算
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum()
        epoch_loss+=loss
        # points, target = prefetcher.next()
    train_instance_acc = correct / num_len
    epoch_loss = epoch_loss / len(Train_DataLoader)
    return train_instance_acc, epoch_loss

def test(model, test_loader, num_class=40):
    model.eval()# 一定要model.eval()在推理之前调用方法以将 dropout 和批量归一化层设置为评估模式。否则会产生不一致的推理结果。
    class_acc = torch.zeros((num_class,3)).cuda()
    num_len = len(test_loader.dataset)
    with torch.no_grad():
        correct=0
        test_loader = tqdm(test_loader)
        for _, (points, target) in enumerate(test_loader):
            points, target = points.cuda(), target.squeeze(-1).cuda()
            pred = model(points)
            pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

            for cat in torch.unique(target):
                cat_idex = (target==cat)
                classacc = pred[cat_idex].eq(target[cat_idex].view_as(pred[cat_idex])).sum()
                class_acc[cat,0] += classacc
                class_acc[cat,1] += cat_idex.sum()
            # correct += pred.eq(target.view_as(pred)).cpu().sum()

        test_instance_acc=correct / num_len
        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
        class_acc_t = torch.mean(class_acc[:,2])
    return test_instance_acc, class_acc_t


if __name__ == '__main__':
    # 垃圾回收gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    gc.collect()
    main()


