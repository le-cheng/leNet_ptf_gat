import torch
import provider  # 数据增强
from utils import *
from tqdm import tqdm

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.data.numpy() #  (16, 1024, 6)
            self.next_input = provider.random_point_dropout(self.next_input)
            self.next_input[:,:, 0:3] = provider.random_scale_point_cloud(self.next_input[:,:, 0:3])
            self.next_input[:,:, 0:3] = provider.shift_point_cloud(self.next_input[:,:, 0:3])
            self.next_input = torch.Tensor(self.next_input)
            self.next_target = self.next_target[:, 0]
            # points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)

            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        points = self.next_input
        target = self.next_target
        self.preload()
        return points, target


def train(model, Train_DataLoader, optimizer, epoch, lossfn):
    model.train()
    correct = 0
    epoch_loss = 0 
    num_len = len(Train_DataLoader.dataset)
    if dist.get_rank() == 0:
        Train_DataLoader = tqdm(Train_DataLoader)

    # for batch_id, data in enumerate(Train_DataLoader, 0):
    prefetcher = data_prefetcher(Train_DataLoader)
    points, target = prefetcher.next()
    i = 0
    while points is not None:
        i += 1
        # points, target = data
        # points = points.data.numpy() #  (16, 1024, 6)
        # points = provider.random_point_dropout(points)
        # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
        # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
        # points = torch.Tensor(points)
        # target = target[:, 0]
        # points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        # Compute prediction and loss

        # 打印参数
        # for parameters in model.parameters():#打印出参数矩阵及值
        # print(len(list(model.named_parameters())))
        # for param_tensor in model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())

        pred = model(points)
        loss = lossfn(pred, target.long())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        #############################
        loss = reduce_value(loss, average=False)  
        # 经测试，每一次会自动同步参数
        ############################
        optimizer.step()
        # 计算
        pred = pred.argmax(dim=1, keepdim=True)
        ############################
        correct += pred.eq(target.view_as(pred)).sum()
        ############################
        epoch_loss+=loss

        points, target = prefetcher.next()
    
    train_instance_acc = correct / num_len
    train_instance_acc = reduce_value(train_instance_acc, average=False)
    epoch_loss = epoch_loss / len(Train_DataLoader)
    return train_instance_acc, epoch_loss
    

def test(model, test_loader, num_class=40):
    model.eval()# 一定要model.eval()在推理之前调用方法以将 dropout 和批量归一化层设置为评估模式。
                # 否则会产生不一致的推理结果。
    class_acc = torch.zeros((num_class,3)).cuda()
    num_len = len(test_loader.dataset)
    with torch.no_grad():
        correct=0
        if dist.get_rank() == 0:
            test_loader = tqdm(test_loader)
        for j, (points, target) in enumerate(test_loader):
            points, target = points.cuda(non_blocking=True), target[:, 0].cuda(non_blocking=True)
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
        test_instance_acc = reduce_value(test_instance_acc, average=False)
        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
        class_acc_t = torch.mean(class_acc[:,2])
        # class_acc_t = reduce_value(class_acc_t)
    return test_instance_acc, class_acc_t