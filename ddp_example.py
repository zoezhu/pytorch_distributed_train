'''
Pytorch分布式训练脚本，需要安装pytorch。

@author zz
@date 2020.3.31
'''

import os
import subprocess
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp


class ConvNet(nn.Module):    
    def __init__(self, num_classes=10):        
        super(ConvNet, self).__init__()        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),            
            nn.BatchNorm2d(16),            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2))        
        self.layer2 = nn.Sequential(            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),            
            nn.BatchNorm2d(32),            
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2))        
        self.fc = nn.Linear(7*7*32, num_classes)    
        
    def forward(self, x):        
        out = self.layer1(x)        
        out = self.layer2(out)        
        out = out.reshape(out.size(0), -1)        
        out = self.fc(out)        
        return out


###################
## 分布式训练
###################
def dist_train(gpu, args):
    rank = gpu  # 当前进程号
    print('Rank id: ', rank)
    dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = ConvNet()
    if args.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if gpu == 0:
            print('Use SyncBN in training')
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))



def main():    
    parser = argparse.ArgumentParser()    
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')  # gpu总数量
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--backend', default='nccl', type=str, help='backend used for distributed train')
    parser.add_argument('--syncbn', default=False, action="store_true", help='whether to use syncbn while training')
    args = parser.parse_args()
       
    args.world_size = args.gpus  # 进程总数
    args.init_method = 'tcp://10.9.1.2:34567'  
    mp.spawn(dist_train, nprocs=args.gpus, args=(args,))         


if __name__ == '__main__':
    main()















