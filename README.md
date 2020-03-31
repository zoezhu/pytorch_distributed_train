### 1. 数据并行的类型
Pytorch中提供了两种数据并行的方法，一种是`torch.nn.DataParallel`，另一种是`torch.nn.DistributedDataParallel`，简单地说，`DataParallel`适用于单机多卡的情况，而`DistributedDataParallel`适用于多机多卡的情况(当然单机多卡也是可以用的)。

- `DataParallel`：在module层上的并行，在模型运行的时候，先把模型复制并分发到多卡上，然后在forward阶段对每一个batch的输入数据进行分割(split)然后分发到各个卡上去运行，在backward阶段再收集所有卡上的梯度信息对模型进行更新。
- `DistributedDataParallel`：与`DataParallel`类似，但是是多线程运行的。


### 2. `torch.nn.DataParallel`
这个方法只需要对模型进行一层封装即可。
```python
device = [0,1,2,3,4,5,6,7]  # 用8卡进行训练
net = net.cuda()
net = nn.DataParallel(net, device_ids=device)
```
其他的训练步骤与单卡的一样，需要注意的是`DataLoader`中的`batch_size`是所有卡的batch_size，因为数据会先全部加载，再split到各个卡中。


### 3. `torch.nn.DistributedDataParallel`
这个方法需要对模型和数据都进行封装，在用`n`张卡训练的情况下，封装后的数据每个batch都是`n`个`batch_size`大小的数据块，当第`n`个进程去调用DataLoader取数据时，DataLoader会自动取第`n`个数据分片在第`n`张卡上做运算。  

```python
# 用nn.parallel.DistributedDataParallel封装模型
model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
# 用torch.utils.data.distributed.DistributedSampler创建sampler，并用此创建loader
# DistributedSampler默认是shuffle=True，是否需要打乱数据在此设置，用DistributedSampler创建的DataLoader必须是shuffle=False
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=args.world_size,
                                                                rank=rank)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler)
```

该分布式方法的总体思路是启用多进程，每个进程管理一个gpu(也可以一个进程管理多个gpu，但此处默认一个进程管理一张卡)，所以定义一个训练方法，让每个进程都运行这个方法。

```python
def dist_train(gpu, args):
    rank = gpu  # 当前进程号
    print('Rank id: ', rank)
    dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100  # 每张卡上batch_size
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
```

在并行训练前，先要保证`torch.distributed`已经被初始化，也就是要先调用`torch.distributed.init_process_group()`，这个方法中需要定义`backend`类型，启动方法`init_method`，pytorch官方建议使用`nccl`因为最快、支持内容也最多，启动方法此处使用'tcp'。调用代码如下。

```python
args.backend = 'nccl'
args.init_method = 'tcp://10.9.1.2:34567'  # ip根据机器修改  
mp.spawn(dist_train, nprocs=args.gpus, args=(args,)) 
```

总体的可运行代码在[ddp_example.py](ddp_example.py)，其中用到的`MNIST`会在运行时自动下载，运行脚本方法如下。

```shell
# 用两块卡训练
python test_dist.py -g 2

# 使用SyncBatchNorm
python test_dist.py -g 2 --syncbn
```




