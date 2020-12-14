# MobileNet_123
Version and some changes

# A PyTorch implementation of MobileNetV3

This is a PyTorch implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

Some details may be different from the original paper, welcome to discuss and help me figure it out.

- **[NEW]** The pretrained model of small version mobilenet-v3 is online, accuracy achieves the same as paper. 
- **[NEW]** The paper updated on 17 May, so I renew the codes for that, but there still are some bugs.
- **[NEW]** I remove the se before the global avg_pool (the paper may add it in error), and now the model size is close to paper.

## Training & Accuracy
### training setting:

1. number of epochs: 150
2. learning rate schedule: cosine learning rate, initial lr=0.05
3. weight decay: 4e-5
4. remove dropout
5. batch size: 256

### MobileNetV3 large
|              | Madds     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Offical 1.0  | 219 M     | 5.4  M     | 75.2%     | -                                                            |
| Offical 0.75 | 155 M     | 4    M     | 73.3%     | -                                                            |
| Ours    1.0  | 224 M     | 5.48 M     | 72.8%     | - |
| Ours    0.75 | 148 M     | 3.91 M     |  -        | - |

### MobileNetV3 small
|              | Madds     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Offical 1.0  | 66  M     | 2.9  M     | 67.4%     | -                                                            |
| Offical 0.75 | 44  M     | 2.4  M     | 65.4%     | -                                                            |
| Ours    1.0  | 63  M     | 2.94 M     | 67.4%     |  [[google drive](https://drive.google.com/open?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C)] |
| Ours    0.75 | 46  M     | 2.38 M     | -         | - |

## Usage
Pretrained models are still training ...
```python
    # pytorch 1.0.1
    # large
    net_large = mobilenetv3(mode='large')
    # small
    net_small = mobilenetv3(mode='small')
    state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
    net_small.load_state_dict(state_dict)
```

## Data Pre-processing

I used the following code for data pre-processing on ImageNet:

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

input_size = 224
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
    traindir, transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=n_worker, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=n_worker, pin_memory=True)
```

Implementation of MobileNet, modified from https://github.com/pytorch/examples/tree/master/imagenet.
imagenet data is processed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)


nohup python main.py -a mobilenet ImageNet-Folder  > log.txt &

Results
- sgd :                    top1 68.848 top5 88.740 [download](https://pan.baidu.com/s/1nuRcK3Z)
- rmsprop:                top1 0.104  top5 0.494
- rmsprop init from sgd :  top1 69.526 top5 88.978 [donwload](https://pan.baidu.com/s/1eRCxYKU)
- paper:                  top1 70.6

Benchmark:

Titan-X, batchsize = 16
```
  resnet18 : 0.004030
   alexnet : 0.001395
     vgg16 : 0.002310
squeezenet : 0.009848
 mobilenet : 0.073611
```
Titan-X, batchsize = 1
```
  resnet18 : 0.003688
   alexnet : 0.001179
     vgg16 : 0.002055
squeezenet : 0.003385
 mobilenet : 0.076977
```

---------

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
```


