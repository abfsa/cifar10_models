import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import math
import random
from PIL import Image
from torchvision.transforms import functional as F
import models
import models.resnet
import models.resnet_cifar
import models.resnext
import models.wideresnet

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(output, target):
    """Computes the accuracy over top1 predictions"""
    with torch.no_grad():
        pred = output.argmax(dim = 1)
        assert pred.shape[0] == len(target)
        correct = pred.eq(target).sum().item()
        accuracy = correct / target.size(0) * 100.0
        return accuracy

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """
    保存当前状态，如果是最优则额外保存一份 best_checkpoint.pth
    state: dict, e.g. {'epoch':…, 'state_dict':…, 'optimizer':…, 'best_acc':…}
    """
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), 'best_' + os.path.basename(filename))
        torch.save(state, best_path)


# 定义mixup和cutout两种数据增强方式

class Cutout:
    """
    随机在图像上挖 n_holes 个正方形洞，洞的边长为 length。
    """
    def __init__(self, n_holes=1, length=4):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        img: PIL Image or Tensor(C,H,W) in [0,1]
        """
        if isinstance(img, torch.Tensor):
            h, w = img.size(1), img.size(2)
        else:
            w, h = img.size

        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1: y2, x1: x2] = 0.0

        # 对于 Tensor，直接乘；对 PIL Image，要先转为 Tensor
        if isinstance(img, torch.Tensor):
            img = img * mask.unsqueeze(0)
        else:
            img = F.to_tensor(img) * mask.unsqueeze(0)
            img = F.to_pil_image(img)
        return img


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_dataloaders(data_dir, batch_size, valid_size=0.1,
                    num_workers=2, pin_memory=True,
                    use_cutout=False, cutout_length=16):
    """
    新增参数：
      use_cutout: 是否在 train 上使用 Cutout
      cutout_length: 挖洞边长
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_cutout:
        train_transforms.append(Cutout(n_holes=1, length=cutout_length))
    train_transforms += [
        transforms.ToTensor(),
        normalize,
    ]
    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 后续同之前：划分 train/val、下载 dataset、创建 DataLoader
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    n_total = len(full_train)
    n_val = int(valid_size * n_total)
    n_train = n_total - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set,   batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n

# model = models.wideresnet.wrn_28_10()
# print(get_number_of_parameters(model))