import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    

# 修改版本

class LeNetWithDropout(nn.Module):
    def __init__(self, p_dropout=0.5):
        super(LeNetWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout_conv = nn.Dropout2d(p=p_dropout)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.dropout_fc = nn.Dropout(p=p_dropout)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleResBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        else:
            self.skip = None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.skip is not None:
            identity = self.skip(identity)
        out += identity
        out = self.pool(out)
        return out

class LeNetResDropout(nn.Module):
    def __init__(self, p_dropout=0.5):
        super(LeNetResDropout, self).__init__()
        # 把第一段 conv1→conv2 换成一个“残差块 + 池化”
        # 注意输入是 3 通道，假设输出我们做 16 通道
        self.res1 = SimpleResBlock(in_c=3, mid_c=8, out_c=16)
        self.res2 = SimpleResBlock(in_c=16, mid_c=16, out_c=32)
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.res1(x)    # shape: batch×16×(32/2)=16×16×16
        x = self.res2(x)    # shape: batch×32×(16/2)=32×8×8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x