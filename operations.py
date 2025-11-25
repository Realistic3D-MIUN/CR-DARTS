import torch
import torch.nn as nn
import collections

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, prefix="", suffix=""):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        '''self.op = nn.Sequential()
        self.op.add_module(prefix + "_reluconvbn_" + str(kernel_size) + "_relu_" + suffix, nn.ReLU(inplace=False))
        self.op.add_module(prefix + "_reluconvbn_" + str(kernel_size) + "_conv2d_" + suffix, nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False))
        self.op.add_module(prefix + "_reluconvbn_" + str(kernel_size) + "_bn_" + suffix, nn.BatchNorm2d(C_out, affine=affine))'''

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, prefix="", suffix=""):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        '''
        self.op = nn.Sequential()
        self.op.add_module(prefix + "_dilconv_" + str(kernel_size) + "_relu_" + suffix, nn.ReLU(inplace=False))
        self.op.add_module(prefix + "_dilconv_" + str(kernel_size) + "_conv1_" + suffix, nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False))
        self.op.add_module(prefix + "_dilconv_" + str(kernel_size) + "_conv2_" + suffix, nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False))
        self.op.add_module(prefix + "_dilconv_" + str(kernel_size) + "_bn_" + suffix, nn.BatchNorm2d(C_out, affine=affine))'''


    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, prefix="", suffix=""):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            #nn.ReLU(inplace=False),
            #nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            #nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_out, affine=affine),
        )
        '''
        self.op = nn.Sequential()
        self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_relu1_" + suffix, nn.ReLU(inplace=False))
        self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_conv1_" + suffix, nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False))
        self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_conv2_" + suffix, nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False))
        self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_bn1_" + suffix, nn.BatchNorm2d(C_in, affine=affine))
        #self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_relu2_" + suffix, nn.ReLU(inplace=False))
        #self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_conv3_" + suffix, nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False))
        #self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_conv4_" + suffix, nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False))
        #self.op.add_module(prefix + "_sepconv_" + str(kernel_size) + "_bn2_" + suffix, nn.BatchNorm2d(C_out, affine=affine))
        '''

    def forward(self, x):
        return self.op(x)

class SpaSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, prefix="", suffix=""):
        super(SpaSepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        '''
        self.op = nn.Sequential()
        self.op.add_module(prefix + "_spasepconv_" + str(kernel_size) + "_relu_" + suffix, nn.ReLU(inplace=False))
        self.op.add_module(prefix + "_spasepconv_" + str(kernel_size) + "_conv1_" + suffix, nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=False))
        self.op.add_module(prefix + "_spasepconv_" + str(kernel_size) + "_conv2_" + suffix, nn.Conv2d(C_in, C_out, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=False))
        self.op.add_module(prefix + "_spasepconv_" + str(kernel_size) + "_bn_" + suffix, nn.BatchNorm2d(C_out, affine=affine))
        '''


    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.device
        padding = torch.zeros((n, c, h, w), dtype=x.dtype, device=device)
        return padding


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True, prefix="", suffix=""):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
