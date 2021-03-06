import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class ShakeBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, p_shakedrop=1.0):
        super(ShakeBasicBlock, self).__init__()
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_ch, out_ch, stride=stride)
        self.shortcut = not self.downsampled and None or nn.AvgPool2d(2)
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = Variable(
            torch.zeros(
                h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)
            ).float()
        ).cuda()
        h0 = torch.cat([h0, pad_zero], dim=1)

        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )


class ShakeBottleneckBlock(nn.Module):
    outchannel_ratio = 4

    def __init__(self, in_ch, out_ch, stride, p_shakedrop=1.0):
        super(ShakeBottleneckBlock, self).__init__()
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_ch, out_ch, stride=stride)
        self.shortcut = (
            not self.downsampled
            and None
            or nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
        )
        self.shake_drop = ShakeDrop(p_shakedrop)

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, (out_ch * 1), kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d((out_ch * 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                (out_ch * 1),
                out_ch * ShakeBottleneckBlock.outchannel_ratio,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch * ShakeBottleneckBlock.outchannel_ratio),
        )

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = Variable(
            torch.zeros(
                h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)
            ).float()
        ).cuda()
        h0 = torch.cat([h0, pad_zero], dim=1)
        return h + h0



class ShakePyramidNet(nn.Module):
    def __init__(self, depth=110, alpha=270, label=10, bottleneck=False):
        super(ShakePyramidNet, self).__init__()
        in_ch = 16

        self.bottleneck = bottleneck
        if self.bottleneck == True:
            n_units = int((depth - 2) / 9)
            block = ShakeBottleneckBlock
            outchannel_ratio = 4
            in_chs = [in_ch] + [
                (in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)))
                * outchannel_ratio
                for i in range(3 * n_units)
            ]
            chs = [in_ch] + [
                (in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)))
                for i in range(3 * n_units)
            ]
            self.in_chs, self.chs, self.u_idx = in_chs, chs, 0
        else:
            n_units = int((depth - 2) / 6)
            block = ShakeBasicBlock
            in_chs = [in_ch] + [
                in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1))
                for i in range(3 * n_units)
            ]
            self.in_chs, self.u_idx = in_chs, 0

        self.ps_shakedrop = [
            1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)
        ]

        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, block, stride=1)
        self.layer2 = self._make_layer(n_units, block, stride=2)
        self.layer3 = self._make_layer(n_units, block, stride=2)
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], label)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.bn_in(self.c_in(x))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(self.bn_out(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, block, stride=1):
        layers = []
        if self.bottleneck == True:
            for i in range(int(n_units)):
                layers.append(
                    block(
                        self.in_chs[self.u_idx],
                        self.chs[self.u_idx + 1],
                        stride,
                        self.ps_shakedrop[self.u_idx],
                    )
                )
                self.u_idx, stride = self.u_idx + 1, 1
        else:
            for i in range(int(n_units)):
                layers.append(
                    block(
                        self.in_chs[self.u_idx],
                        self.in_chs[self.u_idx + 1],
                        stride,
                        self.ps_shakedrop[self.u_idx],
                    )
                )
                self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)

    @staticmethod
    def get_classifiers():
        return [
            "shakepyramidnet",
            "shakepyramidnet2",
            "shakepyramidnet3",
            "shakepyramidnet_bottle",
            "shakepyramidnet_bottle2",
            "shakepyramidnet_bottle3",
            "shakepyramidnet_bottle4",
        ]

    @classmethod
    def build_classifier(cls, arch: str, num_classes: int, input_channels: int):

        CONFIG = {
            "shakepyramidnet": {"depth": 110, "alpha": 270, "bottleneck": False},
            "shakepyramidnet2": {"depth": 110, "alpha": 84, "bottleneck": False},
            "shakepyramidnet3": {"depth": 110, "alpha": 48, "bottleneck": False},
            "shakepyramidnet_bottle": {"depth": 272, "alpha": 200, "bottleneck": True},
            "shakepyramidnet_bottle2": {"depth": 164, "alpha": 270, "bottleneck": True},
            "shakepyramidnet_bottle3": {"depth": 200, "alpha": 236, "bottleneck": True},
            "shakepyramidnet_bottle4": {"depth": 236, "alpha": 220, "bottleneck": True},
        }
        cls_instance = cls(
            depth=CONFIG[arch]["depth"],
            alpha=CONFIG[arch]["alpha"],
            bottleneck=CONFIG[arch]["bottleneck"],
        )
        return cls_instance
