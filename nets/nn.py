import math
import warnings

import torch
import torch.nn as nn

anchors = [[11, 12, 14, 32, 35, 24],
           [32, 61, 72, 56, 62, 141],
           [138, 109, 165, 243, 380, 334]]


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, inp, oup, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, self.pad(k, p), d, g, False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def pad(k, p):
        if p is None:
            p = k // 2
        return p


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.conv1 = Conv(inp, int(oup * e), 1, 1)
        self.conv2 = Conv(int(oup * e), oup, 3, 1, g=g)
        self.add = shortcut and inp == oup

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3(nn.Module):
    def __init__(self, inp, oup, n=1, shortcut=True, g=1):
        super().__init__()
        self.conv1 = Conv(inp, int(oup // 2))
        self.conv2 = Conv(inp, int(oup // 2))
        self.conv3 = Conv(inp=oup, oup=oup)
        self.m = nn.Sequential(*(Bottleneck(int(oup // 2), int(oup // 2), shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class SPPF(nn.Module):
    def __init__(self, inp, oup, k=5):
        super().__init__()
        self.conv1 = Conv(inp, inp // 2)
        self.conv2 = Conv(inp * 2, oup)
        self.n = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.n(x)
            y2 = self.n(y1)
            return self.conv2(torch.cat((x, y1, y2, self.n(y2)), 1))


class Backbone(nn.Module):
    def __init__(self, filters, depths):
        super().__init__()

        self.b1 = nn.Sequential(*[Conv(filters[0], filters[1], 6, 2, 2)])

        self.b2 = nn.Sequential(*[Conv(filters[1], filters[2], 3, 2, 1),
                                  C3(filters[2], filters[2], depths[0])])

        self.b3 = nn.Sequential(*[Conv(filters[2], filters[3], 3, 2, 1),
                                  C3(filters[3], filters[3], depths[1])])
        self.b4 = nn.Sequential(*[Conv(filters[3], filters[4], 3, 2, 1),
                                  C3(filters[4], filters[4], depths[2])])
        self.b5 = nn.Sequential(*[Conv(filters[4], filters[5], 3, 2, 1),
                                  C3(filters[5], filters[5], depths[0]),
                                  SPPF(filters[5], filters[5])])

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)

        return b3, b4, b5


class Head(nn.Module):
    def __init__(self, filters, depths):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.h1 = Conv(filters[5], filters[4])
        self.h2 = C3(filters[5], filters[4], depths[0], False)

        self.h3 = Conv(filters[4], filters[3])
        self.h4 = C3(filters[4], filters[3], depths[0], False)

        self.h5 = Conv(filters[3], filters[3], 3, 2)
        self.h6 = C3(filters[4], filters[4], depths[0], False)

        self.h7 = Conv(filters[4], filters[4], 3, 2)
        self.h8 = C3(filters[5], filters[5], depths[0], False)

    def forward(self, x):
        b3, b4, b5 = x

        h1 = self.h1(b5)
        h2 = self.h2(torch.cat([self.up(h1), b4], 1))

        h3 = self.h3(h2)
        h4 = self.h4(torch.cat([self.up(h3), b3], 1))

        h5 = self.h5(h4)
        h6 = self.h6(torch.cat([h5, h3], 1))

        h7 = self.h7(h6)
        h8 = self.h8(torch.cat([h7, h1], 1))
        return h4, h6, h8


class Detect(nn.Module):
    stride = None

    def __init__(self, filters, num_class):
        super().__init__()
        self.num_class = num_class
        self.num_layer = len(anchors)
        self.num_output = num_class + 5
        self.num_anchor = len(anchors[0]) // 2

        self.grid = [torch.empty(0) for _ in range(self.num_layer)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_layer)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.num_layer, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.num_output * self.num_anchor, 1) for x in filters)

    def forward(self, x):
        z = []
        for i in range(self.num_layer):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchor, self.num_output, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_class + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.num_anchor * nx * ny, self.num_output))

        return x if self.training else torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.num_anchor, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.num_anchor, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Yolo(nn.Module):
    def __init__(self, filters, depths, num_class):
        super().__init__()

        img_dummy = torch.zeros(1, 3, 256, 256)

        self.backbone = Backbone(filters, depths)
        self.head = Head(filters, depths)
        self.detect = Detect((filters[3], filters[4], filters[5]), num_class)

        self.detect.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.check_anchor_order(self.detect)
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self.stride = self.detect.stride
        self._initialize_biases()
        self.initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.detect(list(x))

    def _initialize_biases(self):
        for m, s in zip(self.detect.m, self.detect.stride):
            b = m.bias.view(self.detect.num_anchor, -1).clone()
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (self.detect.num_class - 0.99))
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

    @staticmethod
    def check_anchor_order(m):
        a = m.anchors.prod(-1).mean(-1).view(-1)
        da = a[-1] - a[0]  # delta a
        ds = m.stride[-1] - m.stride[0]  # delta s
        if da and (da.sign() != ds.sign()):  # same order
            m.anchors[:] = m.anchors.flip(0)

    @staticmethod
    def initialize_weights(model):
        for m in model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True


def yolo_v5_n(num_class=80):  # Num Params: 1872157
    depth = [1, 2, 3]
    width = [3, 16, 32, 64, 128, 256]
    return Yolo(width, depth, num_class)


def yolo_v5_s(num_class=80):  # Num Params: 7235389
    depth = [1, 2, 3]
    width = [3, 32, 64, 128, 256, 512]
    return Yolo(width, depth, num_class)


def yolo_v5_m(num_class: int = 80):  # Num Params: 21190557
    depth = [2, 4, 6]
    width = [3, 48, 96, 192, 384, 768]
    return Yolo(width, depth, num_class)


def yolo_v5_l(num_class: int = 80):  # Num Params: 46563709
    depth = [3, 6, 9]
    width = [3, 64, 128, 256, 512, 1024]
    return Yolo(width, depth, num_class)


def yolo_v5_x(num_class: int = 80):  # Num Params: 86749405
    depth = [4, 8, 12]
    width = [3, 80, 160, 320, 640, 1280]
    return Yolo(width, depth, num_class)


def get_yolo_model(model_type, num_class=80):
    model_dict = {
        'n': yolo_v5_n,
        's': yolo_v5_s,
        'm': yolo_v5_m,
        'l': yolo_v5_l,
        'x': yolo_v5_x
    }

    if model_type in model_dict:
        return model_dict[model_type](num_class)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Valid types are 'n', 's', 'm', 'l', 'x'.")
