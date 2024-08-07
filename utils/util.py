import cv2
import copy
import math
import time
import torch
import random
import warnings
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_optimizer(f='best.pt'):
    x = torch.load(f, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f)


def smart_optimizer(model, params):
    g = [], [], []
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)

    optimizer = torch.optim.SGD(g[2], lr=0.01, momentum=0.937, nesterov=True)
    optimizer.add_param_group({"params": g[0], "weight_decay": params['decay']})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer


def check_anchors(args, dataset, model):
    """Evaluates anchor fit to dataset and adjusts if necessary, supporting customizable threshold and image size."""
    shapes = args.input_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        best = x.max(1)[0]
        a = (x > 1 / 4).float().sum(1).mean()
        b = (best > 1 / 4).float().mean()
        return b, a

    m = model.detect
    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)
    anchors = m.anchors.clone() * stride
    b, a = metric(anchors.cpu().view(-1, 2))


def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel,
                           torch.nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class EMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        self.copy_attr(self.ema, model, include, exclude)

    @staticmethod
    def copy_attr(a, b, include=(), exclude=()):
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            else:
                setattr(a, k, v)


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model):
        device = next(model.parameters()).device
        m = de_parallel(model).detect
        self.na = m.num_anchor
        self.nc = m.num_class
        self.nl = m.num_layer
        self.anchors = m.anchors
        self.device = device

        self.cp, self.cn = self.smooth_bce(eps=0.0)
        bce_cls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        bce_obj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.num_layer, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.bce_cls, self.bce_obj, self.gr = bce_cls, bce_obj, 1.0

    def __call__(self, p, targets):
        cls_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        obj_loss = torch.zeros(1, device=self.device)
        target_cls, target_box, indices, anchors = self.build_targets(p, targets)

        # Losses
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            target_obj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            num_targets = b.shape[0]
            if num_targets:
                pxy, pwh, _, pred_cls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = self.bbox_iou(pbox, target_box[i], CIoU=True).squeeze()  # iou(prediction, target)
                box_loss += (1.0 - iou).mean()  # iou loss

                iou = iou.detach().clamp(0).type(target_obj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                target_obj[b, a, gj, gi] = iou  # iou ratio

                if self.nc > 1:
                    target = torch.full_like(pred_cls, self.cn, device=self.device)
                    target[range(num_targets), target_cls[i]] = self.cp
                    cls_loss += self.bce_cls(pred_cls, target)

            obj_i = self.bce_obj(pi[..., 4], target_obj)
            obj_loss += obj_i * self.balance[i]

        box_loss *= 0.05
        obj_loss *= 1.0
        cls_loss *= 0.5
        bs = target_obj.shape[0]

        return (box_loss + obj_loss + cls_loss) * bs, torch.cat((box_loss, obj_loss, cls_loss)).detach()

    @staticmethod
    def bbox_iou(box1, box2, CIoU=True, eps=1e-7):
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
                b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
        if CIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
            if CIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                        b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if CIoU:
                    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)
                return iou - rho2 / c2
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        return iou

    def build_targets(self, p, targets):
        g = 0.5
        targets = targets.to(self.device)
        na, nt = self.na, targets.shape[0]
        t_cls, tbox, indices, anc = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        off = (torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], ], device=self.device, ).float() * g)

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < 4.0
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anc.append(anchors[a])  # anchors
            t_cls.append(c)  # class

        return t_cls, tbox, indices, anc

    @staticmethod
    def smooth_bce(eps=0.1):
        return 1.0 - 0.5 * eps, 0.5 * eps


# ------------------------------------------ helper functions for dataset loading--------------------------------------
class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        prefix = self.colorstr("albumentations: ")
        try:
            import albumentations as A

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, im, labels, p=1.0):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels

    @staticmethod
    def colorstr(*input):
        *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
        colors = {
            "black": "\033[30m",  # basic colors
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_black": "\033[90m",  # bright colors
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
            "end": "\033[0m",  # misc
            "bold": "\033[1m",
            "underline": "\033[4m",
        }
        return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def augment_hsv(image, params):
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']
    if h or s or v:
        r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)


def candidates(box1, box2, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + eps) > 0.1) & (ar < 100)  # candidates


def xyxy2xywhn(x, w=640, h=640, eps=1e-3):
    clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywhn2xyxy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + pad_w
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + pad_h
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + pad_w
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + pad_h
    return y


def mix_up(im, labels, im2, labels2):
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def letterbox(im, new_shape=(640, 640), scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, ratio, (dw, dh)


def random_perspective(samples, targets, input_size, params):
    # Center
    center = np.eye(3)
    center[0, 2] = -float(input_size)  # x translation (pixels)
    center[1, 2] = -float(input_size)  # y translation (pixels)

    # Perspective
    perspective = np.eye(3)

    # Rotation and Scale
    rotation = np.eye(3)
    a = random.uniform(-params['degree'], params['degree'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

    # Shear
    p = params['shear']
    shear = np.eye(3)
    shear[0, 1] = math.tan(random.uniform(-p, p) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-p, p) * math.pi / 180)  # y shear (deg)

    # Translation
    p = params['translate']
    translation = np.eye(3)
    translation[0, 2] = random.uniform(0.5 - p, 0.5 + p) * input_size  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - p, 0.5 + p) * input_size  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    # image changed
    samples = cv2.warpAffine(samples, matrix[:2], dsize=(input_size, input_size))

    n = len(targets)
    if n:
        box = np.ones((n * 4, 3))
        box[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        box = box @ matrix.T  # transform
        box = box[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = box[:, [0, 2, 4, 6]]
        y = box[:, [1, 3, 5, 7]]
        box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, input_size)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, input_size)

        # filter candidates
        i = candidates(targets[:, 1:5].T * s, box.T)
        targets = targets[i]
        targets[:, 1:5] = box[i]

    return samples, targets


# ------------------------------------------------- Validation ---------------------------------------

def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall, precision):
    m_rec = np.concatenate(([0.0], recall, [1.0]))
    m_pre = np.concatenate(([1.0], precision, [0.0]))

    m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

    method = "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, m_rec, m_pre), x)
    else:
        i = np.where(m_rec[1:] != m_rec[:-1])[0]
        ap = np.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])

    return ap, m_pre, m_rec


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def process_batch(detections, labels, iou_v):
    correct = np.zeros((detections.shape[0], iou_v.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iou_v)):
        x = torch.where((iou >= iou_v[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iou_v.device)


def _compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

            # Integrate area under curve
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def non_max_suppression(prediction, conf_threshold=0.001, iou_threshold=0.6):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_threshold

    # Settings
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        # Detections matrix nx6 (xyxy, conf, cls)
        if nc > 1:  # multiple labels per box (adds 0.5ms/img)
            i, j = (x[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if nc == 1 else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > 0.3 + 0.03 * prediction.shape[0]:
            break  # time limit exceeded

    return output


def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="grey")

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


class ConfusionMatrix:
    def __init__(self, args):
        self.matrix = np.zeros((args.num_cls + 1, args.num_cls + 1))
        self.nc = args.num_cls
        self.conf = 0.25
        self.iou_thres = 0.45

    def process_batch(self, detections, labels):
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return tp[:-1], fp[:-1]

    def plot(self, normalize=True, save_dir="", names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(fig)

    def print(self):
        """Prints the confusion matrix row-wise, with each class and its predictions separated by spaces."""
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


class Colors:
    def __init__(self):
        hexs = (
            "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD", "FF444F",
            "CCED00", "00F344", "BD00FF", "00B4FF", "DD00BA", "00FFFF", "26C000",
            "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B"
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
            [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def box_label(im, box, index, label=""):
    colors = Colors()
    color = colors(index, True)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, 2, cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, 1, 2)[0]
        h += 3
        outside = p1[1] >= h
        if p1[0] > im.shape[1] - w:
            p1 = im.shape[1] - w, p1[1]
        p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)

        cv2.line(im, (int(int(box[0])), int(int(box[1]))), (int(int(box[0]) + 15), int(int(box[1]))), (0, 255, 255), 3)
        cv2.line(im, (int(int(box[0])), int(int(box[1]))), (int(int(box[0])), int(int(box[1]) + 15)), (0, 255, 255), 3)

        cv2.line(im, (int(int(box[2])), int(int(box[3]))), (int(int(box[2]) - 15), int(int(box[3]))), (0, 255, 255), 3)
        cv2.line(im, (int(int(box[2])), int(int(box[3]))), (int(int(box[2])), int(int(box[3]) - 15)), (0, 255, 255), 3)

        cv2.line(im, (int(int(box[2]) - 15), int(int(box[1]))), (int(int(box[2])), int(int(box[1]))), (0, 255, 255), 3)
        cv2.line(im, (int(int(box[2])), int(int(box[1]))), (int(int(box[2])), int(int(box[1]) + 15)), (0, 255, 255), 3)

        cv2.line(im, (int(int(box[0])), int(int(box[3]) - 15)), (int(int(box[0])), int(int(box[3]))), (0, 255, 255), 3)
        cv2.line(im, (int(int(box[0])), int(int(box[3]))), (int(int(box[0]) + 15), int(int(box[3]))), (0, 255, 255), 3)
        
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h - 1), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)

