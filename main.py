import argparse
import os
import csv
import cv2
import yaml
import torch
import warnings
import numpy as np
from nets import nn
from tqdm import tqdm
from utils import util
from pathlib import Path
from copy import deepcopy
from utils.dataset import Dataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


def train(args, params):
    util.init_seeds()

    # Model
    model = nn.get_yolo_model(args.model_type, len(params['names'].values())).cuda()
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    nl = model.detect.num_class
    params["box"] *= 3 / nl
    params["obj"] *= (args.input_size / 640) ** 2 * 3 / nl
    params["cls"] *= len(params['names'].values()) / 80 * 3 / nl

    # Optimizer
    accumulate = max(round(64 / args.batch_size), 1)  # 64 is nominal batch size
    params["decay"] *= args.batch_size * accumulate / 64
    optimizer = util.smart_optimizer(model, params)

    # Scheduler
    def lf(x):
        return (1 - x / args.epochs) * (1.0 - params["lrf"]) + params["lrf"]

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    # Data Loader
    dataset = Dataset(args, params, True, True)
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, args.batch_size,
                        not args.distributed, sampler,
                        num_workers=8, pin_memory=True,
                        collate_fn=Dataset.collate_fn)

    util.check_anchors(args, dataset, model)
    model.half().float()

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start Training ....
    best = 0
    opt_step = -1
    num_batch = len(loader)
    scaler = torch.cuda.amp.GradScaler()
    num_warmup = max(round(3 * num_batch), 100)
    criterion = util.ComputeLoss(model)
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            m_loss = torch.zeros(3, device='cuda')

            if args.distributed:
                sampler.set_epoch(epoch)

            pbar = enumerate(loader)

            print(("\n" + "%11s" * 5) % ("Epoch", "GPU", "box_loss", "obj_loss", "cls_loss"))
            if args.local_rank == 0:
                pbar = tqdm(pbar, total=num_batch)
            optimizer.zero_grad()

            for i, (images, labels, _, _) in pbar:
                num_i = i + num_batch * epoch
                images = images.cuda().float() / 255
                labels = labels.cuda()

                # Warm up
                if num_i <= num_warmup:
                    xi = [0, num_warmup]  # x interp
                    accumulate = max(1, np.interp(num_i, xi, [1, 64 / args.batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(num_i, xi, [0.1 if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(num_i, xi, [0.8, 0.937])

                with torch.cuda.amp.autocast():
                    pred = model(images)
                    loss, loss_items = criterion(pred, labels)

                if args.distributed:
                    loss *= args.world_size

                scaler.scale(loss).backward()

                if num_i - opt_step >= accumulate:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                    opt_step = num_i

                if args.local_rank == 0:
                    m_loss = (m_loss * i + loss_items) / (i + 1)
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                    pbar.set_description(("%11s" * 2 + "%11.4g" * 3) % (f"{epoch}/{args.epochs - 1}", mem, *m_loss))

            scheduler.step()

            if args.local_rank == 0:
                # result && # weight combination
                last = test(args, params, ema.ema)
                writer.writerow({'mAP': str(f'{last[1]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'mAP@50': str(f'{last[0]:.3f}')})

                if last[1] > best:
                    best = last[1]

                ckpt = {"model": deepcopy(ema.ema).half()}

                torch.save(ckpt, 'weights/last.pt')
                if best == last[1]:
                    torch.save(ckpt, 'weights/best.pt')
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')
        util.strip_optimizer('./weights/last.pt')

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None, plot=False):
    if model is None:
        model = torch.load('weights/models/yolov5l.pt', map_location='cuda')['model']
    model.half()
    model.eval()

    dataset = Dataset(args, params, False, False)
    loader = DataLoader(dataset, 32, False, num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    seen = 0
    confusion_matrix = util.ConfusionMatrix(args)
    pbar = tqdm(loader, desc=("%11s" * 4) % ("P", "R", "mAP50", "mAP50-95"))

    iou_v = torch.linspace(0.5, 0.95, 10).cuda()
    n_iou = iou_v.numel()

    stats = []
    m_pre = 0.0
    m_rec = 0.0
    map50 = 0.0
    mean_ap = 0.0
    for images, labels, paths, shapes in pbar:
        images = images.cuda()
        labels = labels.cuda()

        images = images.half()  # uint8 to fp16/32
        images = images / 255.0

        _, _, h, w = images.shape

        outputs = model(images)

        labels[:, 2:] *= torch.tensor((w, h, w, h), device='cuda')

        outputs = util.non_max_suppression(outputs)

        for i, output in enumerate(outputs):
            targets = labels[labels[:, 0] == i, 1:]
            num_labels, num_pred = targets.shape[0], output.shape[0]
            path, shape = Path(paths[i]), shapes[i][0]
            correct = torch.zeros(num_pred, n_iou, dtype=torch.bool, device='cuda')
            seen += 1

            clone_output = output.clone()
            util.scale_boxes(images[i].shape[1:], clone_output[:, :4], shape, shapes[i][1])

            if num_labels:
                target_box = util.xywh2xyxy(targets[:, 1:5])
                util.scale_boxes(images[i].shape[1:], target_box, shape, shapes[i][1])
                label_sn = torch.cat((targets[:, 0:1], target_box), 1)
                correct = util.process_batch(clone_output, label_sn, iou_v)
                confusion_matrix.process_batch(clone_output, label_sn)
            stats.append((correct, output[:, 4], output[:, 5], targets[:, 0]))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util._compute_ap(*stats)

    if plot:
        confusion_matrix.plot(save_dir='weights', names=list(params['names'].values()))

    print('%11.3g' * 4 % (m_pre, m_rec, map50, mean_ap))

    model.float()  # training
    return map50, mean_ap


@torch.no_grad()
def demo(args, params):
    model = torch.load('./weights/models/yolov5l.pt', 'cuda')['model'].float()
    model.half()
    model.eval()

    camera = cv2.VideoCapture('res2.mp4')

    # Get video properties
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter('output2.mp4', fourcc, fps, (width, height))

    if not camera.isOpened():
        print("Error opening video stream or file")

    while camera.isOpened():
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]

            r = args.input_size / max(shape[0], shape[1])
            if r != 1:
                resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
            height, width = image.shape[:2]

            # Scale ratio (new / old)
            r = min(1.0, args.input_size / height, args.input_size / width)

            # Compute padding
            pad = int(round(width * r)), int(round(height * r))
            w = (args.input_size - pad[0]) / 2
            h = (args.input_size - pad[1]) / 2

            if (width, height) != pad:  # resize
                image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

            # Convert HWC to CHW, BGR to RGB
            x = image.transpose((2, 0, 1))[::-1]
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            x = x.cuda()
            x = x.half()
            x = x / 255
            # Inference
            outputs = model(x)
            # NMS
            outputs = util.non_max_suppression(outputs, 0.35, 0.7)[0]

            if outputs is not None:
                outputs[:, [0, 2]] -= w
                outputs[:, [1, 3]] -= h
                outputs[:, :4] /= min(height / shape[0], width / shape[1])

                outputs[:, 0].clamp_(0, shape[1])
                outputs[:, 1].clamp_(0, shape[0])
                outputs[:, 2].clamp_(0, shape[1])
                outputs[:, 3].clamp_(0, shape[0])

                for box in outputs:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    class_name = params['names'][int(index)]
                    label = f"{class_name} {score:.2f}"
                    util.box_label(frame, box, index, label)

            cv2.imshow('Frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    camera.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../Datasets/COCO/', type=str)
    parser.add_argument('--model-type', default='x', type=str)
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--num-cls', default=80, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', default=True, action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open(os.path.join('utils', 'data.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args, params)


if __name__ == "__main__":
    main()
