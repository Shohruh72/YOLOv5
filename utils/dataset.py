import os
from PIL import Image
from utils import util
from utils.util import *
from torch.utils import data

FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"


class Dataset(data.Dataset):
    def __init__(self, args, params, augment=False, is_train=True):
        self.args = args
        self.params = params
        self.augment = augment
        self.is_train = is_train
        self.mosaic = self.augment
        self.album = util.Albumentations()

        self.root = os.path.join(self.args.root, f'{"train" if is_train else "val"}2017.txt')

        self._load_data()

        labels, shapes = zip(*self.cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.image_path = list(self.cache.keys())

        self.num_images = len(self.shapes)
        self.indices = np.arange(self.num_images)

        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / self.args.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)

        if not self.is_train:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i_rect = ar.argsort()
            self.image_path = [self.image_path[i] for i in i_rect]
            self.label_files = [self.label_path[i] for i in i_rect]
            self.labels = [self.labels[i] for i in i_rect]
            self.shapes = s[i_rect]  # wh
            ar = ar[i_rect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * self.args.input_size / 32 + 0.5).astype(int) * 32

    def __getitem__(self, index):
        index = self.indices[index]
        mosaic = self.mosaic and random.random() < self.params["mosaic"]

        if mosaic:
            image, labels = self.load_mosaic(index)
            shapes = None
            if random.random() < self.params["mix_up"]:
                images, labels = mix_up(image, labels, *self.load_mosaic(random.choice(self.indices)))
        else:
            image, (h0, w0), (h, w) = self._load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if not self.is_train else self.args.input_size
            image, ratio, pad = letterbox(image, shape, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, pad_w=pad[0], pad_h=pad[1])

            if self.augment:
                image, label = random_perspective(image, labels, self.args.input_size, self.params)

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=image.shape[1], h=image.shape[0])

        if self.augment:
            # Albumentations
            image, labels = self.album(image, labels)
            nl = len(labels)
            augment_hsv(image, self.params)
            # Flip up-down
            if random.random() < self.params["flip_ud"]:
                image = np.flipud(image)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < self.params["flip_lr"]:
                image = np.fliplr(image)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image), labels_out, self.image_path[index], shapes

    def __len__(self):
        return len(self.image_path)

    def _load_data(self):
        samples = []
        self.cache = {}
        with open(Path(self.root)) as file:
            file = file.read().strip().splitlines()
            parent = str(Path(self.root).parent) + os.sep
            samples += [f.replace("./", parent, 1) if f.startswith("./") else f for f in file]

        self.image_path = sorted(f.replace("/", os.sep)
                                 for f in samples if f.lower().endswith(tuple(FORMATS)))
        self.label_path = self.img2label(self.image_path)

        cache_path = Path(self.root).with_suffix(".cache")
        try:
            self.cache = np.load(cache_path, allow_pickle=True).item()
        except FileNotFoundError:
            for img, lbl in zip(self.image_path, self.label_path):
                try:
                    image = Image.open(img)
                    image.verify()
                    shape = image.size
                    assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                    assert image.format.lower() in FORMATS, f"invalid image format {image.format}"

                    if os.path.isfile(lbl):
                        with open(lbl) as f:
                            label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                            label = np.array(label, dtype=np.float32)
                        nl = len(label)
                        if nl:
                            assert label.shape[1] == 5, f"labels require 5 columns, {label.shape[1]} columns detected"
                            assert (label >= 0).all(), f"negative label values {label[label < 0]}"
                            assert (label[:, 1:] <= 1).all(), f"Out of bounds coords {label[:, 1:][label[:, 1:] > 1]}"
                            _, i = np.unique(label, axis=0, return_index=True)
                            if len(i) < nl:  # duplicate row check
                                label = label[i]  # remove duplicates
                        else:
                            label = np.zeros((0, 5), dtype=np.float32)
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)

                    if img:
                        self.cache[img] = [label, shape]
                except Exception as e:
                    print(f"Skipping file {img} due to error: {e}")

            try:
                np.save(cache_path, self.cache)
                cache_path.with_suffix(".cache.npy").rename(cache_path)
            except Exception as e:
                raise RuntimeError(f"Cache directory {cache_path.parent} is not writable: {e}")

    def _load_image(self, index):
        image = cv2.imread(self.image_path[index])
        height, weight = image.shape[:2]
        ratio = self.args.input_size / max(height, weight)
        if ratio != 1:
            interp = cv2.INTER_LINEAR if (self.augment or ratio > 1) else cv2.INTER_AREA
            image = cv2.resize(image, (math.ceil(weight * ratio), math.ceil(height * ratio)), interpolation=interp)
        return image, (height, weight), image.shape[:2]

    def load_mosaic(self, index):
        """Loads a 4-image mosaic for YOLOv5, combining 1 selected and 3 random images, with labels and segments."""
        labels4 = []
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None, None, None, None, None, None, None, None)
        xc = int(random.uniform(self.args.input_size // 2, 2 * self.args.input_size - self.args.input_size // 2))
        yc = int(random.uniform(self.args.input_size // 2, 2 * self.args.input_size - self.args.input_size // 2))
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            img, _, (h, w) = self._load_image(index)

            if i == 0:  # top left
                img4 = np.full((self.args.input_size * 2, self.args.input_size * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.args.input_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.args.input_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.args.input_size * 2), min(self.args.input_size * 2,
                                                                                        yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            labels = self.labels[index].copy()
            if len(labels):
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, pad_w, pad_h)
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:]):
            np.clip(x, 0, 2 * self.args.input_size, out=x)

        # Augment
        img4, labels4 = random_perspective(img4, labels4, self.args.input_size, self.params)

        return img4, labels4

    @staticmethod
    def img2label(img_path):
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_path]

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes
