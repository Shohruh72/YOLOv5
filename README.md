# YOLOv5 implementation using PyTorch


# Demo

https://github.com/user-attachments/assets/b41621ae-ebdf-453c-b2bc-9ffe01a14d08


https://github.com/user-attachments/assets/1b5d4323-64ee-4514-95f0-6a1080672753


### Installation

```
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Inference

* Run `python main.py --demo` for inference

### Results & Pretrained Checkpoints

| Model                                                                                | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|--------------------------------------------------------------------------------------|----------------------|-------------------|--------------------|------------------------|
| [YOLOv5n](https://github.com/Shohruh72/YOLOv5/releases/download/v.1.0.0/yolov5n.pt) | 28.0                 | 45.7              | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/Shohruh72/YOLOv5/releases/download/v.1.0.0/yolov5s.pt) | 37.4                 | 56.8              | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/Shohruh72/YOLOv5/releases/download/v.1.0.0/yolov5m.pt) | 45.4                 | 64.1              | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/Shohruh72/YOLOv5/releases/download/v.1.0.0/yolov5l.pt) | 49.0                 | 67.3              | 46.5               | 109.1                  | 50.7                 | 68.9              | 86.7               | 205.7                  |

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/ultralytics/yolov5
